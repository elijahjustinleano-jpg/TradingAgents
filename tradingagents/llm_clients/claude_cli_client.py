"""Claude through the local Claude Code CLI, authenticated by the user's subscription.

The CLI (``claude -p``) has no function-calling API. Tool calls and structured
output are emulated: the transcript ends with a JSON contract, and a parsed
object becomes ``AIMessage.tool_calls``. LangChain's ``with_structured_output``
already turns one bound tool into a schema object, and the agents already drop
back to free text when that parse fails.

``--bare`` is intentionally not used. It forces API-key auth and never reads
the subscription login.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)

# Claude Code aliases. A full model id the CLI accepts (claude-opus-4-6, ...)
# also works; the catalog offers these three plus a custom id.
_DEFAULT_TIMEOUT = 900.0  # Opus at high effort is slow; one call, not the whole run.
_DEFAULT_RETRIES = 2
_DEFAULT_CONCURRENCY = 2
_EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Removed so the CLI cannot bill an API key instead of the subscription login.
_STRIPPED_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
# CLAUDE_CODE_SIMPLE would turn Bash and file edits back on.
_FORCED_ENV = {
    "CLAUDE_CODE_SAFE_MODE": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
    "CLAUDE_CODE_DISABLE_WEB_FETCH": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_AUTOUPDATER": "1",
}

_SYSTEM_PROMPT = (
    "You are answering a single turn for a research application. "
    "Do not use Claude Code tools, edit files, run commands, or browse. "
    "The transcript is the full conversation: honor system messages as instructions "
    "and use prior assistant tool calls and tool results as the history. "
    "If an output contract ends the transcript, obey it and reply with only that JSON."
)

_ROLES = {"system": "system", "human": "human", "ai": "assistant", "tool": "tool"}

_semaphores: dict[int, threading.BoundedSemaphore] = {}
_semaphores_lock = threading.Lock()


class ClaudeCLIError(RuntimeError):
    """The Claude Code CLI failed. Messages are scrubbed of credentials."""


class ClaudeCLIConfigError(ClaudeCLIError):
    """A permanent setup failure (missing binary). Not retried."""


class ClaudeCLITimeout(ClaudeCLIError):
    """The CLI exceeded the per-call timeout. Not retried; raise the timeout instead."""


class _ProtocolError(ClaudeCLIError):
    """The model attempted the tool JSON contract but the object did not parse."""


def _check_effort(effort: str | None) -> str | None:
    if effort is None or effort == "":
        return None
    if effort not in _EFFORTS:
        allowed = ", ".join(_EFFORTS)
        raise ValueError(
            f"claude_cli effort must be one of {allowed}, got {effort!r}. "
            "Set TRADINGAGENTS_ANTHROPIC_EFFORT."
        )
    return effort


def _scrub(text: str) -> str:
    """Drop credential-shaped substrings before anything is logged or raised."""
    text = re.sub(r"sk-ant-[A-Za-z0-9_\-]+", "[redacted]", text)
    text = re.sub(
        r"(?i)((?:api[_-]?key|auth(?:orization)?|token|secret|password)\s*[:=]\s*)\S+",
        r"\1[redacted]",
        text,
    )
    return text


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _semaphore_for(limit: int) -> threading.BoundedSemaphore:
    with _semaphores_lock:
        sem = _semaphores.get(limit)
        if sem is None:
            sem = threading.BoundedSemaphore(limit)
            _semaphores[limit] = sem
        return sem


def _sandbox_cwd() -> str:
    path = Path(tempfile.gettempdir()) / "tradingagents-claude-cli"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return str(path)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in (*_STRIPPED_ENV, "CLAUDE_CODE_SIMPLE"):
        env.pop(key, None)
    env.update(_FORCED_ENV)
    return env


def _kill_process(proc: subprocess.Popen) -> None:
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    proc.kill()


def _run_process(
    cmd: list[str], prompt: str, env: dict[str, str], cwd: str, timeout: float,
) -> tuple[int, str, str]:
    """Run ``claude``. The prompt is stdin, never an argument (length and ``ps``)."""
    popen_kwargs: dict[str, Any] = {}
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        raise ClaudeCLIConfigError(
            f"Claude Code CLI executable {cmd[0]!r} was not found on PATH. "
            "Install Claude Code 2.1 or newer and log in with `claude` "
            "(a subscription login, not an API key)."
        ) from exc
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process(proc)
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        raise ClaudeCLITimeout(f"claude timed out after {timeout:g}s") from exc
    return int(proc.returncode), stdout or "", stderr or ""


def _loads_envelope(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        raise ClaudeCLIError("claude returned empty output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data
    decoder = json.JSONDecoder()
    found = None
    start = 0
    while True:
        brace = text.find("{", start)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError:
            start = brace + 1
            continue
        if isinstance(obj, dict) and (obj.get("type") == "result" or "result" in obj):
            found = obj
        start = brace + end
    if found is None:
        raise ClaudeCLIError("claude output was not a JSON result envelope")
    return found


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in (None, "text") and "text" in block:
                    parts.append(str(block.get("text") or ""))
                elif "text" in block:
                    parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    return str(content)


def _render_messages(messages: list[BaseMessage]) -> str:
    blocks: list[str] = []
    for message in messages:
        role = _ROLES.get(getattr(message, "type", ""), getattr(message, "type", "message"))
        body = _text_content(message.content)
        if isinstance(message, ToolMessage):
            header = (
                f"----- MESSAGE tool id={message.tool_call_id} "
                f"name={message.name or ''} -----"
            )
        else:
            header = f"----- MESSAGE {role} -----"
        extra = ""
        tool_calls = getattr(message, "tool_calls", None) or []
        if role == "assistant" and tool_calls:
            rendered = []
            for call in tool_calls:
                rendered.append(
                    "<tool_call "
                    f"id=\"{call.get('id', '')}\" name=\"{call.get('name', '')}\">\n"
                    f"{json.dumps(call.get('args') or {}, ensure_ascii=False, default=str)}\n"
                    "</tool_call>"
                )
            extra = "\n" + "\n".join(rendered)
        blocks.append(f"{header}\n{body}{extra}\n----- END MESSAGE -----")
    return "\n\n".join(blocks)


def _tool_specs(tools: list[Any]) -> list[dict[str, Any]]:
    specs = []
    for tool in tools:
        converted = convert_to_openai_tool(tool)
        function = converted.get("function", converted)
        specs.append({
            "name": function.get("name"),
            "description": function.get("description") or "",
            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        })
    return specs


def _choice_clause(tool_choice: Any) -> str:
    if tool_choice in (None, "auto"):
        return (
            "When you can answer without a tool, set tool_calls to [] and put "
            "the full answer in content."
        )
    if tool_choice in ("any", "required"):
        return (
            "You must set tool_calls to one or more of the listed tools. "
            "Do not answer in content only."
        )
    name = None
    if isinstance(tool_choice, str) and tool_choice != "none":
        name = tool_choice
    elif isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
    if name:
        return f"You must call the tool named {name}."
    return ""


def _output_contract(tools: list[dict[str, Any]], tool_choice: Any) -> str:
    if not tools or tool_choice == "none":
        return ""
    clause = _choice_clause(tool_choice)
    listed = json.dumps(tools, ensure_ascii=False, indent=2)
    return (
        "----- OUTPUT CONTRACT -----\n"
        "Reply with one JSON object and nothing else. No markdown fences.\n"
        '{"content": "text the user should see", '
        '"tool_calls": [{"name": "tool name", "arguments": {}}]}\n'
        "tool_calls is an array. Each arguments value is a JSON object. "
        "You may include several tool calls.\n"
        f"{clause}\n"
        f"Tools:\n{listed}\n"
        "----- END OUTPUT CONTRACT -----"
    )


def _render_request(messages: list[BaseMessage], tools: list[Any], tool_choice: Any) -> str:
    specs = _tool_specs(tools) if tools and tool_choice != "none" else []
    parts = [_render_messages(messages)]
    contract = _output_contract(specs, tool_choice)
    if contract:
        parts.append(contract)
    return "\n\n".join(parts)


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    start = 0
    while True:
        brace = text.find("{", start)
        if brace < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError:
            start = brace + 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        start = brace + max(end, 1)
    return found


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _protocol_payload(text: str) -> dict[str, Any] | None:
    stripped = _strip_fence(text)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and ("tool_calls" in value or "content" in value):
        return value
    objects = _json_objects(text)
    with_calls = [obj for obj in objects if "tool_calls" in obj]
    if with_calls:
        return with_calls[-1]
    return None


def _coerce_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {"input": raw}
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    return {"input": raw}


def _promote_bare_object(text: str, tools: list[Any]) -> tuple[str, list[dict[str, Any]]] | None:
    """A single required tool often comes back as the schema object itself.

    ``with_structured_output`` binds exactly one tool and sets ``tool_choice``
    to ``any``. Models regularly emit ``{"rating": "Buy", ...}`` instead of
    wrapping it in the tool-call contract. That object is the tool call.
    """
    if len(tools) != 1:
        return None
    spec = _tool_specs(tools)[0]
    properties = set((spec.get("parameters") or {}).get("properties") or {})
    if not spec.get("name") or not properties:
        return None
    try:
        value = json.loads(_strip_fence(text))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or "tool_calls" in value:
        return None
    if not properties.intersection(value):
        return None
    return "", [{
        "name": str(spec["name"]),
        "args": value,
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "tool_call",
    }]


def _parse_tool_response(text: str) -> tuple[str, list[dict[str, Any]]]:
    attempted = '"tool_calls"' in text or "'tool_calls'" in text
    try:
        payload = _protocol_payload(text)
    except json.JSONDecodeError as exc:
        if attempted:
            raise _ProtocolError("tool JSON did not parse") from exc
        return text.strip(), []
    if payload is None:
        if attempted:
            raise _ProtocolError("tool JSON did not parse")
        return text.strip(), []
    calls_raw = payload.get("tool_calls", [])
    if calls_raw is None:
        calls_raw = []
    if not isinstance(calls_raw, list):
        raise _ProtocolError("tool_calls was not a list")
    content = payload.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    calls = []
    for raw in calls_raw:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        args = raw.get("arguments", raw.get("args", raw.get("parameters", {})))
        calls.append({
            "name": str(raw["name"]),
            "args": _coerce_args(args),
            "id": str(raw.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
            "type": "tool_call",
        })
    return content, calls


def _apply_stop(text: str, stop: list[str] | None) -> str:
    if not stop or not text:
        return text
    cut = len(text)
    for token in stop:
        if not token:
            continue
        found = text.find(token)
        if found != -1:
            cut = min(cut, found)
    return text[:cut]


def _usage(envelope: dict[str, Any]) -> dict[str, int] | None:
    """Token counts from the CLI envelope, in LangChain's usage_metadata shape."""
    raw = envelope.get("usage")
    if not isinstance(raw, dict):
        return None
    incoming = raw.get("input_tokens")
    outgoing = raw.get("output_tokens")
    if isinstance(incoming, bool) or isinstance(outgoing, bool):
        return None
    if not isinstance(incoming, int) or not isinstance(outgoing, int):
        return None
    return {
        "input_tokens": incoming,
        "output_tokens": outgoing,
        "total_tokens": incoming + outgoing,
    }


def _chat_result(
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
) -> ChatResult:
    message = AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=usage,
    )
    return ChatResult(generations=[ChatGeneration(message=message)])


class ChatClaudeCLI(BaseChatModel):
    """One Claude Code CLI process per completion, with emulated tool calls."""

    model: str
    cli_bin: str = "claude"
    effort: str | None = None
    timeout: float = _DEFAULT_TIMEOUT
    max_retries: int = _DEFAULT_RETRIES
    max_concurrency: int | None = None
    cwd: str | None = None

    @property
    def _llm_type(self) -> str:
        return "claude_cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        """Bind tools the way the analyst loop and structured-output path expect.

        ``with_structured_output`` calls this with the schema as the only tool
        and ``tool_choice="any"``, then reads ``.tool_calls`` on the reply.
        """
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=list(tools), **kwargs)

    def _concurrency_limit(self) -> int:
        if self.max_concurrency is not None:
            return max(1, int(self.max_concurrency))
        raw = os.environ.get("TRADINGAGENTS_CLAUDE_CLI_MAX_CONCURRENCY")
        if raw is None or raw.strip() == "":
            return _DEFAULT_CONCURRENCY
        try:
            limit = int(raw)
        except ValueError as exc:
            raise ValueError(
                "TRADINGAGENTS_CLAUDE_CLI_MAX_CONCURRENCY must be a positive integer, "
                f"got {raw!r}"
            ) from exc
        if limit < 1:
            raise ValueError(
                f"TRADINGAGENTS_CLAUDE_CLI_MAX_CONCURRENCY must be >= 1, got {limit}"
            )
        return limit

    def _cwd(self) -> str:
        if self.cwd:
            return self.cwd
        override = os.environ.get("TRADINGAGENTS_CLAUDE_CLI_CWD")
        if override:
            return override
        return _sandbox_cwd()

    def _command(self) -> list[str]:
        cmd = [
            self.cli_bin,
            "-p",
            "--output-format", "json",
            "--model", self.model,
            "--max-turns", "1",
            "--tools", "",
            "--permission-mode", "plan",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--system-prompt", _SYSTEM_PROMPT,
        ]
        effort = _check_effort(self.effort)
        if effort:
            cmd.extend(["--effort", effort])
        return cmd

    def _execute(self, prompt: str) -> tuple[str, dict[str, int] | None]:
        sem = _semaphore_for(self._concurrency_limit())
        sem.acquire()
        try:
            code, stdout, stderr = _run_process(
                self._command(), prompt, _subprocess_env(), self._cwd(), self.timeout,
            )
        finally:
            sem.release()
        if code != 0:
            detail = _scrub(_tail(stderr or stdout))
            raise ClaudeCLIError(f"claude exited {code}: {detail}")
        envelope = _loads_envelope(stdout)
        if envelope.get("is_error"):
            detail = envelope.get("result") or "claude reported an error"
            raise ClaudeCLIError(f"claude error: {_scrub(_tail(str(detail)))}")
        if "result" not in envelope or envelope.get("result") is None:
            raise ClaudeCLIError("claude JSON output had no result")
        result = envelope["result"]
        text = json.dumps(result) if isinstance(result, dict) else str(result)
        return text, _usage(envelope)

    def _backoff(self, attempt: int) -> None:
        time.sleep(min(2 ** attempt, 8))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = list(kwargs.get("tools") or [])
        tool_choice = kwargs.get("tool_choice")
        prompt = _render_request(messages, tools, tool_choice)
        wants_tools = bool(tools) and tool_choice != "none"
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                text, usage = self._execute(prompt)
            except (ClaudeCLITimeout, ClaudeCLIConfigError):
                raise
            except ClaudeCLIError:
                if attempt >= attempts:
                    raise
                logger.warning(
                    "claude_cli call failed (attempt %s/%s); retrying", attempt, attempts,
                )
                self._backoff(attempt)
                continue
            if not wants_tools:
                return _chat_result(_apply_stop(text, stop), usage=usage)
            try:
                content, calls = _parse_tool_response(text)
            except _ProtocolError:
                if attempt >= attempts:
                    logger.warning(
                        "claude_cli tool response was not valid JSON after %s attempts; "
                        "using the text as the answer",
                        attempts,
                    )
                    return _chat_result(_apply_stop(text, stop), usage=usage)
                logger.warning(
                    "claude_cli tool JSON did not parse (attempt %s/%s); retrying",
                    attempt, attempts,
                )
                self._backoff(attempt)
                continue
            if not calls and tool_choice in ("any", "required"):
                promoted = _promote_bare_object(text, tools)
                if promoted is not None:
                    content, calls = promoted
            return _chat_result(_apply_stop(content, stop), calls, usage)
        raise ClaudeCLIError("claude_cli exhausted retries")


class ClaudeCLIClient(BaseLLMClient):
    """Factory-facing client. No API key; the ``claude`` login on this machine is the credential."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        self.provider = "claude_cli"

    def get_llm(self) -> ChatClaudeCLI:
        self.warn_if_unknown_model()
        if self.base_url:
            warnings.warn(
                "claude_cli ignores backend_url; it runs the local claude CLI.",
                RuntimeWarning,
                stacklevel=2,
            )
        for ignored in ("temperature", "max_tokens"):
            if self.kwargs.get(ignored) not in (None, ""):
                warnings.warn(
                    f"claude_cli ignores {ignored}; the Claude Code CLI does not expose it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        effort = _check_effort(self.kwargs.get("effort"))
        timeout = self.kwargs.get("timeout")
        if timeout is None:
            raw = os.environ.get("TRADINGAGENTS_CLAUDE_CLI_TIMEOUT")
            timeout = float(raw) if raw else _DEFAULT_TIMEOUT
        max_retries = self.kwargs.get("max_retries")
        if max_retries is None:
            max_retries = _DEFAULT_RETRIES
        chat_kwargs: dict[str, Any] = {
            "model": self.model,
            "cli_bin": (
                self.kwargs.get("cli_bin")
                or os.environ.get("TRADINGAGENTS_CLAUDE_CLI_BIN")
                or "claude"
            ),
            "timeout": float(timeout),
            "max_retries": int(max_retries),
        }
        if effort:
            chat_kwargs["effort"] = effort
        if self.kwargs.get("max_concurrency") is not None:
            chat_kwargs["max_concurrency"] = int(self.kwargs["max_concurrency"])
        if self.kwargs.get("callbacks"):
            chat_kwargs["callbacks"] = self.kwargs["callbacks"]
        if self.kwargs.get("cwd"):
            chat_kwargs["cwd"] = self.kwargs["cwd"]
        return ChatClaudeCLI(**chat_kwargs)

    def validate_model(self) -> bool:
        """Any model string: Claude Code aliases and full ids both change over time."""
        return validate_model("claude_cli", self.model)
