"""claude_cli: subscription auth through the local Claude Code CLI, no API key.

The subprocess is mocked. A live call is opt-in (``pytest -m integration``) and
skipped unless ``claude`` is on PATH.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import threading
import time
import warnings
from typing import Literal
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from tradingagents.llm_clients import claude_cli_client as mod
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.claude_cli_client import (
    ChatClaudeCLI,
    ClaudeCLIConfigError,
    ClaudeCLIError,
)
from tradingagents.llm_clients.factory import build_llm_kwargs, create_llm_client
from tradingagents.llm_clients.openai_client import is_openai_compatible
from tradingagents.llm_clients.validators import validate_model


def _envelope(result, is_error=False) -> str:
    return json.dumps({
        "type": "result",
        "subtype": "error_during_execution" if is_error else "success",
        "is_error": is_error,
        "result": result,
    })


class _FakeCLI:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, cmd, prompt, env, cwd, timeout):
        self.calls.append({
            "cmd": list(cmd), "prompt": prompt, "env": dict(env), "cwd": cwd, "timeout": timeout,
        })
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, tuple):
            return reply
        return 0, _envelope(reply), ""


@pytest.fixture
def fake_cli(monkeypatch):
    def install(replies):
        fake = _FakeCLI(replies)
        monkeypatch.setattr(mod, "_run_process", fake)
        monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
        return fake

    return install


def _llm(**kwargs) -> ChatClaudeCLI:
    params = {"model": "opus", "timeout": 5, "max_retries": 0, "max_concurrency": 2}
    params.update(kwargs)
    return ChatClaudeCLI(**params)


_STOCK = {
    "type": "function",
    "function": {
        "name": "get_stock_data",
        "description": "Daily prices",
        "parameters": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
}
_INDICATORS = {
    "type": "function",
    "function": {
        "name": "get_indicators",
        "description": "Technical indicators",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "indicator": {"type": "string"},
            },
            "required": ["symbol", "indicator"],
        },
    },
}


def _tool_payload():
    return json.dumps({
        "content": "fetching",
        "tool_calls": [
            {"name": "get_stock_data", "arguments": {"symbol": "NVDA"}},
            {"name": "get_indicators", "arguments": {"symbol": "NVDA", "indicator": "rsi"}},
        ],
    })


@pytest.mark.unit
def test_plain_invoke_reads_the_json_envelope(fake_cli):
    fake = fake_cli(["NVDA looks extended."])
    message = _llm().invoke("What about NVDA?")
    assert message.content == "NVDA looks extended."
    assert message.tool_calls == []
    prompt = fake.calls[0]["prompt"]
    assert "What about NVDA?" in prompt
    assert "What about NVDA?" not in fake.calls[0]["cmd"]


@pytest.mark.unit
def test_message_history_includes_system_tool_calls_and_results(fake_cli):
    fake = fake_cli([json.dumps({"content": "Report body.", "tool_calls": []})])
    history = [
        SystemMessage(content="You are a market analyst."),
        HumanMessage(content="Analyze NVDA."),
        AIMessage(
            content="Looking up prices.",
            tool_calls=[{
                "name": "get_stock_data",
                "args": {"symbol": "NVDA"},
                "id": "call_abc",
                "type": "tool_call",
            }],
        ),
        ToolMessage(content="close=100", name="get_stock_data", tool_call_id="call_abc"),
        HumanMessage(content="Continue."),
    ]
    message = _llm().bind_tools([_STOCK]).invoke(history)

    prompt = fake.calls[0]["prompt"]
    assert "You are a market analyst." in prompt
    assert "Analyze NVDA." in prompt
    assert "Looking up prices." in prompt
    assert "call_abc" in prompt
    assert "get_stock_data" in prompt
    assert '"symbol": "NVDA"' in prompt
    assert "close=100" in prompt
    assert "Continue." in prompt
    assert "OUTPUT CONTRACT" in prompt
    assert message.content == "Report body."
    assert message.tool_calls == []


@pytest.mark.unit
def test_multiple_tool_calls_become_ai_message_tool_calls(fake_cli):
    fake_cli(["```json\n" + _tool_payload() + "\n```"])
    message = _llm().bind_tools([_STOCK, _INDICATORS]).invoke("Analyze NVDA.")

    assert [call["name"] for call in message.tool_calls] == ["get_stock_data", "get_indicators"]
    assert message.tool_calls[0]["args"] == {"symbol": "NVDA"}
    assert message.tool_calls[1]["args"] == {"symbol": "NVDA", "indicator": "rsi"}
    assert message.tool_calls[0]["type"] == "tool_call"
    assert message.tool_calls[0]["id"] != message.tool_calls[1]["id"]
    assert message.content == "fetching"


@pytest.mark.unit
def test_tool_arguments_may_be_a_json_string(fake_cli):
    fake_cli([json.dumps({
        "content": "",
        "tool_calls": [{
            "name": "get_stock_data",
            "arguments": "{\"symbol\": \"NVDA\"}",
        }],
    })])
    message = _llm().bind_tools([_STOCK]).invoke("go")
    assert message.tool_calls[0]["args"] == {"symbol": "NVDA"}


@pytest.mark.unit
def test_prose_while_tools_are_bound_is_the_answer(fake_cli):
    fake_cli(["The trend is up."])
    message = _llm().bind_tools([_STOCK]).invoke("go")
    assert message.content == "The trend is up."
    assert not message.tool_calls


@pytest.mark.unit
def test_broken_tool_json_degrades_to_text(fake_cli):
    fake_cli(['{"tool_calls": ['])
    message = _llm(max_retries=0).bind_tools([_STOCK]).invoke("go")
    assert message.content == '{"tool_calls": ['
    assert not message.tool_calls


@pytest.mark.unit
def test_token_counts_from_the_envelope_are_kept(fake_cli):
    body = json.dumps({
        "type": "result", "is_error": False, "result": "ok",
        "usage": {"input_tokens": 11, "output_tokens": 7},
    })
    fake_cli([(0, body, "")])
    message = _llm().invoke("hi")
    assert message.usage_metadata["input_tokens"] == 11
    assert message.usage_metadata["output_tokens"] == 7
    assert message.usage_metadata["total_tokens"] == 18


@pytest.mark.unit
def test_stop_sequence_truncates_plain_text(fake_cli):
    fake_cli(["hello STOP world"])
    message = _llm().invoke("go", stop=["STOP"])
    assert message.content == "hello "


class Decision(BaseModel):
    """A choice."""

    rating: Literal["Buy", "Sell"] = Field(description="the call")
    note: str


@pytest.mark.unit
def test_structured_output_parses_the_schema_tool_call(fake_cli):
    fake_cli([json.dumps({
        "content": "",
        "tool_calls": [{"name": "Decision", "arguments": {"rating": "Buy", "note": "trend is up"}}],
    })])
    parsed = _llm().with_structured_output(Decision).invoke("Decide.")
    assert parsed == Decision(rating="Buy", note="trend is up")


@pytest.mark.unit
def test_a_bare_schema_object_is_the_structured_tool_call(fake_cli):
    fake_cli([json.dumps({"rating": "Sell", "note": "trend is down"})])
    parsed = _llm().with_structured_output(Decision).invoke("Decide.")
    assert parsed == Decision(rating="Sell", note="trend is down")


@pytest.mark.unit
def test_structured_output_failure_falls_back_instead_of_crashing(fake_cli):
    from tradingagents.agents.structured import invoke_structured

    invalid = json.dumps({
        "content": "",
        "tool_calls": [{"name": "Decision", "arguments": {"rating": "Maybe", "note": "nope"}}],
    })
    fake_cli(["I would buy, but not as JSON.", "I would buy, but not as JSON.", invalid])
    structured = _llm().with_structured_output(Decision)
    assert structured.invoke("Decide.") is None
    assert invoke_structured(structured, "Decide.", "Portfolio Manager") is None
    assert invoke_structured(structured, "Decide.", "Portfolio Manager") is None


@pytest.mark.unit
def test_cli_flags_disable_tools_and_keep_the_prompt_off_the_command_line(fake_cli):
    fake = fake_cli(["ok"])
    _llm(effort="high").invoke("secret-prompt-body")
    cmd = fake.calls[0]["cmd"]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert "--bare" not in cmd
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--effort") + 1] == "high"
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--permission-mode") + 1] == "plan"
    for flag in ("--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands"):
        assert flag in cmd
    assert "secret-prompt-body" not in cmd
    assert "secret-prompt-body" in fake.calls[0]["prompt"]
    assert "tradingagents-claude-cli" in fake.calls[0]["cwd"]


@pytest.mark.unit
def test_effort_is_omitted_when_unset(fake_cli):
    fake = fake_cli(["ok"])
    _llm().invoke("hi")
    assert "--effort" not in fake.calls[0]["cmd"]


@pytest.mark.unit
def test_api_key_is_stripped_and_never_logged(fake_cli, monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecretvalue")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-supersecretvalue")
    monkeypatch.setenv("CLAUDE_CODE_SIMPLE", "1")
    failure = (1, "", "failed api_key=sk-ant-supersecretvalue")
    fake = fake_cli([failure, failure])
    caplog.set_level(logging.WARNING)
    with pytest.raises(ClaudeCLIError) as caught:
        _llm(max_retries=1).invoke("hi")
    assert "sk-ant-supersecretvalue" not in str(caught.value)
    assert "token-supersecretvalue" not in str(caught.value)
    assert "sk-ant-supersecretvalue" not in caplog.text
    assert "[redacted]" in str(caught.value)
    for call in fake.calls:
        assert "ANTHROPIC_API_KEY" not in call["env"]
        assert "ANTHROPIC_AUTH_TOKEN" not in call["env"]
        assert "CLAUDE_CODE_SIMPLE" not in call["env"]
        assert call["env"]["CLAUDE_CODE_SAFE_MODE"] == "1"
        joined = " ".join(call["cmd"])
        assert "sk-ant-supersecretvalue" not in joined
        assert "token-supersecretvalue" not in joined
    assert len(fake.calls) == 2  # one retry, then the scrubbed error


@pytest.mark.unit
def test_envelope_with_a_leading_log_line_and_is_error_retries(fake_cli):
    fake = fake_cli([
        (0, "warning: warmup\n" + _envelope("ignored", is_error=True), ""),
        "recovered",
    ])
    message = _llm(max_retries=1).invoke("hi")
    assert message.content == "recovered"
    assert len(fake.calls) == 2


@pytest.mark.unit
def test_timeout_is_not_retried(fake_cli):
    fake = fake_cli([mod.ClaudeCLITimeout("claude timed out after 5s")])
    with pytest.raises(mod.ClaudeCLITimeout):
        _llm(max_retries=3).invoke("hi")
    assert len(fake.calls) == 1


@pytest.mark.unit
def test_missing_binary_is_reported_once():
    with pytest.raises(ClaudeCLIConfigError, match="not found"):
        mod._run_process(
            ["no-such-claude-binary-xyz"], "hi", {}, None, 5,
        )


@pytest.mark.unit
def test_a_timed_out_process_is_killed():
    started = time.monotonic()
    with pytest.raises(mod.ClaudeCLITimeout):
        mod._run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            "hi", {}, None, 0.4,
        )
    assert time.monotonic() - started < 5


@pytest.mark.unit
def test_concurrent_calls_share_one_process_limit():
    state = {"current": 0, "max": 0}
    lock = threading.Lock()

    def run(*_args, **_kwargs):
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.15)
        with lock:
            state["current"] -= 1
        return 0, _envelope("ok"), ""

    first = _llm(max_concurrency=1)
    second = _llm(max_concurrency=1)
    with patch.object(mod, "_run_process", run):
        errors: list[BaseException] = []

        def invoke(model):
            try:
                model.invoke("hi")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=invoke, args=(model,)) for model in (first, second, first)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []
    assert state["max"] == 1


@pytest.mark.unit
def test_factory_registry_and_effort_env(monkeypatch):
    assert get_api_key_env("claude_cli") is None
    assert not is_openai_compatible("claude_cli")
    assert validate_model("claude_cli", "opus")
    assert validate_model("claude_cli", "claude-opus-4-6")
    from tradingagents.llm_clients.model_catalog import get_model_options
    quick = {value for _, value in get_model_options("claude_cli", "quick")}
    assert {"opus", "sonnet", "haiku", "custom"} <= quick
    assert "opus" in {value for _, value in get_model_options("claude_cli", "deep")}

    from cli.prompts import _llm_provider_table
    assert ("claude_cli", None) in [(key, url) for _, key, url in _llm_provider_table()]

    kwargs = build_llm_kwargs({
        "llm_provider": "claude_cli",
        "anthropic_effort": "high",
        "llm_max_retries": "1",
    })
    assert kwargs["effort"] == "high"
    assert kwargs["max_retries"] == 1
    llm = create_llm_client("Claude_CLI", "opus", **kwargs).get_llm()
    assert isinstance(llm, ChatClaudeCLI)
    assert llm.model == "opus"
    assert llm.effort == "high"
    assert llm.max_retries == 1

    with pytest.raises(ValueError, match="effort"):
        create_llm_client("claude_cli", "opus", effort="turbo").get_llm()


@pytest.mark.unit
def test_headless_env_selects_opus_without_an_api_key(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "claude_cli")
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "opus")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "opus")
    monkeypatch.setenv("TRADINGAGENTS_ANTHROPIC_EFFORT", "high")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from tradingagents.default_config import build_default_config

    config = build_default_config()
    assert config["llm_provider"] == "claude_cli"
    assert config["deep_think_llm"] == "opus"
    assert config["quick_think_llm"] == "opus"
    assert config["anthropic_effort"] == "high"
    llm = create_llm_client(
        config["llm_provider"], config["deep_think_llm"], **build_llm_kwargs(config),
    ).get_llm()
    assert llm.model == "opus" and llm.effort == "high"


@pytest.mark.unit
def test_no_api_key_prompt_and_ignored_sampling_knobs(monkeypatch):
    import cli.prompts as prompts

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch.object(prompts, "questionary") as questionary:
        assert prompts.ensure_api_key("claude_cli") is None
    questionary.password.assert_not_called()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_llm_client(
            "claude_cli", "opus", temperature=0.2, max_tokens=128,
            base_url="https://api.anthropic.com/",
        ).get_llm()
    messages = " ".join(str(item.message) for item in caught)
    assert "temperature" in messages
    assert "max_tokens" in messages
    assert "backend_url" in messages


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI is not on PATH")
def test_live_cli_answers_when_installed():
    llm = ChatClaudeCLI(model="haiku", effort="low", timeout=180, max_retries=0, max_concurrency=1)
    message = llm.invoke("Reply with the single word pong and nothing else.")
    assert "pong" in message.content.lower()
