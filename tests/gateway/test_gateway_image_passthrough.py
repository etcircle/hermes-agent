import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None
    last_run = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        type(self).last_run = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
            "persist_user_message": persist_user_message,
        }
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _make_source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


@pytest.mark.asyncio
async def test_gateway_auto_mode_uses_multimodal_passthrough_for_supported_runtime(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"auxiliary": {"vision": {"gateway_mode": "auto"}}})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "***",
        },
    )
    monkeypatch.setattr(gateway_run, "resolve_native_vision_support", lambda **kwargs: (True, "supports vision"))
    monkeypatch.setattr(
        gateway_run,
        "build_multimodal_user_content",
        lambda text, image_paths: [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
        ],
    )

    _CapturingAgent.last_run = None
    result = await runner._run_agent(
        message="look at this",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
        image_paths=[str(tmp_path / "a.png")],
    )

    assert result["final_response"] == "ok"
    assert isinstance(_CapturingAgent.last_run["user_message"], list)
    assert _CapturingAgent.last_run["persist_user_message"] == "look at this\n\n[User attached 1 image]"
    runner._enrich_message_with_vision.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_auto_mode_falls_back_to_auxiliary_description_when_runtime_not_supported(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"auxiliary": {"vision": {"gateway_mode": "auto"}}})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "***",
        },
    )
    monkeypatch.setattr(gateway_run, "resolve_native_vision_support", lambda **kwargs: (False, "no vision support"))

    _CapturingAgent.last_run = None
    result = await runner._run_agent(
        message="look at this",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
        image_paths=[str(tmp_path / "a.png")],
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_run["user_message"] == "ENRICHED"
    assert _CapturingAgent.last_run["persist_user_message"] is None
    runner._enrich_message_with_vision.assert_awaited_once()


@pytest.mark.asyncio
async def test_gateway_passthrough_mode_returns_clear_error_when_runtime_support_unknown(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"auxiliary": {"vision": {"gateway_mode": "passthrough"}}})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://example.invalid/v1",
            "api_key": "***",
        },
    )
    monkeypatch.setattr(gateway_run, "resolve_native_vision_support", lambda **kwargs: (None, "unknown model capabilities"))

    _CapturingAgent.last_run = None
    result = await runner._run_agent(
        message="look at this",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
        image_paths=[str(tmp_path / "a.png")],
    )

    assert "Native image passthrough is forced" in result["final_response"]
    assert _CapturingAgent.last_run is None
    runner._enrich_message_with_vision.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_model_switch_note_is_prepended_to_multimodal_payload_and_persist_shadow(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()
    session_key = "agent:main:telegram:dm:12345"
    runner._pending_model_notes[session_key] = "[Switched to gpt-5.4]"

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"auxiliary": {"vision": {"gateway_mode": "auto"}}})
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "***",
        },
    )
    monkeypatch.setattr(gateway_run, "resolve_native_vision_support", lambda **kwargs: (True, "supports vision"))
    monkeypatch.setattr(
        gateway_run,
        "build_multimodal_user_content",
        lambda text, image_paths: [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
        ],
    )

    _CapturingAgent.last_run = None
    result = await runner._run_agent(
        message="look at this",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key=session_key,
        image_paths=[str(tmp_path / "a.png")],
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_run["user_message"][0] == {"type": "text", "text": "[Switched to gpt-5.4]"}
    assert _CapturingAgent.last_run["persist_user_message"].startswith("[Switched to gpt-5.4]\n\n")
