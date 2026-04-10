from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


class _FakeAgent:
    def __init__(self):
        self.interrupt = MagicMock()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    return runner, adapter


class TestStallAutoNudge:
    def test_stall_timeout_deadline_extends_once_after_nudge(self):
        from gateway.run import GatewayRunner

        assert GatewayRunner._stall_timeout_should_fire(
            idle_secs=1800,
            timeout_secs=1800,
            nudge_sent_at=None,
            now_monotonic=100.0,
        ) is True
        assert GatewayRunner._stall_timeout_should_fire(
            idle_secs=9999,
            timeout_secs=1800,
            nudge_sent_at=100.0,
            now_monotonic=1500.0,
        ) is False
        assert GatewayRunner._stall_timeout_should_fire(
            idle_secs=9999,
            timeout_secs=1800,
            nudge_sent_at=100.0,
            now_monotonic=1900.0,
        ) is True

    @pytest.mark.asyncio
    async def test_auto_nudge_interrupts_once_when_no_approval_pending(self):
        runner, adapter = _make_runner()
        agent = _FakeAgent()

        with patch("tools.approval.has_blocking_approval", return_value=False):
            did_nudge = await runner._maybe_auto_nudge_stalled_agent(
                session_key="agent:main:telegram:dm:12345",
                source=_make_source(),
                agent=agent,
                timeout_secs=1800,
                already_nudged=False,
                status_thread_metadata={"thread_id": "1795"},
            )

        assert did_nudge is True
        agent.interrupt.assert_called_once_with("Please continue from where you left off.")
        adapter.send.assert_awaited_once()
        sent_text = adapter.send.await_args.args[1]
        assert "30 min" in sent_text
        assert "nudging the agent once" in sent_text

    @pytest.mark.asyncio
    async def test_auto_nudge_skips_when_already_nudged_or_approval_pending(self):
        runner, adapter = _make_runner()
        agent = _FakeAgent()

        with patch("tools.approval.has_blocking_approval", return_value=False):
            did_nudge = await runner._maybe_auto_nudge_stalled_agent(
                session_key="agent:main:telegram:dm:12345",
                source=_make_source(),
                agent=agent,
                timeout_secs=1800,
                already_nudged=True,
                status_thread_metadata=None,
            )
        assert did_nudge is False

        with patch("tools.approval.has_blocking_approval", return_value=True):
            did_nudge = await runner._maybe_auto_nudge_stalled_agent(
                session_key="agent:main:telegram:dm:12345",
                source=_make_source(),
                agent=agent,
                timeout_secs=1800,
                already_nudged=False,
                status_thread_metadata=None,
            )
        assert did_nudge is False
        agent.interrupt.assert_not_called()
        adapter.send.assert_not_awaited()
