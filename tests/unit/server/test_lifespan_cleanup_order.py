from __future__ import annotations

import anyio
import pytest

from market_support_crewai_agent.server import lifespan, main
from market_support_crewai_agent.settings_model import Settings


def test_lifespan_starts_health_only_after_cleanup_is_ready_and_stops_in_reverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def cleanup_loop(
        coordinator,
        stop_event: anyio.Event,
        interval_seconds: int,
        *,
        task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        del coordinator, interval_seconds
        events.append("cleanup_ready")
        task_status.started()
        try:
            await stop_event.wait()
        finally:
            events.append("cleanup_exit")

    def start_health(settings: Settings, *, process_health_key: bytes) -> None:
        del settings
        assert len(process_health_key) == 32
        events.append("health_start")

    async def stop_health() -> None:
        events.append("health_stop")

    monkeypatch.setattr(lifespan, "get_settings", lambda: Settings())
    monkeypatch.setattr(lifespan, "get_reply_state_coordinator", lambda: object())
    monkeypatch.setattr(lifespan, "_cleanup_coordinator_loop", cleanup_loop)
    monkeypatch.setattr(lifespan, "start_llm_health_monitor", start_health)
    monkeypatch.setattr(lifespan, "stop_llm_health_monitor", stop_health)

    async def run() -> None:
        async with lifespan.app_lifespan(main.app):
            events.append("serving")
        events.append("closed")

    anyio.run(run)

    assert events == [
        "cleanup_ready",
        "health_start",
        "serving",
        "health_stop",
        "cleanup_exit",
        "closed",
    ]


def test_lifespan_cancels_cleanup_when_health_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def cleanup_loop(
        coordinator,
        stop_event: anyio.Event,
        interval_seconds: int,
        *,
        task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        del coordinator, stop_event, interval_seconds
        events.append("cleanup_ready")
        task_status.started()
        try:
            await anyio.sleep_forever()
        finally:
            events.append("cleanup_exit")

    def start_health(settings: Settings, *, process_health_key: bytes) -> None:
        del settings
        assert len(process_health_key) == 32
        events.append("health_start")
        raise RuntimeError("health_start_failed")

    monkeypatch.setattr(lifespan, "get_settings", lambda: Settings())
    monkeypatch.setattr(lifespan, "get_reply_state_coordinator", lambda: object())
    monkeypatch.setattr(lifespan, "_cleanup_coordinator_loop", cleanup_loop)
    monkeypatch.setattr(lifespan, "start_llm_health_monitor", start_health)

    async def run() -> None:
        with pytest.raises(ExceptionGroup) as error:
            async with lifespan.app_lifespan(main.app):
                raise AssertionError("lifespan yielded after health startup failure")
        assert isinstance(error.value.exceptions[0], RuntimeError)
        assert str(error.value.exceptions[0]) == "health_start_failed"

    anyio.run(run)

    assert events == ["cleanup_ready", "health_start", "cleanup_exit"]


def test_lifespan_does_not_start_health_when_cleanup_fails_before_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def cleanup_loop(
        coordinator,
        stop_event: anyio.Event,
        interval_seconds: int,
        *,
        task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        del coordinator, stop_event, interval_seconds, task_status
        events.append("cleanup_start")
        raise RuntimeError("cleanup_start_failed")

    def start_health(settings: Settings, *, process_health_key: bytes) -> None:
        del settings
        assert len(process_health_key) == 32
        events.append("health_start")

    monkeypatch.setattr(lifespan, "get_settings", lambda: Settings())
    monkeypatch.setattr(lifespan, "get_reply_state_coordinator", lambda: object())
    monkeypatch.setattr(lifespan, "_cleanup_coordinator_loop", cleanup_loop)
    monkeypatch.setattr(lifespan, "start_llm_health_monitor", start_health)

    async def run() -> None:
        with pytest.raises(ExceptionGroup) as error:
            async with lifespan.app_lifespan(main.app):
                raise AssertionError("lifespan yielded after cleanup startup failure")
        assert isinstance(error.value.exceptions[0], RuntimeError)
        assert str(error.value.exceptions[0]) == "cleanup_start_failed"

    anyio.run(run)

    assert events == ["cleanup_start"]
