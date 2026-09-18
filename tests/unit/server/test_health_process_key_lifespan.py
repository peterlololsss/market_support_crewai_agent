from __future__ import annotations

import secrets

import anyio
import pytest

from market_support_crewai_agent.server import lifespan, main
from market_support_crewai_agent.settings_model import Settings


def test_lifespan_creates_one_32_byte_process_health_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a deterministic process-key source and ready cleanup loop.
    process_key = b"p" * 32
    requested_lengths: list[int] = []
    received_keys: list[bytes | None] = []

    def fake_token_bytes(length: int) -> bytes:
        requested_lengths.append(length)
        return process_key

    async def cleanup_loop(
        coordinator,
        stop_event: anyio.Event,
        interval_seconds: int,
        *,
        task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
    ) -> None:
        del coordinator, interval_seconds
        task_status.started()
        await stop_event.wait()

    def start_health(
        settings: Settings,
        process_health_key: bytes | None = None,
    ) -> None:
        del settings
        received_keys.append(process_health_key)

    async def stop_health() -> None:
        return None

    monkeypatch.setattr(secrets, "token_bytes", fake_token_bytes)
    monkeypatch.setattr(lifespan, "get_settings", lambda: Settings())
    monkeypatch.setattr(lifespan, "get_reply_state_coordinator", lambda: object())
    monkeypatch.setattr(lifespan, "_cleanup_coordinator_loop", cleanup_loop)
    monkeypatch.setattr(lifespan, "start_llm_health_monitor", start_health)
    monkeypatch.setattr(lifespan, "stop_llm_health_monitor", stop_health)

    async def run_lifespan() -> None:
        async with lifespan.app_lifespan(main.app):
            pass

    # When: the application lifespan starts and stops once.
    anyio.run(run_lifespan)

    # Then: one non-configurable 32-byte key is injected into health startup.
    assert requested_lengths == [32]
    assert received_keys == [process_key]
