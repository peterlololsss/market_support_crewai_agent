from __future__ import annotations

import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from anyio import TASK_STATUS_IGNORED, Event, create_task_group, move_on_after
from anyio.abc import TaskStatus
from anyio.to_thread import run_sync
from fastapi import FastAPI

from market_support_crewai_agent.health.llm_health import (
    start_llm_health_monitor,
    stop_llm_health_monitor,
)
from market_support_crewai_agent.runtime.state.coordinator_provider import (
    get_reply_state_coordinator,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.settings import get_settings


@asynccontextmanager
async def app_lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    coordinator = get_reply_state_coordinator()
    stop_event = Event()
    process_health_key = secrets.token_bytes(32)
    async with create_task_group() as task_group:
        await task_group.start(
            _cleanup_coordinator_loop,
            coordinator,
            stop_event,
            settings.agent_conversation_cleanup_interval_seconds,
        )
        _ = start_llm_health_monitor(
            settings,
            process_health_key=process_health_key,
        )
        try:
            yield
        finally:
            try:
                await stop_llm_health_monitor()
            finally:
                del process_health_key
                stop_event.set()
                task_group.cancel_scope.cancel()


async def _cleanup_coordinator_loop(
    coordinator: ReplyStateTransactionCoordinatorV1,
    stop_event: Event,
    interval_seconds: int,
    task_status: TaskStatus[None] = TASK_STATUS_IGNORED,
) -> None:
    _ = await run_sync(coordinator.cleanup)
    _ = task_status.started()
    while True:
        with move_on_after(interval_seconds):
            await stop_event.wait()
        if stop_event.is_set():
            return
        _ = await run_sync(coordinator.cleanup)
