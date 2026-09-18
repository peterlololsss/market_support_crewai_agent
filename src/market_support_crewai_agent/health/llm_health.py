from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import final
from zoneinfo import ZoneInfo

from market_support_crewai_agent.health import formatting, probe
from market_support_crewai_agent.health.feishu_sender import FeishuTextSender
from market_support_crewai_agent.health.llm_health_hooks import (
    register_llm_health_hooks,
)
from market_support_crewai_agent.health.models import (
    HealthTargetSlot,
    LlmHealthState,
    LlmHealthTarget,
)
from market_support_crewai_agent.runtime.integrations.crewai.agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
    normalize_provider_failure_code,
)
from market_support_crewai_agent.settings_model import Settings

logger = logging.getLogger(__name__)


@final
class LlmHealthMonitor:
    def __init__(self, settings: Settings, *, process_health_key: bytes) -> None:
        self.settings: Settings = settings
        self._process_health_key: bytes = process_health_key
        self.targets: list[LlmHealthTarget] = list(
            probe.discover_llm_health_targets(
                settings,
                process_health_key=process_health_key,
            )
        )
        self.targets_by_key: dict[str, LlmHealthTarget] = {
            target.key: target for target in self.targets
        }
        self.states: dict[str, LlmHealthState] = {}
        self.sender: FeishuTextSender = FeishuTextSender(settings)
        self.agent_factory: CrewAIAgentFactory = CrewAIAgentFactory(settings)
        self._tasks: list[asyncio.Task[None]] = []
        self._stop_event: asyncio.Event | None = None
        self.tz: ZoneInfo | timezone = formatting.timezone_for(
            settings.llm_health_timezone
        )

    def start(self) -> None:
        if self._tasks:
            return
        self._stop_event = asyncio.Event()
        self._tasks = [
            asyncio.create_task(self._probe_loop(), name="llm-health-probe"),
            asyncio.create_task(
                self._daily_report_loop(), name="llm-health-daily-report"
            ),
        ]

    async def stop(self) -> None:
        if self._stop_event is not None:
            _ = self._stop_event.set()
        if self._tasks:
            _ = await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def mark_agent_success(
        self, agent: CrewAIAgentAdapterV1 | None, stage: str
    ) -> None:
        target = self._target_for_agent(agent, stage)
        if target is not None:
            self.mark_success(target)

    def mark_agent_failure(
        self,
        agent: CrewAIAgentAdapterV1 | None,
        stage: str,
        reason: ProviderFailureCodeV1,
    ) -> None:
        target = self._target_for_agent(agent, stage)
        if target is not None:
            self.mark_failure(target, reason=reason)

    def mark_success(self, target: LlmHealthTarget) -> None:
        state = self._state_for(target)
        recovered_outage = state.record_success(self._now())
        if recovered_outage is not None:
            self._notify(
                formatting.format_recovery(target, recovered_outage, self._now())
            )

    def mark_failure(
        self,
        target: LlmHealthTarget,
        *,
        reason: ProviderFailureCodeV1,
    ) -> None:
        now = self._now()
        should_warn = self._state_for(target).record_failure(
            now,
            reason=reason,
            warning_cooldown_seconds=(
                self.settings.llm_health_warning_cooldown_seconds
            ),
        )
        if should_warn:
            self._notify(formatting.format_warning(target, reason, now))

    async def _probe_loop(self) -> None:
        while not self._stopped:
            now = self._now()
            for target in list(self.targets):
                state = self._state_for(target)
                interval = (
                    self.settings.llm_health_failure_interval_seconds
                    if state.status == "unhealthy"
                    else self.settings.llm_health_check_interval_seconds
                )
                if (
                    state.last_probe_at is None
                    or (now - state.last_probe_at).total_seconds() >= interval
                ):
                    state.last_probe_at = now
                    await self._probe_target(target)
            await self._sleep(
                min(30.0, self.settings.llm_health_failure_interval_seconds)
            )

    async def _daily_report_loop(self) -> None:
        while not self._stopped:
            now = self._now()
            report_at = self._next_daily_report_at(now)
            await self._sleep(max(1.0, (report_at - now).total_seconds()))
            if not self._stopped:
                end = self._now()
                self._notify(self.format_daily_report(end - timedelta(days=1), end))

    async def _probe_target(self, target: LlmHealthTarget) -> None:
        if not self._api_key_configured(target.target_slot):
            self.mark_failure(target, reason="provider_auth_failed")
            return
        reason = await probe.dispatch_health_probe(
            self._build_probe_agent(target),
            target,
            timeout_seconds=self.settings.llm_health_probe_timeout_seconds,
        )
        if reason is None:
            self.mark_success(target)
            return
        self.mark_failure(target, reason=reason)

    def _build_probe_agent(self, target: LlmHealthTarget) -> CrewAIAgentAdapterV1:
        return self.agent_factory.build_health_probe_agent(
            target_slot=target.target_slot,
            prompt_profile=probe.health_probe_program().profile,
        )

    def format_daily_report(self, start: datetime, end: datetime) -> str:
        entries = tuple(
            formatting.HealthReportTarget(target, self._state_for(target))
            for target in self.targets
        )
        return formatting.format_daily_report(entries, start, end, self._now())

    def _target_for_agent(
        self, agent: CrewAIAgentAdapterV1 | None, stage: str
    ) -> LlmHealthTarget | None:
        return probe.target_for_agent(
            agent,
            stage,
            self.targets_by_key,
            self._process_health_key,
        )

    def _api_key_configured(self, target_slot: HealthTargetSlot) -> bool:
        if target_slot == "planner":
            return bool(self.settings.planner_llm_api_key)
        return bool(self.settings.llm_api_key)

    def _state_for(self, target: LlmHealthTarget) -> LlmHealthState:
        return self.states.setdefault(target.key, LlmHealthState())

    def _notify(self, text: str) -> None:
        if not self.sender.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        _ = loop.create_task(self._send_text(text))

    async def _send_text(self, text: str) -> None:
        try:
            await asyncio.to_thread(self.sender.send_text, text)
        except (OSError, RuntimeError):
            logger.warning("Feishu health notification failed")

    def _next_daily_report_at(self, now: datetime) -> datetime:
        return formatting.next_daily_report_at(
            now,
            self.settings.llm_health_daily_report_time,
            self.tz,
        )

    async def _sleep(self, seconds: float) -> None:
        if self._stop_event is None:
            await asyncio.sleep(seconds)
            return
        try:
            _ = await asyncio.wait_for(
                self._stop_event.wait(), timeout=max(0.1, seconds)
            )
        except TimeoutError:
            return

    @property
    def _stopped(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _now(self) -> datetime:
        return datetime.now(self.tz)


_monitor: LlmHealthMonitor | None = None


def start_llm_health_monitor(
    settings: Settings,
    *,
    process_health_key: bytes,
) -> LlmHealthMonitor | None:
    global _monitor
    if not settings.llm_health_enabled:
        return None
    if _monitor is None:
        _monitor = LlmHealthMonitor(
            settings,
            process_health_key=process_health_key,
        )
        _monitor.start()
    return _monitor


async def stop_llm_health_monitor() -> None:
    global _monitor
    monitor = _monitor
    _monitor = None
    if monitor is not None:
        await monitor.stop()


def record_llm_success_for_agent(
    agent: CrewAIAgentAdapterV1 | None, stage: str
) -> None:
    if _monitor is not None:
        _monitor.mark_agent_success(agent, stage)


def record_llm_failure_for_agent(
    agent: CrewAIAgentAdapterV1 | None,
    stage: str,
    reason: str,
) -> None:
    if _monitor is not None:
        _monitor.mark_agent_failure(
            agent,
            stage,
            normalize_provider_failure_code(reason),
        )


register_llm_health_hooks(
    success_hook=record_llm_success_for_agent,
    failure_hook=lambda agent, stage, reason: record_llm_failure_for_agent(
        agent,
        stage,
        reason,
    ),
)
