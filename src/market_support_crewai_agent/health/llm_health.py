from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Literal
from urllib import request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.profiles import PromptProfile
from market_support_crewai_agent.runtime.llm.retry import RetryPolicy, run_with_retry
from market_support_crewai_agent.runtime.orchestration.crewai_agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.orchestration.crewai_io import (
    run_crewai_kickoff,
    safe_short_text,
)
from market_support_crewai_agent.settings import Settings

logger = logging.getLogger(__name__)

TargetKind = Literal["default", "planner"]


class HealthProbeResponse(BaseModel):
    ok: bool


@dataclass(frozen=True)
class LlmHealthTarget:
    key: str
    kind: TargetKind
    stage_label: str
    provider: str
    model: str
    base_url: str
    api_key_configured: bool


@dataclass
class OutageWindow:
    started_at: datetime
    ended_at: datetime | None = None
    reason: str = ""


@dataclass
class LlmHealthState:
    status: Literal["unknown", "healthy", "unhealthy"] = "unknown"
    healthy_since: datetime | None = None
    unhealthy_since: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_probe_at: datetime | None = None
    last_warning_at: datetime | None = None
    last_error: str = ""
    outages: list[OutageWindow] = field(default_factory=list)


class FeishuTextSender:
    def __init__(self, settings: Settings) -> None:
        self.app_id = settings.feishu_app_id
        self.app_secret = settings.feishu_app_secret
        self.chat_id = settings.feishu_chat_id
        self.timeout_seconds = 10
        self._opener = request.build_opener(request.ProxyHandler({}))

    @property
    def enabled(self) -> bool:
        return bool(self.app_id and self.app_secret and self.chat_id)

    def send_text(self, text: str) -> None:
        if not self.enabled:
            return
        token = self._tenant_access_token()
        payload = {
            "receive_id": self.chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        self._post_json(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            payload,
            headers={"Authorization": f"Bearer {token}"},
        )

    def _tenant_access_token(self) -> str:
        data = self._post_json(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/",
            {"app_id": self.app_id, "app_secret": self.app_secret},
        )
        token = data.get("tenant_access_token")
        if not token:
            raise RuntimeError(f"Feishu token missing: {data}")
        return str(token)

    def _post_json(self, url: str, payload: dict, *, headers: dict | None = None) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                **(headers or {}),
            },
            method="POST",
        )
        with self._opener.open(http_request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("code", 0) != 0:
            raise RuntimeError(f"Feishu API error: {data}")
        return data


class LlmHealthMonitor:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.targets = list(discover_llm_health_targets(settings))
        self.targets_by_key = {target.key: target for target in self.targets}
        self.states: dict[str, LlmHealthState] = {}
        self.sender = FeishuTextSender(settings)
        self.agent_factory = CrewAIAgentFactory(settings)
        self._tasks: list[asyncio.Task] = []
        self._stop_event: asyncio.Event | None = None
        self.tz = _timezone(settings.llm_health_timezone)

    def start(self) -> None:
        if self._tasks:
            return
        self._stop_event = asyncio.Event()
        self._tasks = [
            asyncio.create_task(self._probe_loop(), name="llm-health-probe"),
            asyncio.create_task(self._daily_report_loop(), name="llm-health-daily-report"),
        ]

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def mark_agent_success(self, agent, stage: str) -> None:
        target = self._target_for_agent(agent, stage)
        if target is not None:
            self.mark_success(target)

    def mark_agent_failure(self, agent, stage: str, reason: str) -> None:
        target = self._target_for_agent(agent, stage)
        if target is not None:
            self.mark_failure(target, reason=reason, stage=stage)

    def mark_success(self, target: LlmHealthTarget) -> None:
        now = self._now()
        state = self._state_for(target)
        previous_outage = (
            state.outages[-1]
            if state.status == "unhealthy" and state.outages
            else None
        )

        state.status = "healthy"
        state.last_success_at = now
        state.last_error = ""
        state.unhealthy_since = None
        if state.healthy_since is None or previous_outage is not None:
            state.healthy_since = now

        if previous_outage and previous_outage.ended_at is None:
            previous_outage.ended_at = now
            self._notify(self._format_recovery(target, previous_outage))

    def mark_failure(self, target: LlmHealthTarget, *, reason: str, stage: str) -> None:
        now = self._now()
        state = self._state_for(target)
        clean_reason = safe_short_text(reason) or "未知错误"

        first_failure = state.status != "unhealthy"
        state.status = "unhealthy"
        state.last_failure_at = now
        state.last_error = clean_reason
        state.healthy_since = None

        if first_failure:
            state.unhealthy_since = now
            state.outages.append(OutageWindow(started_at=now, reason=clean_reason))

        if first_failure or self._warning_due(state, now):
            state.last_warning_at = now
            self._notify(self._format_warning(target, stage, clean_reason, now))

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
            await self._sleep(min(30.0, self.settings.llm_health_failure_interval_seconds))

    async def _daily_report_loop(self) -> None:
        while not self._stopped:
            now = self._now()
            report_at = self._next_daily_report_at(now)
            await self._sleep(max(1.0, (report_at - now).total_seconds()))
            if not self._stopped:
                end = self._now()
                self._notify(self.format_daily_report(end - timedelta(days=1), end))

    async def _probe_target(self, target: LlmHealthTarget) -> None:
        if not target.api_key_configured:
            self.mark_failure(target, reason="API key 未配置", stage="health_probe")
            return

        agent = self._build_probe_agent(target)
        program = _health_probe_program()

        async def call():
            try:
                result, _execution = await run_crewai_kickoff(
                    agent,
                    program,
                    timeout_seconds=self.settings.llm_health_probe_timeout_seconds,
                )
                return result
            except asyncio.TimeoutError as exc:
                raise RuntimeError("健康检查超时") from exc

        try:
            await run_with_retry(
                call,
                policy=RetryPolicy(
                    retry_attempts=self.settings.llm_health_probe_retry_attempts,
                    base_delay_seconds=self.settings.llm_health_probe_retry_base_seconds,
                ),
                should_retry_result=_probe_retry_reason,
                should_retry_exception=lambda exc: safe_short_text(exc) or type(exc).__name__,
            )
        except Exception as exc:
            self.mark_failure(
                target,
                reason=safe_short_text(exc) or type(exc).__name__,
                stage="health_probe",
            )
            return

        self.mark_success(target)

    def _build_probe_agent(self, target: LlmHealthTarget):
        if target.kind == "planner":
            return self.agent_factory.build_planner_agent()
        return self.agent_factory.build_composer_agent(stage="smalltalk_composer")

    def format_daily_report(self, start: datetime, end: datetime) -> str:
        lines = [f"【LLM 健康日报】{_fmt_date_time(start)} ~ {_fmt_date_time(end)}", ""]
        total_seconds = max(1.0, (end - start).total_seconds())

        for target in self.targets:
            state = self._state_for(target)
            outage_seconds = self._outage_seconds(state, start, end)

            lines.append(f"{target.model}：{_status_text(state.status)}")
            if state.status == "healthy" and state.healthy_since is not None:
                availability = max(
                    0.0,
                    100.0 * (total_seconds - outage_seconds) / total_seconds,
                )
                lines.append(
                    f"- 连续可用：{_fmt_duration((end - state.healthy_since).total_seconds())}"
                )
                lines.append(f"- 不可用时间段：{self._format_outages(state, start, end)}")
                lines.append(f"- 今日总不可用：{_fmt_duration(outage_seconds)}")
                lines.append(f"- 可用率：{availability:.2f}%")
            elif state.status == "unhealthy" and state.unhealthy_since is not None:
                availability = max(
                    0.0,
                    100.0 * (total_seconds - outage_seconds) / total_seconds,
                )
                lines.append(
                    f"- 已不可用：{_fmt_duration((end - state.unhealthy_since).total_seconds())}"
                )
                lines.append(f"- 不可用时间段：{self._format_outages(state, start, end)}")
                lines.append(f"- 今日总不可用：{_fmt_duration(outage_seconds)}")
                lines.append(f"- 可用率：{availability:.2f}%")
            else:
                lines.append("- 检测结果：尚未完成首次健康检查")
            if state.last_error:
                lines.append(f"- 最后错误：{state.last_error}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _target_for_agent(self, agent, stage: str) -> LlmHealthTarget | None:
        llm = getattr(agent, "llm", None)
        if llm is None:
            return None
        provider = str(getattr(llm, "provider", "") or "")
        model = str(getattr(llm, "model", "") or "")
        base_url = _agent_base_url(llm)
        if not model:
            return None

        key = _target_key(provider, model, base_url)
        target = self.targets_by_key.get(key)
        if target is not None:
            return target

        target = LlmHealthTarget(
            key=key,
            kind="default",
            stage_label=stage,
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_configured=bool(getattr(llm, "api_key", None)),
        )
        self.targets.append(target)
        self.targets_by_key[key] = target
        return target

    def _state_for(self, target: LlmHealthTarget) -> LlmHealthState:
        return self.states.setdefault(target.key, LlmHealthState())

    def _warning_due(self, state: LlmHealthState, now: datetime) -> bool:
        if state.last_warning_at is None:
            return True
        return (
            now - state.last_warning_at
        ).total_seconds() >= self.settings.llm_health_warning_cooldown_seconds

    def _notify(self, text: str) -> None:
        if not self.sender.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._send_text(text))

    async def _send_text(self, text: str) -> None:
        try:
            await asyncio.to_thread(self.sender.send_text, text)
        except Exception as exc:
            logger.warning("Feishu health notification failed: %s", exc)

    def _format_warning(
        self,
        target: LlmHealthTarget,
        stage: str,
        reason: str,
        now: datetime,
    ) -> str:
        return "\n".join(
            [
                "【LLM 故障告警】",
                f"模型：{target.model}",
                f"Provider：{target.provider}",
                f"阶段：{stage or target.stage_label}",
                "状态：重试后仍失败",
                f"时间：{_fmt_date_time(now)}",
                f"错误：{reason}",
            ]
        )

    def _format_recovery(self, target: LlmHealthTarget, outage: OutageWindow) -> str:
        ended_at = outage.ended_at or self._now()
        return "\n".join(
            [
                "【LLM 恢复通知】",
                f"模型：{target.model}",
                f"Provider：{target.provider}",
                f"故障时间：{_fmt_time(outage.started_at)}~{_fmt_time(ended_at)}（{_fmt_duration((ended_at - outage.started_at).total_seconds())}）",
                "当前状态：已恢复",
            ]
        )

    def _format_outages(self, state: LlmHealthState, start: datetime, end: datetime) -> str:
        parts = []
        for outage in state.outages:
            overlap = _overlap(outage, start, end, self._now())
            if overlap is None:
                continue
            overlap_start, overlap_end, seconds, open_ended = overlap
            end_text = "现在" if open_ended else _fmt_time(overlap_end)
            parts.append(f"{_fmt_time(overlap_start)}~{end_text}（{_fmt_duration(seconds)}）")
        return "无" if not parts else "；".join(parts)

    def _outage_seconds(self, state: LlmHealthState, start: datetime, end: datetime) -> float:
        total = 0.0
        now = self._now()
        for outage in state.outages:
            overlap = _overlap(outage, start, end, now)
            if overlap is not None:
                total += overlap[2]
        return total

    def _next_daily_report_at(self, now: datetime) -> datetime:
        report_time = _parse_hhmm(self.settings.llm_health_daily_report_time)
        candidate = datetime.combine(now.date(), report_time, tzinfo=self.tz)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    async def _sleep(self, seconds: float) -> None:
        if self._stop_event is None:
            await asyncio.sleep(seconds)
            return
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=max(0.1, seconds))
        except asyncio.TimeoutError:
            return

    @property
    def _stopped(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def _now(self) -> datetime:
        return datetime.now(self.tz)


_MONITOR: LlmHealthMonitor | None = None


def start_llm_health_monitor(settings: Settings) -> LlmHealthMonitor | None:
    global _MONITOR
    if not settings.llm_health_enabled:
        return None
    if _MONITOR is None:
        _MONITOR = LlmHealthMonitor(settings)
        _MONITOR.start()
    return _MONITOR


async def stop_llm_health_monitor() -> None:
    global _MONITOR
    monitor = _MONITOR
    _MONITOR = None
    if monitor is not None:
        await monitor.stop()


def record_llm_success_for_agent(agent, stage: str) -> None:
    if _MONITOR is not None:
        _MONITOR.mark_agent_success(agent, stage)


def record_llm_failure_for_agent(agent, stage: str, reason: str) -> None:
    if _MONITOR is not None:
        _MONITOR.mark_agent_failure(agent, stage, reason)


def discover_llm_health_targets(settings: Settings) -> tuple[LlmHealthTarget, ...]:
    rows = [
        (
            "default",
            "composer/alignment",
            settings.llm_provider,
            settings.llm_model,
            settings.llm_base_url,
            bool(settings.llm_api_key),
        ),
        (
            "planner",
            "planner",
            settings.planner_llm_provider,
            settings.planner_llm_model,
            settings.planner_llm_base_url,
            bool(settings.planner_llm_api_key),
        ),
    ]

    targets: dict[tuple[str, str, str], LlmHealthTarget] = {}
    for kind, stage_label, provider, model, base_url, api_key_configured in rows:
        dedupe_key = (provider.lower(), model, base_url)
        existing = targets.get(dedupe_key)
        if existing is None:
            targets[dedupe_key] = LlmHealthTarget(
                key=_target_key(provider, model, base_url),
                kind=kind,  # type: ignore[arg-type]
                stage_label=stage_label,
                provider=provider,
                model=model,
                base_url=base_url,
                api_key_configured=api_key_configured,
            )
            continue
        targets[dedupe_key] = LlmHealthTarget(
            key=existing.key,
            kind=existing.kind,
            stage_label=f"{existing.stage_label}/{stage_label}",
            provider=existing.provider,
            model=existing.model,
            base_url=existing.base_url,
            api_key_configured=existing.api_key_configured or api_key_configured,
        )

    return tuple(targets.values())


def _probe_retry_reason(result) -> str | None:
    parsed = getattr(result, "pydantic", None)
    if isinstance(parsed, HealthProbeResponse) and parsed.ok:
        return None
    return "健康检查返回异常"


def _health_probe_program() -> PromptProgram:
    text = 'Return JSON only: {"ok": true}'
    return PromptProgram(
        profile=PromptProfile(
            id="llm_health_probe.generic",
            stage="guardrail",
            base_template_name="",
            response_model=HealthProbeResponse,
            model_family="generic",
            temperature=0.0,
            max_tokens=64,
        ),
        fragment_ids=(),
        prompt_text=text,
        prompt_hash=_sha256(text),
        fragment_hashes={},
        layers=("task",),
    )


def _target_key(provider: str, model: str, base_url: str) -> str:
    raw = f"{provider.lower()}|{model}|{base_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _agent_base_url(llm) -> str:
    base_url = str(getattr(llm, "base_url", "") or "")
    if base_url:
        return base_url
    client_params = getattr(llm, "client_params", None)
    if isinstance(client_params, dict):
        return str(
            client_params.get("http_options", {}).get("base_url", "") or ""
        )
    return ""


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _timezone(name: str):
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return timezone.utc


def _parse_hhmm(value: str) -> time:
    try:
        hour, minute = value.split(":", 1)
        return time(hour=int(hour), minute=int(minute))
    except Exception:
        return time(hour=9, minute=0)


def _overlap(
    outage: OutageWindow,
    start: datetime,
    end: datetime,
    now: datetime,
) -> tuple[datetime, datetime, float, bool] | None:
    outage_end = outage.ended_at or now
    overlap_start = max(outage.started_at, start)
    overlap_end = min(outage_end, end)
    seconds = (overlap_end - overlap_start).total_seconds()
    if seconds <= 0:
        return None
    return overlap_start, overlap_end, seconds, outage.ended_at is None


def _status_text(status: str) -> str:
    if status == "healthy":
        return "正常"
    if status == "unhealthy":
        return "异常"
    return "暂无数据"


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分钟")
    if not parts:
        parts.append(f"{seconds}秒")
    return "".join(parts)


def _fmt_time(value: datetime) -> str:
    return value.strftime("%H:%M")


def _fmt_date_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")
