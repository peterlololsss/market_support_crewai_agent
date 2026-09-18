from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from market_support_crewai_agent.health.models import (
    HealthStatus,
    LlmHealthState,
    LlmHealthTarget,
    OutageWindow,
)


@dataclass(frozen=True, slots=True)
class HealthReportTarget:
    target: LlmHealthTarget
    state: LlmHealthState


def format_daily_report(
    targets: tuple[HealthReportTarget, ...],
    start: datetime,
    end: datetime,
    now: datetime,
) -> str:
    lines = [f"【LLM 健康日报】{fmt_date_time(start)} ~ {fmt_date_time(end)}", ""]
    total_seconds = max(1.0, (end - start).total_seconds())

    for entry in targets:
        outage_seconds = outage_seconds_between(entry.state, start, end, now)
        availability = max(
            0.0,
            100.0 * (total_seconds - outage_seconds) / total_seconds,
        )
        lines.append(
            f"{entry.target.target_slot}（{entry.target.htk1}）：{status_text(entry.state.status)}"
        )
        lines.extend(_state_lines(entry.state, start, end, now, availability))
        if entry.state.last_error:
            lines.append(f"- 最后错误：{entry.state.last_error}")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_warning(target: LlmHealthTarget, reason: str, now: datetime) -> str:
    return "\n".join(
        [
            "【LLM 故障告警】",
            f"目标：{target.target_slot}",
            f"目标标识：{target.htk1}",
            "状态：重试后仍失败",
            f"时间：{fmt_date_time(now)}",
            f"错误：{reason}",
        ]
    )


def format_recovery(
    target: LlmHealthTarget, outage: OutageWindow, now: datetime
) -> str:
    ended_at = outage.ended_at or now
    duration = fmt_duration((ended_at - outage.started_at).total_seconds())
    return "\n".join(
        [
            "【LLM 恢复通知】",
            f"目标：{target.target_slot}",
            f"目标标识：{target.htk1}",
            f"故障时间：{fmt_time(outage.started_at)}~{fmt_time(ended_at)}（{duration}）",
            "当前状态：已恢复",
        ]
    )


def next_daily_report_at(
    value: datetime, report_time_text: str, tz: ZoneInfo | timezone
) -> datetime:
    report_time = parse_hhmm(report_time_text)
    candidate = datetime.combine(value.date(), report_time, tzinfo=tz)
    if candidate <= value:
        candidate += timedelta(days=1)
    return candidate


def timezone_for(name: str) -> ZoneInfo | timezone:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return UTC


def parse_hhmm(value: str) -> time:
    try:
        hour, minute = value.split(":", 1)
        return time(hour=int(hour), minute=int(minute))
    except ValueError:
        return time(hour=9, minute=0)


def status_text(status: HealthStatus) -> str:
    match status:
        case "healthy":
            return "正常"
        case "unhealthy":
            return "异常"
        case "unknown":
            return "暂无数据"


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分钟")
    if not parts:
        parts.append(f"{seconds}秒")
    return "".join(parts)


def fmt_time(value: datetime) -> str:
    return value.strftime("%H:%M")


def fmt_date_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


def outage_seconds_between(
    state: LlmHealthState,
    start: datetime,
    end: datetime,
    now: datetime,
) -> float:
    total = 0.0
    for outage in state.outages:
        overlap = overlap_outage(outage, start, end, now)
        if overlap is not None:
            total += overlap.seconds
    return total


@dataclass(frozen=True, slots=True)
class OutageOverlap:
    started_at: datetime
    ended_at: datetime
    seconds: float
    open_ended: bool


def overlap_outage(
    outage: OutageWindow,
    start: datetime,
    end: datetime,
    now: datetime,
) -> OutageOverlap | None:
    outage_end = outage.ended_at or now
    overlap_start = max(outage.started_at, start)
    overlap_end = min(outage_end, end)
    seconds = (overlap_end - overlap_start).total_seconds()
    if seconds <= 0:
        return None
    return OutageOverlap(overlap_start, overlap_end, seconds, outage.ended_at is None)


def _state_lines(
    state: LlmHealthState,
    start: datetime,
    end: datetime,
    now: datetime,
    availability: float,
) -> list[str]:
    if state.status == "healthy" and state.healthy_since is not None:
        return [
            f"- 连续可用：{fmt_duration((end - state.healthy_since).total_seconds())}",
            f"- 不可用时间段：{format_outages(state, start, end, now)}",
            f"- 今日总不可用：{fmt_duration(outage_seconds_between(state, start, end, now))}",
            f"- 可用率：{availability:.2f}%",
        ]
    if state.status == "unhealthy" and state.unhealthy_since is not None:
        return [
            f"- 已不可用：{fmt_duration((end - state.unhealthy_since).total_seconds())}",
            f"- 不可用时间段：{format_outages(state, start, end, now)}",
            f"- 今日总不可用：{fmt_duration(outage_seconds_between(state, start, end, now))}",
            f"- 可用率：{availability:.2f}%",
        ]
    return ["- 检测结果：尚未完成首次健康检查"]


def format_outages(
    state: LlmHealthState,
    start: datetime,
    end: datetime,
    now: datetime,
) -> str:
    parts: list[str] = []
    for outage in state.outages:
        overlap = overlap_outage(outage, start, end, now)
        if overlap is None:
            continue
        end_text = "现在" if overlap.open_ended else fmt_time(overlap.ended_at)
        parts.append(
            f"{fmt_time(overlap.started_at)}~{end_text}（{fmt_duration(overlap.seconds)}）"
        )
    return "无" if not parts else "；".join(parts)
