from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

from market_support_crewai_agent.runtime.state.runtime_trace import RUNTIME_TRACE_VERSION

_TRACE_MARKER = "agent_runtime_trace "


def latency_report_from_lines(lines: Iterable[str], *, limit: int = 20) -> dict[str, Any]:
    records = list(iter_runtime_trace_records(lines))
    if limit > 0:
        records = records[-limit:]

    trace_rows: list[dict[str, Any]] = []
    span_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "total_ms": 0.0, "max_ms": 0.0}
    )
    for record in records:
        trace = record["runtime_trace"]
        spans = [
            event
            for event in trace.get("events", [])
            if event.get("type") == "span" and _number(event.get("duration_ms")) is not None
        ]
        total_ms = _number(trace.get("total_ms"))
        if total_ms is None:
            total_ms = sum(float(event["duration_ms"]) for event in spans)
        trace_rows.append(
            {
                "context_id": record.get("context_id") or "",
                "conversation_key": record.get("conversation_key") or "",
                "total_ms": round(float(total_ms), 3),
                "span_count": len(spans),
            }
        )
        for span in spans:
            name = str(span.get("name") or "unknown")
            duration = float(span["duration_ms"])
            stats = span_stats[name]
            stats["count"] += 1
            stats["total_ms"] += duration
            stats["max_ms"] = max(stats["max_ms"], duration)

    spans = [
        {
            "name": name,
            "count": int(stats["count"]),
            "total_ms": round(stats["total_ms"], 3),
            "avg_ms": round(stats["total_ms"] / stats["count"], 3),
            "max_ms": round(stats["max_ms"], 3),
        }
        for name, stats in span_stats.items()
        if stats["count"]
    ]
    return {
        "trace_count": len(records),
        "slowest_traces": sorted(
            trace_rows,
            key=lambda row: row["total_ms"],
            reverse=True,
        )[:5],
        "top_spans": sorted(
            spans,
            key=lambda row: row["total_ms"],
            reverse=True,
        ),
    }


def iter_runtime_trace_records(lines: Iterable[str]) -> Iterable[dict[str, Any]]:
    for line in lines:
        payload = _json_payload_from_line(line)
        if not isinstance(payload, dict):
            continue
        trace = payload.get("runtime_trace")
        if (
            trace is None
            and payload.get("schema_version") == RUNTIME_TRACE_VERSION
            and isinstance(payload.get("events"), list)
        ):
            trace = payload
        if not isinstance(trace, dict):
            continue
        if trace.get("schema_version") != RUNTIME_TRACE_VERSION:
            continue
        yield {
            "context_id": payload.get("context_id"),
            "conversation_key": payload.get("conversation_key"),
            "runtime_trace": trace,
        }


def format_latency_report(report: dict[str, Any], *, top: int = 15) -> str:
    if not report.get("trace_count"):
        return "No runtime traces found."

    lines = [f"traces: {report['trace_count']}", "", "Slowest traces"]
    lines.append("total_ms  spans  context_id  conversation_key")
    for row in report.get("slowest_traces", []):
        lines.append(
            "{total_ms:>8.1f}  {span_count:>5}  {context_id}  {conversation_key}".format(
                **row
            ).rstrip()
        )

    lines.extend(["", "Top spans by total_ms"])
    lines.append("total_ms  avg_ms  max_ms  count  name")
    for row in report.get("top_spans", [])[:top]:
        lines.append(
            "{total_ms:>8.1f}  {avg_ms:>6.1f}  {max_ms:>6.1f}  {count:>5}  {name}".format(
                **row
            )
        )
    return "\n".join(lines)


def _json_payload_from_line(line: str) -> Any:
    text = line.strip()
    if not text:
        return None
    if _TRACE_MARKER in text:
        text = text.split(_TRACE_MARKER, 1)[1].strip()
    elif not text.startswith("{"):
        brace = text.find("{")
        if brace < 0:
            return None
        text = text[brace:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
