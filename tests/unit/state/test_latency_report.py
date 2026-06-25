from __future__ import annotations

import json

from scripts.report_runtime_latency import _read_log_text
from market_support_crewai_agent.runtime.state.latency_report import (
    format_latency_report,
    latency_report_from_lines,
)


def _log_line(context_id: str, total_ms: float, planner_ms: float) -> str:
    payload = {
        "context_id": context_id,
        "conversation_key": f"conv-{context_id}",
        "runtime_trace": {
            "schema_version": "runtime-trace-v1",
            "total_ms": total_ms,
            "events": [
                {"type": "span", "name": "planner", "duration_ms": planner_ms},
                {"type": "event", "name": "ignored"},
                {"type": "span", "name": "composer", "duration_ms": 20.0},
            ],
        },
    }
    return "INFO agent_runtime_trace " + json.dumps(payload, separators=(",", ":"))


def test_latency_report_summarizes_recent_runtime_trace_spans():
    report = latency_report_from_lines(
        [_log_line("old", 10.0, 5.0), _log_line("new", 100.0, 80.0)],
        limit=1,
    )

    assert report["trace_count"] == 1
    assert report["slowest_traces"][0]["context_id"] == "new"
    assert report["top_spans"][0] == {
        "name": "planner",
        "count": 1,
        "total_ms": 80.0,
        "avg_ms": 80.0,
        "max_ms": 80.0,
    }


def test_format_latency_report_is_human_readable():
    report = latency_report_from_lines([_log_line("ctx", 120.0, 50.0)])

    output = format_latency_report(report, top=2)

    assert "traces: 1" in output
    assert "Slowest traces" in output
    assert "Top spans by total_ms" in output
    assert "planner" in output


def test_report_cli_reads_powershell_utf16_logs(tmp_path):
    path = tmp_path / "server.log"
    path.write_text(_log_line("ctx", 120.0, 50.0), encoding="utf-16")

    report = latency_report_from_lines(_read_log_text(path).splitlines())

    assert report["trace_count"] == 1


def test_latency_report_ignores_live_runtime_events():
    live_event = (
        'INFO agent_runtime_event '
        '{"schema_version":"runtime-trace-v1","seq":1,"type":"span_end","duration_ms":5}'
    )

    report = latency_report_from_lines([live_event, _log_line("ctx", 120.0, 50.0)])

    assert report["trace_count"] == 1
