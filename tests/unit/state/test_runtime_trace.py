import json
import logging

from market_support_crewai_agent.runtime.state.runtime_trace import RuntimeTrace


def _event_payloads(caplog):
    payloads = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("agent_runtime_event "):
            payloads.append(json.loads(message.removeprefix("agent_runtime_event ")))
    return payloads


def test_runtime_trace_live_event_logging_is_centralized(monkeypatch, caplog):
    monkeypatch.setenv("MARKET_AGENT_TRACE_LOG_EVENTS", "true")
    caplog.set_level(logging.INFO, logger="market_support_crewai_agent.runtime_trace")
    trace = RuntimeTrace(context={"context_id": "ctx-1", "api_key": "secret"})

    with trace.span("planner.step", prompt_chars=123):
        trace.event("state.ready", token="secret", answer="ok")

    payloads = _event_payloads(caplog)
    assert [payload["type"] for payload in payloads] == [
        "span_start",
        "event",
        "span_end",
    ]
    assert [payload["seq"] for payload in payloads] == [1, 2, 3]
    assert payloads[0]["context"] == {"context_id": "ctx-1"}
    assert payloads[0]["attrs"] == {"prompt_chars": 123}
    assert payloads[1]["attrs"] == {"answer": "ok"}
    assert payloads[2]["status"] == "ok"
    assert "duration_ms" in payloads[2]


def test_runtime_trace_live_event_logging_defaults_off(caplog):
    caplog.set_level(logging.INFO, logger="market_support_crewai_agent.runtime_trace")
    trace = RuntimeTrace()

    with trace.span("silent"):
        pass

    assert _event_payloads(caplog) == []
