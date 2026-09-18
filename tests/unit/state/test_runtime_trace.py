import json
import logging

import pytest

from market_support_crewai_agent.runtime.observability.runtime_trace import RuntimeTrace


class _RaisingTraceHandler(logging.Handler):
    def __init__(self, error_type: type[BaseException]) -> None:
        super().__init__()
        self.error_type = error_type

    def emit(self, record: logging.LogRecord) -> None:
        del record
        raise self.error_type("trace_handler_failure")


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


@pytest.mark.parametrize("error_type", (OSError, RuntimeError))
def test_runtime_trace_operational_handler_failure_stays_observational(
    monkeypatch, error_type
):
    monkeypatch.setenv("MARKET_AGENT_TRACE_LOG_EVENTS", "true")
    live_logger = logging.getLogger("market_support_crewai_agent.runtime_trace")
    trace_logger = logging.getLogger("market_support_crewai_agent.runtime_trace.test")
    live_handler = _RaisingTraceHandler(error_type)
    trace_handler = _RaisingTraceHandler(error_type)
    original_live_level = live_logger.level
    original_trace_level = trace_logger.level
    original_live_propagate = live_logger.propagate
    original_trace_propagate = trace_logger.propagate
    live_logger.setLevel(logging.INFO)
    trace_logger.setLevel(logging.INFO)
    live_logger.propagate = False
    trace_logger.propagate = False
    live_logger.addHandler(live_handler)
    trace_logger.addHandler(trace_handler)
    try:
        trace = RuntimeTrace()
        with trace.span("planner.step"):
            trace.event("state.ready")
        trace.log_trace(trace_logger, context_id="ctx-1", conversation_key="key-1")
    finally:
        live_logger.removeHandler(live_handler)
        trace_logger.removeHandler(trace_handler)
        live_logger.setLevel(original_live_level)
        trace_logger.setLevel(original_trace_level)
        live_logger.propagate = original_live_propagate
        trace_logger.propagate = original_trace_propagate

    assert [event["type"] for event in trace.to_dict()["events"]] == ["span", "event"]


@pytest.mark.parametrize(
    "error_type", (SystemExit, KeyboardInterrupt, TypeError, AttributeError, KeyError)
)
def test_runtime_trace_control_or_programming_handler_failure_propagates(
    monkeypatch, error_type
):
    monkeypatch.setenv("MARKET_AGENT_TRACE_LOG_EVENTS", "true")
    live_logger = logging.getLogger("market_support_crewai_agent.runtime_trace")
    handler = _RaisingTraceHandler(error_type)
    original_level = live_logger.level
    original_propagate = live_logger.propagate
    live_logger.setLevel(logging.INFO)
    live_logger.propagate = False
    live_logger.addHandler(handler)
    try:
        with pytest.raises(error_type):
            RuntimeTrace().event("state.ready")
    finally:
        live_logger.removeHandler(handler)
        live_logger.setLevel(original_level)
        live_logger.propagate = original_propagate
