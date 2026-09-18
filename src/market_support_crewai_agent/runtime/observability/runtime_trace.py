from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from threading import RLock
from time import perf_counter
from types import TracebackType
from typing import Literal, final

from market_support_crewai_agent.runtime.observability.runtime_trace_direct import (
    DIRECT_SPAN_FAILURE_STATUS,
    DIRECT_SPAN_OK_STATUS,
    DIRECT_SPAN_START_STATUS,
    RuntimeTracePayload,
    TraceAttributes,
    TraceEvent,
    TraceEventType,
    TraceLogEnvelope,
    TraceValue,
    direct_trace_name,
)
from market_support_crewai_agent.runtime.observability.runtime_trace_sanitization import (
    emit_trace_log,
    safe_attrs,
    short_text,
    trace_event_log_payload,
    trace_json,
    trace_log_events_enabled,
)

RUNTIME_TRACE_VERSION = "runtime-trace-v1"
_LIVE_LOGGER = logging.getLogger("market_support_crewai_agent.runtime_trace")
_CURRENT_TRACE: ContextVar[RuntimeTrace | None] = ContextVar(
    "market_agent_runtime_trace", default=None
)


@final
class RuntimeTrace:
    def __init__(
        self,
        context: TraceAttributes | None = None,
        events: list[TraceEvent] | None = None,
        direct: bool = False,
    ) -> None:
        self.context: TraceAttributes = safe_attrs(
            {} if context is None else context, direct=direct
        )
        self.events: list[TraceEvent] = [] if events is None else events
        self.direct: bool = direct
        self._started_at = perf_counter()
        self._lock = RLock()
        self._seq = 0

    def span(self, name: str, **attrs: TraceValue) -> _TraceSpan:
        event: TraceEvent = {
            "type": "span",
            "name": direct_trace_name(name, is_span=True) if self.direct else name,
            "status": DIRECT_SPAN_START_STATUS if self.direct else "running",
            "started_at": _utc_now(),
            "attrs": safe_attrs(attrs, direct=self.direct),
        }
        return _TraceSpan(self, event)

    def event(self, name: str, **attrs: TraceValue) -> None:
        event: TraceEvent = {
            "type": "event",
            "name": direct_trace_name(name, is_span=False) if self.direct else name,
            "at": _utc_now(),
            "attrs": safe_attrs(attrs, direct=self.direct),
        }
        self.append_event(event)
        self.emit_live(event)

    def to_dict(self) -> RuntimeTracePayload:
        with self._lock:
            events = [
                _direct_trace_event(event) if self.direct else event.copy()
                for event in self.events
            ]
        return RuntimeTracePayload(
            schema_version=RUNTIME_TRACE_VERSION,
            total_ms=round((perf_counter() - self._started_at) * 1000, 3),
            events=events,
        )

    def log_trace(
        self,
        logger: logging.Logger,
        *,
        context_id: str | None,
        conversation_key: str,
    ) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        payload = self.to_dict()
        envelope: TraceLogEnvelope
        if self.direct:
            envelope = {"runtime_trace": payload, "redacted": True}
        else:
            envelope = {
                "context_id": context_id,
                "conversation_key": conversation_key,
                "runtime_trace": payload,
            }
        emit_trace_log(logger, "agent_runtime_trace %s", trace_json(envelope))

    def emit_live(self, event: TraceEvent) -> None:
        if not trace_log_events_enabled() or not _LIVE_LOGGER.isEnabledFor(
            logging.INFO
        ):
            return
        with self._lock:
            self._seq += 1
            sequence = self._seq
        payload = trace_event_log_payload(
            RUNTIME_TRACE_VERSION,
            sequence,
            safe_attrs(self.context, direct=True) if self.direct else self.context,
            _direct_trace_event(event) if self.direct else event,
        )
        emit_trace_log(_LIVE_LOGGER, "agent_runtime_event %s", trace_json(payload))

    def append_event(self, event: TraceEvent) -> None:
        with self._lock:
            self.events.append(event)

    def finish_span(
        self,
        event: TraceEvent,
        started_at: float,
        status: str,
        error: str | None = None,
    ) -> None:
        with self._lock:
            event["status"] = (
                DIRECT_SPAN_FAILURE_STATUS
                if self.direct and status == "error"
                else DIRECT_SPAN_OK_STATUS
                if self.direct
                else status
            )
            event["duration_ms"] = round((perf_counter() - started_at) * 1000, 3)
            event["ended_at"] = _utc_now()
            if error is not None and not self.direct:
                event["error"] = error


@final
class _TraceScope:
    def __init__(self, trace: RuntimeTrace) -> None:
        self.trace: RuntimeTrace = trace
        self.token: Token[RuntimeTrace | None] | None = None

    def __enter__(self) -> RuntimeTrace:
        self.token = _CURRENT_TRACE.set(self.trace)
        return self.trace

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self.token is not None:
            _CURRENT_TRACE.reset(self.token)
        return False


@final
class _TraceSpan:
    def __init__(self, trace: RuntimeTrace, event: TraceEvent) -> None:
        self.trace: RuntimeTrace = trace
        self.event: TraceEvent = event
        self.started_at: float = 0.0

    def __enter__(self) -> None:
        self.started_at = perf_counter()
        self.trace.append_event(self.event)
        self.trace.emit_live(
            TraceEvent(
                type="span_start",
                name=self.event["name"],
                at=self.event.get("started_at", ""),
                attrs=self.event.get("attrs", {}),
            )
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if exc is not None:
            self.trace.finish_span(
                self.event, self.started_at, "error", short_text(exc)
            )
        else:
            self.trace.finish_span(self.event, self.started_at, "ok")
        finished = TraceEvent(
            type="span_end",
            name=self.event["name"],
            at=self.event.get("ended_at", ""),
            status=self.event.get("status", ""),
            duration_ms=self.event.get("duration_ms", 0.0),
            attrs=self.event.get("attrs", {}),
        )
        if "error" in self.event:
            finished["error"] = self.event["error"]
        self.trace.emit_live(finished)
        return False


@final
class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False


def use_runtime_trace(trace: RuntimeTrace) -> _TraceScope:
    return _TraceScope(trace)


def trace_span(name: str, **attrs: TraceValue) -> _TraceSpan | _NullSpan:
    trace = _CURRENT_TRACE.get()
    return _NullSpan() if trace is None else trace.span(name, **attrs)


def trace_event(name: str, **attrs: TraceValue) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.event(name, **attrs)


def current_runtime_trace() -> RuntimeTrace | None:
    return _CURRENT_TRACE.get()


def _direct_trace_event(event: TraceEvent) -> TraceEvent:
    safe_event_type: TraceEventType
    match event.get("type"):
        case ("event" | "span" | "span_start" | "span_end") as event_type:
            safe_event_type = event_type
        case _:
            safe_event_type = "event"
    payload = TraceEvent(
        type=safe_event_type,
        name=direct_trace_name("", is_span=safe_event_type != "event"),
        attrs={},
    )
    for key in ("at", "started_at", "ended_at", "duration_ms"):
        value = event.get(key)
        if value is not None:
            payload[key] = value
    if safe_event_type != "event":
        status = event.get("status")
        payload["status"] = (
            DIRECT_SPAN_START_STATUS
            if safe_event_type == "span_start"
            else status
            if status
            in {
                DIRECT_SPAN_START_STATUS,
                DIRECT_SPAN_OK_STATUS,
                DIRECT_SPAN_FAILURE_STATUS,
            }
            else DIRECT_SPAN_FAILURE_STATUS
        )
    payload["attrs"] = safe_attrs(event["attrs"], direct=True)
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
