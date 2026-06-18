from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from time import perf_counter
from typing import Any

RUNTIME_TRACE_VERSION = "runtime-trace-v1"
TRACE_LOG_EVENTS_ENV = "MARKET_AGENT_TRACE_LOG_EVENTS"
_LIVE_LOGGER = logging.getLogger("market_support_crewai_agent.runtime_trace")

_CURRENT_TRACE: ContextVar[RuntimeTrace | None] = ContextVar(
    "market_agent_runtime_trace",
    default=None,
)


@dataclass
class RuntimeTrace:
    context: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._started_at = perf_counter()
        self._lock = RLock()
        self._seq = 0
        self.context = _safe_attrs(self.context)

    def span(self, name: str, **attrs: Any) -> "_TraceSpan":
        event = {
            "type": "span",
            "name": name,
            "status": "running",
            "started_at": _utc_now(),
            "attrs": _safe_attrs(attrs),
        }
        return _TraceSpan(self, event)

    def event(self, name: str, **attrs: Any) -> None:
        event = {
            "type": "event",
            "name": name,
            "at": _utc_now(),
            "attrs": _safe_attrs(attrs),
        }
        with self._lock:
            self.events.append(event)
        self._emit_live(event)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            events = [dict(event) for event in self.events]
        return {
            "schema_version": RUNTIME_TRACE_VERSION,
            "total_ms": round((perf_counter() - self._started_at) * 1000, 3),
            "events": events,
        }

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
        logger.info(
            "agent_runtime_trace %s",
            json.dumps(
                {
                    "context_id": context_id,
                    "conversation_key": conversation_key,
                    "runtime_trace": payload,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    def _emit_live(self, event: dict[str, Any]) -> None:
        if not _trace_log_events_enabled() or not _LIVE_LOGGER.isEnabledFor(logging.INFO):
            return
        with self._lock:
            self._seq += 1
            seq = self._seq
        payload = {
            "schema_version": RUNTIME_TRACE_VERSION,
            "seq": seq,
            "context": self.context,
            **event,
        }
        _LIVE_LOGGER.info(
            "agent_runtime_event %s",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def _finish_span(
        self,
        event: dict[str, Any],
        started_at: float,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        with self._lock:
            event["status"] = status
            event["duration_ms"] = round((perf_counter() - started_at) * 1000, 3)
            event["ended_at"] = _utc_now()
            if error:
                event["error"] = error


class _TraceScope:
    def __init__(self, trace: RuntimeTrace) -> None:
        self.trace = trace
        self.token = None

    def __enter__(self) -> RuntimeTrace:
        self.token = _CURRENT_TRACE.set(self.trace)
        return self.trace

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.token is not None:
            _CURRENT_TRACE.reset(self.token)
        return False


class _TraceSpan:
    def __init__(self, trace: RuntimeTrace, event: dict[str, Any]) -> None:
        self.trace = trace
        self.event = event
        self.started_at = 0.0

    def __enter__(self) -> None:
        self.started_at = perf_counter()
        with self.trace._lock:
            self.trace.events.append(self.event)
        self.trace._emit_live(
            {
                "type": "span_start",
                "name": self.event["name"],
                "at": self.event["started_at"],
                "attrs": self.event.get("attrs", {}),
            }
        )

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            self.trace._finish_span(
                self.event,
                self.started_at,
                "error",
                error=_short_text(exc),
            )
        else:
            self.trace._finish_span(self.event, self.started_at, "ok")
        self.trace._emit_live(
            {
                "type": "span_end",
                "name": self.event["name"],
                "at": self.event["ended_at"],
                "status": self.event["status"],
                "duration_ms": self.event["duration_ms"],
                "attrs": self.event.get("attrs", {}),
                **({"error": self.event["error"]} if "error" in self.event else {}),
            }
        )
        return False


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


def use_runtime_trace(trace: RuntimeTrace) -> _TraceScope:
    return _TraceScope(trace)


def trace_span(name: str, **attrs: Any) -> _TraceSpan | _NullSpan:
    trace = _CURRENT_TRACE.get()
    if trace is None:
        return _NullSpan()
    return trace.span(name, **attrs)


def trace_event(name: str, **attrs: Any) -> None:
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        trace.event(name, **attrs)


def current_runtime_trace() -> RuntimeTrace | None:
    return _CURRENT_TRACE.get()


def _safe_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _safe_value(value)
        for key, value in attrs.items()
        if not _looks_sensitive(str(key))
    }


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 300 else value[:297] + "..."
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in list(value)[:20]]
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in list(value.items())[:50]
            if not _looks_sensitive(str(key))
        }
    return _short_text(value)


def _looks_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in ("api_key", "token", "secret", "password"))


def _trace_log_events_enabled() -> bool:
    return os.getenv(TRACE_LOG_EVENTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _short_text(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 300 else text[:297] + "..."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
