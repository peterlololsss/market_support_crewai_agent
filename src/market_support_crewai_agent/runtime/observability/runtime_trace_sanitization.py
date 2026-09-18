from __future__ import annotations

import json
import logging
import os
from types import TracebackType
from typing import Final, final

from market_support_crewai_agent.runtime.observability.runtime_trace_direct import (
    LiveTracePayload,
    TraceAttributes,
    TraceEvent,
    TraceLogEnvelope,
    TraceValue,
    direct_attr_allowed,
)

TRACE_LOG_EVENTS_ENV: Final = "MARKET_AGENT_TRACE_LOG_EVENTS"
_SENSITIVE_KEY_PARTS: Final[tuple[str, ...]] = (
    "api_key",
    "token",
    "secret",
    "password",
)


def safe_attrs(attrs: TraceAttributes, *, direct: bool) -> TraceAttributes:
    if direct:
        return {
            str(key): value
            for key, value in attrs.items()
            if direct_attr_allowed(str(key), value)
        }
    return {
        str(key): safe_value(value)
        for key, value in attrs.items()
        if not looks_sensitive(str(key))
    }


def safe_value(value: TraceValue) -> TraceValue:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 300 else value[:297] + "..."
    if isinstance(value, list | tuple):
        return [safe_value(item) for item in value[:20]]
    return {
        str(key): safe_value(item)
        for key, item in list(value.items())[:50]
        if not looks_sensitive(str(key))
    }


def looks_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS)


def short_text(value: BaseException) -> str:
    text = str(value)
    return text if len(text) <= 300 else text[:297] + "..."


def trace_log_events_enabled() -> bool:
    return os.getenv(TRACE_LOG_EVENTS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def trace_json(payload: TraceLogEnvelope | LiveTracePayload) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def trace_event_log_payload(
    schema_version: str,
    sequence: int,
    context: TraceAttributes,
    event: TraceEvent,
) -> LiveTracePayload:
    payload: LiveTracePayload = {
        "schema_version": schema_version,
        "seq": sequence,
        "context": context,
        "type": event["type"],
        "name": event["name"],
        "attrs": event["attrs"],
    }
    for key in ("status", "at", "started_at", "ended_at", "duration_ms", "error"):
        value = event.get(key)
        if value is not None:
            payload[key] = value
    return payload


@final
class _ObservabilitySinkBoundary:
    """Contain ordinary handler I/O/runtime failures without suppressing control flow."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc, traceback
        if exc_type is None:
            return False
        return issubclass(exc_type, (OSError, RuntimeError))


def emit_trace_log(logger: logging.Logger, template: str, payload: str) -> None:
    with _ObservabilitySinkBoundary():
        logger.info(template, payload)
