from __future__ import annotations

from typing import Final, Literal, NotRequired, TypeAlias, TypedDict

TraceValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["TraceValue"]
    | tuple["TraceValue", ...]
    | dict[str, "TraceValue"]
)
TraceAttributes: TypeAlias = dict[str, TraceValue]
TraceEventType: TypeAlias = Literal["event", "span", "span_start", "span_end"]


class TraceEvent(TypedDict):
    type: str
    name: str
    attrs: TraceAttributes
    status: NotRequired[str]
    at: NotRequired[str]
    started_at: NotRequired[str]
    ended_at: NotRequired[str]
    duration_ms: NotRequired[float]
    error: NotRequired[str]


class RuntimeTracePayload(TypedDict):
    schema_version: str
    total_ms: float
    events: list[TraceEvent]


class TraceLogEnvelope(TypedDict, total=False):
    context_id: str | None
    conversation_key: str
    runtime_trace: RuntimeTracePayload
    redacted: bool


class LiveTracePayload(TypedDict, total=False):
    schema_version: str
    seq: int
    context: TraceAttributes
    type: str
    name: str
    attrs: TraceAttributes
    status: str
    at: str
    started_at: str
    ended_at: str
    duration_ms: float
    error: str


DIRECT_EVENT_NAME: Final = "direct.runtime_event"
DIRECT_SPAN_NAME: Final = "direct.runtime_span"
DIRECT_SPAN_START_STATUS: Final = "started"
DIRECT_SPAN_OK_STATUS: Final = "completed"
DIRECT_SPAN_FAILURE_STATUS: Final = "failed"
_DIRECT_NUMERIC_ATTR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "action_count",
        "action_history_count",
        "adapter_resolve_count",
        "alignment_verdict_count",
        "candidate_count",
        "evidence_fact_count",
        "guardrail_decision_count",
        "history_count",
        "issue_count",
        "request_count",
    }
)
_DIRECT_DAH1_ATTR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "dah1",
        "message_dah1",
        "request_id_dah1",
        "request_dah1",
        "reply_dah1",
        "evidence_dah1",
        "directive_dah1",
        "plan_dah1",
        "planner_input_dah1",
        "planner_output_dah1",
        "composer_input_dah1",
        "composer_output_dah1",
        "verifier_input_dah1",
        "verifier_output_dah1",
        "approved_selector_input_dah1",
        "approved_selector_output_dah1",
        "document_selector_input_dah1",
        "document_selector_output_dah1",
    }
)


def direct_attr_allowed(key: str, value: TraceValue) -> bool:
    if key in _DIRECT_NUMERIC_ATTR_KEYS:
        return type(value) is int and value >= 0
    if key in _DIRECT_DAH1_ATTR_KEYS:
        return isinstance(value, str) and is_direct_audit_digest(value)
    return False


def is_direct_audit_digest(value: str) -> bool:
    if len(value) != 69 or not value.startswith("dah1:"):
        return False
    return all(character in "0123456789abcdef" for character in value[5:])


def direct_trace_name(name: str, *, is_span: bool) -> str:
    del name
    return DIRECT_SPAN_NAME if is_span else DIRECT_EVENT_NAME
