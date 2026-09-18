from dataclasses import dataclass

from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorConfigError,
)


@dataclass(frozen=True, slots=True)
class CoordinatorConfigV1:
    issued_response_capacity: int
    conversation_max_sessions: int
    feedback_receipt_capacity: int
    conversation_ttl_ms: int
    conversation_max_messages: int
    issued_ttl_ns: int
    issued_pending_ttl_ns: int
    state_revision_capacity: int


def build_coordinator_config(
    *,
    issued_response_capacity: int,
    conversation_max_sessions: int,
    feedback_receipt_capacity: int,
    conversation_ttl_seconds: int,
    conversation_max_messages: int,
    issued_response_ttl_seconds: int,
    issued_response_pending_ttl_seconds: int,
) -> CoordinatorConfigV1:
    if issued_response_capacity < 1:
        raise CoordinatorConfigError("issued_response_capacity must be positive")
    if conversation_max_sessions < 1:
        raise CoordinatorConfigError("conversation_max_sessions must be positive")
    if feedback_receipt_capacity < issued_response_capacity:
        raise CoordinatorConfigError(
            "feedback_receipt_capacity must cover issued_response_capacity"
        )
    if conversation_ttl_seconds < 1 or conversation_max_messages < 1:
        raise CoordinatorConfigError("conversation retention must be positive")
    if issued_response_ttl_seconds < 60 or issued_response_pending_ttl_seconds < 5:
        raise CoordinatorConfigError(
            "issued response retention is below the contract minimum"
        )
    if issued_response_pending_ttl_seconds > issued_response_ttl_seconds:
        raise CoordinatorConfigError(
            "issued pending retention cannot exceed complete retention"
        )
    return CoordinatorConfigV1(
        issued_response_capacity=issued_response_capacity,
        conversation_max_sessions=conversation_max_sessions,
        feedback_receipt_capacity=feedback_receipt_capacity,
        conversation_ttl_ms=conversation_ttl_seconds * 1_000,
        conversation_max_messages=conversation_max_messages,
        issued_ttl_ns=issued_response_ttl_seconds * 1_000_000_000,
        issued_pending_ttl_ns=issued_response_pending_ttl_seconds * 1_000_000_000,
        state_revision_capacity=(
            conversation_max_sessions + issued_response_capacity + 5_000 + 1
        ),
    )
