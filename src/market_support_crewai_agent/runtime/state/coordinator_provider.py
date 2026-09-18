from threading import RLock

from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.settings import get_settings

_coordinator_lock = RLock()
_coordinator: ReplyStateTransactionCoordinatorV1 | None = None
_coordinator_settings: tuple[int, int, int, int, int, int, int] | None = None


def get_reply_state_coordinator() -> ReplyStateTransactionCoordinatorV1:
    settings = get_settings()
    settings_key = (
        settings.issued_response_capacity,
        settings.agent_conversation_max_sessions,
        settings.feedback_receipt_capacity,
        settings.agent_conversation_ttl_seconds,
        settings.agent_conversation_max_messages,
        settings.issued_response_ttl_seconds,
        settings.issued_response_pending_ttl_seconds,
    )
    global _coordinator, _coordinator_settings
    with _coordinator_lock:
        if _coordinator is None or _coordinator_settings != settings_key:
            _coordinator = ReplyStateTransactionCoordinatorV1(
                issued_response_capacity=settings.issued_response_capacity,
                conversation_max_sessions=settings.agent_conversation_max_sessions,
                feedback_receipt_capacity=settings.feedback_receipt_capacity,
                conversation_ttl_seconds=settings.agent_conversation_ttl_seconds,
                conversation_max_messages=settings.agent_conversation_max_messages,
                issued_response_ttl_seconds=settings.issued_response_ttl_seconds,
                issued_response_pending_ttl_seconds=(
                    settings.issued_response_pending_ttl_seconds
                ),
            )
            _coordinator_settings = settings_key
        return _coordinator
