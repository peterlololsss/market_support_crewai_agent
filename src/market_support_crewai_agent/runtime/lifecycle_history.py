from __future__ import annotations

from datetime import UTC, datetime

from market_support_crewai_agent.runtime.state.conversation_records import (
    ConversationTurnRecordV2,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)


def history_from_snapshot(
    turns: tuple[ConversationTurnRecordV2, ...],
) -> list[ConversationMessage]:
    return [
        ConversationMessage(
            role=turn.role,
            content=turn.text,
            created_at=datetime.fromtimestamp(turn.created_at_epoch_ms / 1_000, tz=UTC),
        )
        for turn in turns
    ]
