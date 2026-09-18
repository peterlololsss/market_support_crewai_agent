from __future__ import annotations

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    IssuedFoundV1,
    IssuedIdentityMismatchV1,
    IssuedNotFoundV1,
    IssuedResponseLookupV1,
)


class IssuedResponseStoreV1:
    def lookup(
        self,
        root: CoordinatorStateRootV1,
        state_key: ConversationStateKey,
        response_id: str,
    ) -> IssuedResponseLookupV1:
        indexed = root.response_index.get(response_id)
        if indexed is None:
            return IssuedNotFoundV1()
        indexed_state_key, request_id = indexed
        if indexed_state_key != state_key:
            return IssuedIdentityMismatchV1()
        record = root.issued_records.get((indexed_state_key, request_id))
        if record is None or record.phase != "complete":
            return IssuedNotFoundV1()
        return IssuedFoundV1(record=record)
