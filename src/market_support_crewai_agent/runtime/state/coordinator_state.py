from dataclasses import dataclass
from typing import Final, Literal

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.identity.state_key import (
    canonical_json_bytes,
    canonical_state_key_payload,
)
from market_support_crewai_agent.runtime.state.audit_records import AuditRecordV1
from market_support_crewai_agent.runtime.state.audit_types import (
    ReplyStateJournalOperationV1,
)
from market_support_crewai_agent.runtime.state.conversation_records import (
    ConversationTurnRecordV2,
    PendingClarificationRecordV1,
    StateRevisionRecordV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    ActionLedgerRecordV2,
    FeedbackReceiptV1,
    IssuedResponseRecordV1,
    PreparedFeedbackTokenRecordV1,
)
from market_support_crewai_agent.runtime.state.persistent_map import FrozenMapV1

RootRecordKeyV1 = (
    ConversationStateKey
    | tuple[ConversationStateKey, str]
    | tuple[ConversationStateKey, str, str]
)
RootOrderEntryV1 = tuple[int, bytes, RootRecordKeyV1]


@dataclass(frozen=True, slots=True)
class CoordinatorStateRootV1:
    root_revision: int
    issued_records: FrozenMapV1[
        tuple[ConversationStateKey, str], IssuedResponseRecordV1
    ]
    pending_request_by_state: FrozenMapV1[
        ConversationStateKey, tuple[ConversationStateKey, str]
    ]
    feedback_receipts: FrozenMapV1[tuple[ConversationStateKey, str], FeedbackReceiptV1]
    prepared_feedback_tokens: FrozenMapV1[
        tuple[ConversationStateKey, str], PreparedFeedbackTokenRecordV1
    ]
    state_revisions: FrozenMapV1[ConversationStateKey, StateRevisionRecordV1]
    response_index: FrozenMapV1[str, tuple[ConversationStateKey, str]]
    action_index: FrozenMapV1[str, tuple[ConversationStateKey, str, str]]
    conversation_turns: FrozenMapV1[
        ConversationStateKey, tuple[ConversationTurnRecordV2, ...]
    ]
    clarification_records: FrozenMapV1[
        ConversationStateKey, PendingClarificationRecordV1
    ]
    audit_records: FrozenMapV1[tuple[ConversationStateKey, str], AuditRecordV1]
    action_ledger_records: FrozenMapV1[
        tuple[ConversationStateKey, str, str], ActionLedgerRecordV2
    ]
    issued_expiry_order: tuple[RootOrderEntryV1, ...] = ()
    issued_complete_eviction_order: tuple[RootOrderEntryV1, ...] = ()
    conversation_expiry_order: tuple[RootOrderEntryV1, ...] = ()
    state_session_eviction_order: tuple[RootOrderEntryV1, ...] = ()
    clarification_expiry_order: tuple[RootOrderEntryV1, ...] = ()
    audit_expiry_order: tuple[RootOrderEntryV1, ...] = ()
    audit_eviction_order: tuple[RootOrderEntryV1, ...] = ()
    action_ledger_expiry_order: tuple[RootOrderEntryV1, ...] = ()
    action_ledger_eviction_order: tuple[RootOrderEntryV1, ...] = ()
    feedback_receipt_expiry_order: tuple[RootOrderEntryV1, ...] = ()
    prepared_token_expiry_order: tuple[RootOrderEntryV1, ...] = ()
    state_revision_expiry_order: tuple[RootOrderEntryV1, ...] = ()
    state_revision_eviction_order: tuple[RootOrderEntryV1, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplyStateJournalV2:
    transaction_id: bytes
    operation: ReplyStateJournalOperationV1
    initiating_state_key: ConversationStateKey | None
    affected_state_keys: tuple[ConversationStateKey, ...]
    prior_root: CoordinatorStateRootV1
    candidate_root: CoordinatorStateRootV1
    outcome: Literal["staging", "committed", "rolled_back"]
    contract_version: Literal["reply-state-journal.v2"] = "reply-state-journal.v2"


def empty_coordinator_root() -> CoordinatorStateRootV1:
    return CoordinatorStateRootV1(
        root_revision=0,
        issued_records=FrozenMapV1(),
        pending_request_by_state=FrozenMapV1(),
        feedback_receipts=FrozenMapV1(),
        prepared_feedback_tokens=FrozenMapV1(),
        state_revisions=FrozenMapV1(),
        response_index=FrozenMapV1(),
        action_index=FrozenMapV1(),
        conversation_turns=FrozenMapV1(),
        clarification_records=FrozenMapV1(),
        audit_records=FrozenMapV1(),
        action_ledger_records=FrozenMapV1(),
    )


EMPTY_COORDINATOR_ROOT: Final[CoordinatorStateRootV1] = empty_coordinator_root()


def state_session_eviction_order(
    turns: FrozenMapV1[ConversationStateKey, tuple[ConversationTurnRecordV2, ...]],
    clarifications: FrozenMapV1[ConversationStateKey, PendingClarificationRecordV1],
) -> tuple[tuple[int, bytes, ConversationStateKey], ...]:
    keys = frozenset((*turns, *clarifications))
    rows: list[tuple[int, bytes, ConversationStateKey]] = []
    for key in keys:
        turn_epoch = max(
            (turn.created_at_epoch_ms for turn in turns.get(key, ())), default=-1
        )
        clarification = clarifications.get(key)
        clarification_epoch = (
            clarification.created_at_epoch_ms if clarification is not None else -1
        )
        rows.append(
            (
                max(turn_epoch, clarification_epoch),
                canonical_json_bytes(canonical_state_key_payload(key)),
                key,
            )
        )
    return tuple(sorted(rows, key=lambda item: (item[0], item[1])))
