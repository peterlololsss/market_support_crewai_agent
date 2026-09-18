from dataclasses import replace

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.conversation_records import (
    ConversationTurnRecordV2,
)
from market_support_crewai_agent.runtime.state.coordinator_config import (
    CoordinatorConfigV1,
)
from market_support_crewai_agent.runtime.state.coordinator_retention import (
    without_unpinned_revisions,
)
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    IssuedResponseRecordV1,
)
from market_support_crewai_agent.runtime.state.persistent_map import FrozenMapV1


def cleanup_candidate(
    root: CoordinatorStateRootV1,
    *,
    now_monotonic_ns: int,
    now_epoch_ms: int,
    config: CoordinatorConfigV1,
) -> tuple[CoordinatorStateRootV1 | None, int]:
    retained_turns: dict[
        ConversationStateKey, tuple[ConversationTurnRecordV2, ...]
    ] = {}
    for state_key, turns in root.conversation_turns.items():
        retained = tuple(
            turn
            for turn in turns
            if turn.created_at_epoch_ms + config.conversation_ttl_ms > now_epoch_ms
        )[-config.conversation_max_messages :]
        if retained:
            retained_turns[state_key] = retained
    expired_turn_states = frozenset(root.conversation_turns).difference(retained_turns)
    changed_turn_states = frozenset(
        state_key
        for state_key, turns in retained_turns.items()
        if turns != root.conversation_turns.get(state_key)
    )
    expired_clarifications = frozenset(
        state_key
        for state_key, clarification in root.clarification_records.items()
        if clarification.created_at_epoch_ms + config.conversation_ttl_ms
        <= now_epoch_ms
    )
    expired_audits = frozenset(
        key
        for key, record in root.audit_records.items()
        if record.expires_at_monotonic_ns <= now_monotonic_ns
    )
    expired_tokens = frozenset(
        key
        for key, value in root.prepared_feedback_tokens.items()
        if value.preparation.expires_at_monotonic_ns <= now_monotonic_ns
    )
    expired_records = frozenset(
        key
        for key, value in root.issued_records.items()
        if value.expires_at_monotonic_ns <= now_monotonic_ns
    )
    expired_feedback_receipts = frozenset(
        key
        for key, value in root.feedback_receipts.items()
        if value.expires_at_monotonic_ns <= now_monotonic_ns
    )
    retained_issued_records: dict[
        tuple[ConversationStateKey, str], IssuedResponseRecordV1
    ] = {}
    changed_issued_receipts = False
    for key, value in root.issued_records.items():
        if key in expired_records:
            continue
        if value.phase == "complete":
            receipts = tuple(
                receipt
                for receipt in value.receipts
                if receipt.expires_at_monotonic_ns > now_monotonic_ns
            )
            if receipts != value.receipts:
                retained_issued_records[key] = replace(value, receipts=receipts)
                changed_issued_receipts = True
                continue
        retained_issued_records[key] = value
    expired_ledger_rows = frozenset(
        key
        for key, value in root.action_ledger_records.items()
        if value.expires_at_monotonic_ns <= now_monotonic_ns
    )
    if not (
        expired_tokens
        or expired_records
        or expired_feedback_receipts
        or expired_ledger_rows
        or expired_turn_states
        or changed_turn_states
        or expired_clarifications
        or expired_audits
        or changed_issued_receipts
    ):
        return None, 0
    pending_states = frozenset(
        state_key
        for state_key, key in root.pending_request_by_state.items()
        if key in expired_records
    )
    response_ids = frozenset(
        value.response.response_id
        for key, value in root.issued_records.items()
        if key in expired_records and value.phase == "complete"
    )
    action_ids = frozenset(
        effect.action_id
        for key, value in root.issued_records.items()
        if key in expired_records and value.phase == "complete"
        for effect in value.effects
        if effect.action_id is not None
    )
    candidate = replace(
        root,
        root_revision=root.root_revision + 1,
        issued_records=FrozenMapV1(retained_issued_records),
        pending_request_by_state=root.pending_request_by_state.without_keys(
            pending_states
        ),
        prepared_feedback_tokens=root.prepared_feedback_tokens.without_keys(
            expired_tokens
        ),
        response_index=root.response_index.without_keys(response_ids),
        action_index=root.action_index.without_keys(action_ids),
        feedback_receipts=root.feedback_receipts.without_keys(
            expired_feedback_receipts
        ),
        action_ledger_records=root.action_ledger_records.without_keys(
            expired_ledger_rows
        ),
        conversation_turns=FrozenMapV1(retained_turns),
        clarification_records=root.clarification_records.without_keys(
            expired_clarifications
        ),
        audit_records=root.audit_records.without_keys(expired_audits),
    )
    removed = (
        len(expired_tokens)
        + len(expired_records)
        + len(expired_feedback_receipts)
        + len(expired_ledger_rows)
        + len(expired_turn_states)
        + len(changed_turn_states)
        + len(expired_clarifications)
        + len(expired_audits)
    )
    return without_unpinned_revisions(candidate), removed
