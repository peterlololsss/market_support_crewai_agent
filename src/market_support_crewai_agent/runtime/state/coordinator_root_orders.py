from dataclasses import replace

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.identity.state_key import (
    canonical_json_bytes,
    canonical_state_key_payload,
)
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
    state_session_eviction_order,
)


def key_bytes(state_key: ConversationStateKey, suffix: str = "") -> bytes:
    return canonical_json_bytes(canonical_state_key_payload(state_key)) + suffix.encode(
        "utf-8"
    )


def with_root_orders(root: CoordinatorStateRootV1) -> CoordinatorStateRootV1:
    issued_expiry = tuple(
        sorted(
            (record.expires_at_monotonic_ns, key_bytes(key[0], key[1]), key)
            for key, record in root.issued_records.items()
        )
    )
    issued_complete = tuple(
        sorted(
            (record.completed_at_monotonic_ns, key_bytes(key[0], key[1]), key)
            for key, record in root.issued_records.items()
            if record.phase == "complete"
        )
    )
    conversation_expiry = tuple(
        sorted(
            (
                turn.created_at_epoch_ms,
                key_bytes(state_key, str(turn.ordinal)),
                (state_key, str(turn.ordinal)),
            )
            for state_key, turns in root.conversation_turns.items()
            for turn in turns
        )
    )
    clarification_expiry = tuple(
        sorted(
            (
                record.created_at_epoch_ms,
                key_bytes(state_key, record.clarification_ref),
                state_key,
            )
            for state_key, record in root.clarification_records.items()
        )
    )
    feedback_expiry = tuple(
        sorted(
            (record.expires_at_monotonic_ns, key_bytes(key[0], key[1]), key)
            for key, record in root.feedback_receipts.items()
        )
    )
    prepared_expiry = tuple(
        sorted(
            (
                record.preparation.expires_at_monotonic_ns,
                key_bytes(key[0], key[1]),
                key,
            )
            for key, record in root.prepared_feedback_tokens.items()
        )
    )
    revision_expiry = tuple(
        sorted(
            (record.expires_at_monotonic_ns, key_bytes(key), key)
            for key, record in root.state_revisions.items()
        )
    )
    ledger_expiry = tuple(
        sorted(
            (record.expires_at_monotonic_ns, key_bytes(key[0], key[2]), key)
            for key, record in root.action_ledger_records.items()
        )
    )
    ledger_eviction = tuple(
        sorted(
            (record.received_at_epoch_ms, key_bytes(key[0], key[2]), key)
            for key, record in root.action_ledger_records.items()
        )
    )
    audit_expiry = tuple(
        sorted(
            (record.expires_at_monotonic_ns, key_bytes(key[0], key[1]), key)
            for key, record in root.audit_records.items()
        )
    )
    audit_eviction = tuple(
        sorted(
            (record.created_at_epoch_ms, key_bytes(key[0], key[1]), key)
            for key, record in root.audit_records.items()
        )
    )
    session_order = tuple(
        (epoch, key, state_key)
        for epoch, key, state_key in state_session_eviction_order(
            root.conversation_turns, root.clarification_records
        )
    )
    return replace(
        root,
        issued_expiry_order=issued_expiry,
        issued_complete_eviction_order=issued_complete,
        conversation_expiry_order=conversation_expiry,
        clarification_expiry_order=clarification_expiry,
        feedback_receipt_expiry_order=feedback_expiry,
        prepared_token_expiry_order=prepared_expiry,
        state_revision_expiry_order=revision_expiry,
        state_revision_eviction_order=revision_expiry,
        action_ledger_expiry_order=ledger_expiry,
        action_ledger_eviction_order=ledger_eviction,
        audit_expiry_order=audit_expiry,
        audit_eviction_order=audit_eviction,
        state_session_eviction_order=session_order,
    )
