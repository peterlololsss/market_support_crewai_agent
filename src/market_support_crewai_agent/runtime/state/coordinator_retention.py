from dataclasses import replace
from typing import Final

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.coordinator_root_orders import (
    key_bytes,
    with_root_orders,
)
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
    state_session_eviction_order,
)

_ACTION_LEDGER_CAPACITY: Final = 5_000


def enforce_session_capacity(
    root: CoordinatorStateRootV1, capacity: int
) -> CoordinatorStateRootV1:
    turns = root.conversation_turns
    clarifications = root.clarification_records
    while len(state_session_eviction_order(turns, clarifications)) > capacity:
        _, _, victim = state_session_eviction_order(turns, clarifications)[0]
        if victim in root.pending_request_by_state:
            raise CoordinatorError("state_store_unavailable")
        turns = turns.without_keys(frozenset({victim}))
        clarifications = clarifications.without_keys(frozenset({victim}))
    return without_unpinned_revisions(
        replace(root, conversation_turns=turns, clarification_records=clarifications)
    )


def issued_capacity_eviction_candidate(
    root: CoordinatorStateRootV1, capacity: int
) -> CoordinatorStateRootV1 | None:
    if len(root.issued_records) < capacity:
        return None
    complete = [
        (record.completed_at_monotonic_ns, key, record)
        for key, record in root.issued_records.items()
        if record.phase == "complete"
    ]
    if not complete:
        raise CoordinatorError("issued_response_store_unavailable")
    _, key, record = min(complete, key=lambda item: (item[0], item[1][1]))
    action_ids = frozenset(
        effect.action_id for effect in record.effects if effect.action_id is not None
    )
    return replace(
        root,
        root_revision=root.root_revision + 1,
        issued_records=root.issued_records.without_keys(frozenset({key})),
        response_index=root.response_index.without_keys(
            frozenset({record.response.response_id})
        ),
        action_index=root.action_index.without_keys(action_ids),
    )


def prepare_root_for_publication(
    root: CoordinatorStateRootV1, state_revision_capacity: int
) -> CoordinatorStateRootV1:
    bounded = _enforce_inactive_record_capacities(root)
    bounded = _enforce_state_revision_capacity(bounded, state_revision_capacity)
    ordered = with_root_orders(bounded)
    _validate_root(ordered)
    return ordered


def without_unpinned_revisions(
    root: CoordinatorStateRootV1,
) -> CoordinatorStateRootV1:
    pinned = _pinned_state_keys(root)
    removable = frozenset(key for key in root.state_revisions if key not in pinned)
    return replace(root, state_revisions=root.state_revisions.without_keys(removable))


def _enforce_inactive_record_capacities(
    root: CoordinatorStateRootV1,
) -> CoordinatorStateRootV1:
    ledger = root.action_ledger_records
    while len(ledger) > _ACTION_LEDGER_CAPACITY:
        victim = min(
            ledger,
            key=lambda key: (
                ledger[key].received_at_epoch_ms,
                key_bytes(key[0], key[2]),
            ),
        )
        ledger = ledger.without_keys(frozenset({victim}))
    return replace(root, action_ledger_records=ledger)


def _validate_root(root: CoordinatorStateRootV1) -> None:
    for state_key, issued_key in root.pending_request_by_state.items():
        record = root.issued_records.get(issued_key)
        if record is None or record.phase != "pending" or record.state_key != state_key:
            raise CoordinatorError("state_store_unavailable")
    for response_id, issued_key in root.response_index.items():
        record = root.issued_records.get(issued_key)
        if (
            record is None
            or record.phase != "complete"
            or record.response.response_id != response_id
        ):
            raise CoordinatorError("state_store_unavailable")
    for action_id, indexed in root.action_index.items():
        state_key, response_id, effect_key = indexed
        issued_key = root.response_index.get(response_id)
        if issued_key is None or issued_key[0] != state_key:
            raise CoordinatorError("state_store_unavailable")
        record = root.issued_records.get(issued_key)
        if record is None or record.phase != "complete":
            raise CoordinatorError("state_store_unavailable")
        effect = next(
            (item for item in record.effects if item.effect_key == effect_key), None
        )
        if effect is None or effect.action_id != action_id:
            raise CoordinatorError("state_store_unavailable")
    expected_orders = with_root_orders(root)
    order_fields = (
        "issued_expiry_order",
        "issued_complete_eviction_order",
        "conversation_expiry_order",
        "state_session_eviction_order",
        "clarification_expiry_order",
        "audit_expiry_order",
        "audit_eviction_order",
        "action_ledger_expiry_order",
        "action_ledger_eviction_order",
        "feedback_receipt_expiry_order",
        "prepared_token_expiry_order",
        "state_revision_expiry_order",
        "state_revision_eviction_order",
    )
    for field_name in order_fields:
        if getattr(root, field_name) != getattr(expected_orders, field_name):
            raise CoordinatorError("state_store_unavailable")


def _enforce_state_revision_capacity(
    root: CoordinatorStateRootV1, capacity: int
) -> CoordinatorStateRootV1:
    pruned = without_unpinned_revisions(root)
    if len(pruned.state_revisions) <= capacity:
        return pruned
    pinned = _pinned_state_keys(pruned)
    removable = sorted(
        (
            record.updated_at_monotonic_ns,
            key_bytes(state_key),
            state_key,
        )
        for state_key, record in pruned.state_revisions.items()
        if state_key not in pinned
    )
    excess = len(pruned.state_revisions) - capacity
    if len(removable) < excess:
        raise CoordinatorError("state_store_unavailable")
    keys = frozenset(row[2] for row in removable[:excess])
    return replace(pruned, state_revisions=pruned.state_revisions.without_keys(keys))


def _pinned_state_keys(root: CoordinatorStateRootV1) -> frozenset[ConversationStateKey]:
    pinned = set(root.conversation_turns)
    pinned.update(root.clarification_records)
    pinned.update(key[0] for key in root.issued_records)
    pinned.update(key[0] for key in root.feedback_receipts)
    pinned.update(key[0] for key in root.prepared_feedback_tokens)
    pinned.update(key[0] for key in root.action_ledger_records)
    pinned.update(key[0] for key in root.audit_records)
    return frozenset(pinned)
