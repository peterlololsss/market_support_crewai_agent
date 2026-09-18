from collections.abc import Callable
from dataclasses import replace
from secrets import token_bytes

from market_support_crewai_agent.runtime.hashing import (
    feedback_hash,
    feedback_receipt_id,
)
from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.conversation_records import (
    StateRevisionRecordV1,
)
from market_support_crewai_agent.runtime.state.coordinator_config import (
    CoordinatorConfigV1,
)
from market_support_crewai_agent.runtime.state.coordinator_effects import (
    apply_effect_transitions,
    prepare_effect_transitions,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.coordinator_ledger import (
    derive_ledger_updates,
)
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    CompleteIssuedResponseRecordV1,
    FeedbackCommitResultV1,
    FeedbackPreparationV1,
    FeedbackReceiptReplayV1,
    FeedbackReceiptV1,
    PreparedFeedbackTokenRecordV1,
    PreparedFeedbackV1,
)
from market_support_crewai_agent.runtime.state.issued_response_store import (
    IssuedResponseStoreV1,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2


def resolve_feedback_record(
    store: IssuedResponseStoreV1,
    root: CoordinatorStateRootV1,
    state_key: ConversationStateKey,
    feedback: ActionFeedbackRequestV2,
) -> CompleteIssuedResponseRecordV1:
    lookup = store.lookup(root, state_key, feedback.response_id)
    if lookup.kind == "not_found":
        raise CoordinatorError("issued_response_not_found")
    if lookup.kind == "identity_mismatch":
        raise CoordinatorError("feedback_identity_mismatch")
    if lookup.record.request_id != feedback.request_id:
        raise CoordinatorError("feedback_effect_mismatch")
    return lookup.record


def feedback_commit_inputs(
    store: IssuedResponseStoreV1,
    root: CoordinatorStateRootV1,
    prepared: PreparedFeedbackV1,
    monotonic_clock_ns: Callable[[], int],
) -> tuple[CompleteIssuedResponseRecordV1, StateRevisionRecordV1, int]:
    token_key = (prepared.state_key, prepared.feedback_id)
    stored_token = root.prepared_feedback_tokens.get(token_key)
    if (
        stored_token is None
        or stored_token.preparation.commit_token != prepared.commit_token
    ):
        raise CoordinatorError("feedback_token_not_found")
    now_monotonic_ns = monotonic_clock_ns()
    if now_monotonic_ns >= prepared.expires_at_monotonic_ns:
        raise CoordinatorError("feedback_token_expired")
    lookup = store.lookup(root, prepared.state_key, prepared.response_id)
    revision = root.state_revisions.get(prepared.state_key)
    if lookup.kind != "found" or revision is None:
        raise CoordinatorError("conversation_state_changed")
    record = lookup.record
    if (
        revision.revision_epoch != prepared.revision_epoch
        or revision.revision != prepared.expected_state_revision
        or record.record_revision != prepared.expected_issued_record_revision
    ):
        raise CoordinatorError("conversation_state_changed")
    return record, revision, now_monotonic_ns


def requires_prepared_token_drop(error: CoordinatorError) -> bool:
    return error.code in {"feedback_token_expired", "conversation_state_changed"}


def prepare_feedback_candidate(
    root: CoordinatorStateRootV1,
    *,
    state_key: ConversationStateKey,
    feedback: ActionFeedbackRequestV2,
    record: CompleteIssuedResponseRecordV1,
    config: CoordinatorConfigV1,
    monotonic_clock_ns: Callable[[], int],
) -> tuple[CoordinatorStateRootV1 | None, FeedbackPreparationV1]:
    digest = feedback_hash(feedback)
    token_key = (state_key, feedback.feedback_id)
    receipt = root.feedback_receipts.get(token_key)
    if receipt is not None:
        if receipt.feedback_hash != digest:
            raise CoordinatorError("feedback_id_conflict")
        return None, FeedbackReceiptReplayV1(
            state_key=state_key,
            feedback_id=feedback.feedback_id,
            feedback_hash=digest,
            request_id=feedback.request_id,
            response_id=feedback.response_id,
            receipt_id=receipt.receipt_id,
            issued_record_revision=receipt.issued_record_revision,
        )
    existing_token = root.prepared_feedback_tokens.get(token_key)
    if existing_token is not None:
        if existing_token.preparation.feedback_hash != digest:
            raise CoordinatorError("feedback_id_conflict")
        return None, existing_token.preparation
    if len(root.prepared_feedback_tokens) >= config.issued_response_capacity:
        raise CoordinatorError("feedback_store_unavailable")
    revision = root.state_revisions.get(state_key)
    if revision is None:
        raise CoordinatorError("conversation_state_changed")
    prepared = PreparedFeedbackV1(
        state_key=state_key,
        feedback_id=feedback.feedback_id,
        feedback_hash=digest,
        request_id=feedback.request_id,
        response_id=feedback.response_id,
        commit_token=token_bytes(32),
        revision_epoch=revision.revision_epoch,
        expected_state_revision=revision.revision,
        expected_issued_record_revision=record.record_revision,
        transitions=prepare_effect_transitions(record, feedback),
        expires_at_monotonic_ns=monotonic_clock_ns() + 30_000_000_000,
    )
    candidate = replace(
        root,
        root_revision=root.root_revision + 1,
        prepared_feedback_tokens=root.prepared_feedback_tokens.with_updates(
            {token_key: PreparedFeedbackTokenRecordV1(prepared, monotonic_clock_ns())}
        ),
    )
    return candidate, prepared


def commit_feedback_candidate(
    root: CoordinatorStateRootV1,
    *,
    prepared: PreparedFeedbackV1,
    store: IssuedResponseStoreV1,
    monotonic_clock_ns: Callable[[], int],
    epoch_clock_ms: Callable[[], int],
    config: CoordinatorConfigV1,
) -> tuple[CoordinatorStateRootV1, FeedbackCommitResultV1]:
    record, revision, now_monotonic_ns = feedback_commit_inputs(
        store, root, prepared, monotonic_clock_ns
    )
    now_epoch_ms = epoch_clock_ms()
    effects = apply_effect_transitions(record.effects, prepared.transitions)
    if (
        len(record.receipts) >= 64
        or len(root.feedback_receipts) >= config.feedback_receipt_capacity
    ):
        raise CoordinatorError("feedback_store_unavailable")
    stored = sum(1 for transition in prepared.transitions if not transition.is_noop)
    next_revision = StateRevisionRecordV1(
        revision_epoch=revision.revision_epoch,
        revision=revision.revision + 1,
        created_at_monotonic_ns=revision.created_at_monotonic_ns,
        updated_at_monotonic_ns=now_monotonic_ns,
        expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
    )
    record_revision = record.record_revision + 1
    receipt_id = feedback_receipt_id(
        prepared.state_key,
        prepared.feedback_id,
        prepared.feedback_hash,
        prepared.response_id,
        record_revision,
    )
    receipt = FeedbackReceiptV1(
        receipt_id=receipt_id,
        state_key=prepared.state_key,
        feedback_id=prepared.feedback_id,
        feedback_hash=prepared.feedback_hash,
        response_id=prepared.response_id,
        issued_record_revision=record_revision,
        stored=stored,
        created_at_epoch_ms=now_epoch_ms,
        created_at_monotonic_ns=now_monotonic_ns,
        expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
    )
    complete = replace(
        record,
        effects=effects,
        receipts=(*record.receipts, receipt),
        record_revision=record_revision,
    )
    issued_key = (prepared.state_key, prepared.request_id)
    ledger_updates = derive_ledger_updates(
        root=root,
        record=complete,
        transitions=prepared.transitions,
        received_at_epoch_ms=now_epoch_ms,
        expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
    )
    token_key = (prepared.state_key, prepared.feedback_id)
    candidate = replace(
        root,
        root_revision=root.root_revision + 1,
        issued_records=root.issued_records.with_updates({issued_key: complete}),
        feedback_receipts=root.feedback_receipts.with_updates({token_key: receipt}),
        prepared_feedback_tokens=root.prepared_feedback_tokens.without_keys(
            frozenset({token_key})
        ),
        state_revisions=root.state_revisions.with_updates(
            {prepared.state_key: next_revision}
        ),
        action_ledger_records=root.action_ledger_records.with_updates(ledger_updates),
    )
    return candidate, FeedbackCommitResultV1(
        receipt_id=receipt_id, stored=stored, replayed=False
    )


def drop_prepared_token_candidate(
    root: CoordinatorStateRootV1, prepared: PreparedFeedbackV1
) -> CoordinatorStateRootV1:
    token_key = (prepared.state_key, prepared.feedback_id)
    return replace(
        root,
        root_revision=root.root_revision + 1,
        prepared_feedback_tokens=root.prepared_feedback_tokens.without_keys(
            frozenset({token_key})
        ),
    )
