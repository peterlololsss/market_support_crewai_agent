from collections.abc import Callable
from dataclasses import replace
from secrets import token_bytes

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.conversation_records import (
    ConversationTurnRecordV2,
    PendingClarificationRecordV1,
    StateRevisionRecordV1,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.coordinator_ledger import (
    admitted_action_ledger_rows,
)
from market_support_crewai_agent.runtime.state.coordinator_retention import (
    enforce_session_capacity,
)
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
    state_session_eviction_order,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    PendingIssuedResponseRecordV1,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    CompletedReplyReplayV1,
    NewReplyReservationV1,
    ReservationResultV1,
    TurnAdmissionSnapshotV1,
)

from .coordinator_config import CoordinatorConfigV1


def read_turn_admission_snapshot(
    root: CoordinatorStateRootV1,
    *,
    state_key: ConversationStateKey,
    par1: str,
    grh1: str,
    bsh1: str,
    monotonic_clock_ns: Callable[[], int],
    epoch_clock_ms: Callable[[], int],
    config: CoordinatorConfigV1,
) -> TurnAdmissionSnapshotV1:
    revision = root.state_revisions.get(state_key)
    now_monotonic_ns = monotonic_clock_ns()
    if revision is None:
        revision = StateRevisionRecordV1(
            revision_epoch=token_bytes(16),
            revision=0,
            created_at_monotonic_ns=now_monotonic_ns,
            updated_at_monotonic_ns=now_monotonic_ns,
            expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
        )
    turns = tuple(
        turn
        for turn in root.conversation_turns.get(state_key, ())
        if turn.par1 == par1 and turn.grh1 == grh1 and turn.bsh1 == bsh1
    )
    clarification = root.clarification_records.get(state_key)
    if clarification is not None and (
        clarification.par1 != par1
        or clarification.grh1 != grh1
        or clarification.bsh1 != bsh1
    ):
        clarification = None
    return TurnAdmissionSnapshotV1(
        state_key=state_key,
        par1=par1,
        grh1=grh1,
        bsh1=bsh1,
        ledger_rows=admitted_action_ledger_rows(
            root, state_key=state_key, par1=par1, grh1=grh1, bsh1=bsh1
        ),
        turns=turns,
        pending_clarification=clarification,
        revision_epoch=revision.revision_epoch,
        state_revision=revision.revision,
        snapshot_at_epoch_ms=epoch_clock_ms(),
        snapshot_at_monotonic_ns=now_monotonic_ns,
        pending_reply=state_key in root.pending_request_by_state,
    )


def replace_session_state_candidate(
    root: CoordinatorStateRootV1,
    *,
    state_key: ConversationStateKey,
    turns: tuple[ConversationTurnRecordV2, ...],
    clarification: PendingClarificationRecordV1 | None,
    monotonic_clock_ns: Callable[[], int],
    config: CoordinatorConfigV1,
) -> CoordinatorStateRootV1:
    turn_updates = root.conversation_turns
    clarification_updates = root.clarification_records
    if turns:
        turn_updates = turn_updates.with_updates({state_key: turns})
    else:
        turn_updates = turn_updates.without_keys(frozenset({state_key}))
    if clarification is None:
        clarification_updates = clarification_updates.without_keys(
            frozenset({state_key})
        )
    else:
        clarification_updates = clarification_updates.with_updates(
            {state_key: clarification}
        )
    session_order = state_session_eviction_order(turn_updates, clarification_updates)
    while len(session_order) > config.conversation_max_sessions:
        _, _, victim = session_order[0]
        if victim in root.pending_request_by_state:
            raise CoordinatorError("state_store_unavailable")
        turn_updates = turn_updates.without_keys(frozenset({victim}))
        clarification_updates = clarification_updates.without_keys(frozenset({victim}))
        session_order = state_session_eviction_order(
            turn_updates, clarification_updates
        )
    prior = root.state_revisions.get(state_key)
    now_monotonic_ns = monotonic_clock_ns()
    revision = StateRevisionRecordV1(
        revision_epoch=prior.revision_epoch if prior is not None else token_bytes(16),
        revision=(prior.revision + 1) if prior is not None else 1,
        created_at_monotonic_ns=(
            prior.created_at_monotonic_ns if prior is not None else now_monotonic_ns
        ),
        updated_at_monotonic_ns=now_monotonic_ns,
        expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
    )
    candidate = replace(
        root,
        root_revision=root.root_revision + 1,
        conversation_turns=turn_updates,
        clarification_records=clarification_updates,
        state_session_eviction_order=tuple(
            (epoch, key) for epoch, key, _ in session_order
        ),
        state_revisions=root.state_revisions.with_updates({state_key: revision}),
    )
    return enforce_session_capacity(candidate, config.conversation_max_sessions)


def reserve_reply_candidate(
    root: CoordinatorStateRootV1,
    *,
    state_key: ConversationStateKey,
    request_id: str,
    request_hash: str,
    replay_eligible: bool,
    monotonic_clock_ns: Callable[[], int],
    epoch_clock_ms: Callable[[], int],
    config: CoordinatorConfigV1,
) -> tuple[CoordinatorStateRootV1 | None, ReservationResultV1]:
    existing_result = existing_reservation_result(
        root,
        state_key=state_key,
        request_id=request_id,
        request_hash=request_hash,
        replay_eligible=replay_eligible,
    )
    if existing_result is not None:
        return None, existing_result
    now_monotonic_ns = monotonic_clock_ns()
    state_revision = root.state_revisions.get(state_key)
    if state_revision is None:
        state_revision = StateRevisionRecordV1(
            revision_epoch=token_bytes(16),
            revision=0,
            created_at_monotonic_ns=now_monotonic_ns,
            updated_at_monotonic_ns=now_monotonic_ns,
            expires_at_monotonic_ns=now_monotonic_ns + config.issued_ttl_ns,
        )
    owner_token = token_bytes(32)
    existing_key = (state_key, request_id)
    record = PendingIssuedResponseRecordV1(
        state_key=state_key,
        request_id=request_id,
        request_hash=request_hash,
        replay_eligible=replay_eligible,
        owner_token=owner_token,
        revision_epoch=state_revision.revision_epoch,
        reserved_state_revision=state_revision.revision,
        created_at_epoch_ms=epoch_clock_ms(),
        created_at_monotonic_ns=now_monotonic_ns,
        expires_at_monotonic_ns=now_monotonic_ns + config.issued_pending_ttl_ns,
    )
    candidate = replace(
        root,
        root_revision=root.root_revision + 1,
        issued_records=root.issued_records.with_updates({existing_key: record}),
        pending_request_by_state=root.pending_request_by_state.with_updates(
            {state_key: existing_key}
        ),
        state_revisions=root.state_revisions.with_updates({state_key: state_revision}),
    )
    return candidate, NewReplyReservationV1(
        state_key=state_key,
        request_id=request_id,
        request_hash=request_hash,
        replay_eligible=replay_eligible,
        owner_token=owner_token,
        revision_epoch=state_revision.revision_epoch,
        state_revision=state_revision.revision,
    )


def existing_reservation_result(
    root: CoordinatorStateRootV1,
    *,
    state_key: ConversationStateKey,
    request_id: str,
    request_hash: str,
    replay_eligible: bool,
) -> ReservationResultV1 | None:
    existing_key = (state_key, request_id)
    existing = root.issued_records.get(existing_key)
    if existing is not None:
        if (
            existing.request_hash != request_hash
            or existing.replay_eligible != replay_eligible
        ):
            raise CoordinatorError("request_id_conflict")
        if existing.phase == "pending":
            raise CoordinatorError("request_in_progress")
        if not replay_eligible:
            raise CoordinatorError("request_id_conflict")
        return CompletedReplyReplayV1(
            state_key=state_key,
            request_id=request_id,
            request_hash=request_hash,
            response=existing.response,
            issued_at_epoch_ms=existing.issued_at_epoch_ms,
            issued_record_revision=existing.record_revision,
        )
    if state_key in root.pending_request_by_state:
        raise CoordinatorError("conversation_turn_in_progress")
    return None


def abort_reply_candidate(
    root: CoordinatorStateRootV1,
    state_key: ConversationStateKey,
    request_id: str,
    owner_token: bytes,
) -> CoordinatorStateRootV1:
    key = (state_key, request_id)
    pending = root.issued_records.get(key)
    if pending is None:
        raise CoordinatorError("reservation_not_found")
    if pending.phase != "pending" or pending.owner_token != owner_token:
        raise CoordinatorError("reservation_not_found")
    return replace(
        root,
        root_revision=root.root_revision + 1,
        issued_records=root.issued_records.without_keys(frozenset({key})),
        pending_request_by_state=root.pending_request_by_state.without_keys(
            frozenset({state_key})
        ),
    )
