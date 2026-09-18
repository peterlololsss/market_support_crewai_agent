from collections.abc import Callable
from dataclasses import replace
from threading import RLock
from time import monotonic_ns, time
from typing import final

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.schemas import feedback as feedback_schemas
from market_support_crewai_agent.schemas.reply import ReplyResponse

from . import (
    audit_types,
    coordinator_audit,
    coordinator_cleanup,
    coordinator_config,
    coordinator_errors,
    coordinator_feedback,
    coordinator_reply_commit,
    coordinator_reservations,
    coordinator_retention,
    coordinator_state,
    effect_records,
    issued_response_store,
    transaction_records,
)
from .coordinator_retention import prepare_root_for_publication
from .coordinator_state import EMPTY_COORDINATOR_ROOT, CoordinatorStateRootV1


@final
class ReplyStateTransactionCoordinatorV1:
    def __init__(
        self,
        root: CoordinatorStateRootV1 = EMPTY_COORDINATOR_ROOT,
        issued_response_capacity: int = 5000,
        conversation_max_sessions: int = 5000,
        feedback_receipt_capacity: int = 20_000,
        conversation_ttl_seconds: int = 86_400,
        conversation_max_messages: int = 12,
        issued_response_ttl_seconds: int = 86_400,
        issued_response_pending_ttl_seconds: int = 180,
        monotonic_clock_ns: Callable[[], int] = monotonic_ns,
        wall_clock_epoch_ms: Callable[[], int] | None = None,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self._config = coordinator_config.build_coordinator_config(
            issued_response_capacity=issued_response_capacity,
            conversation_max_sessions=conversation_max_sessions,
            feedback_receipt_capacity=feedback_receipt_capacity,
            conversation_ttl_seconds=conversation_ttl_seconds,
            conversation_max_messages=conversation_max_messages,
            issued_response_ttl_seconds=issued_response_ttl_seconds,
            issued_response_pending_ttl_seconds=issued_response_pending_ttl_seconds,
        )
        self._root = root
        self._lock = RLock()
        self._issued_store = issued_response_store.IssuedResponseStoreV1()
        self._monotonic_clock_ns = monotonic_clock_ns
        self._wall_clock_epoch_ms = wall_clock_epoch_ms or (lambda: int(time() * 1_000))
        self._fault_hook = fault_hook
        self._last_journal: coordinator_state.ReplyStateJournalV2 | None = None

    def lookup_issued_response(
        self, state_key: ConversationStateKey, response_id: str
    ) -> effect_records.IssuedResponseLookupV1:
        with self._lock:
            return self._issued_store.lookup(self._root, state_key, response_id)

    def root_revision(self) -> int:
        with self._lock:
            return self._root.root_revision

    def last_journal(self) -> coordinator_state.ReplyStateJournalV2 | None:
        with self._lock:
            return self._last_journal

    def read_turn_admission_snapshot(
        self, state_key: ConversationStateKey, par1: str, grh1: str, bsh1: str
    ) -> transaction_records.TurnAdmissionSnapshotV1:
        with self._lock:
            return coordinator_reservations.read_turn_admission_snapshot(
                self._root,
                state_key=state_key,
                par1=par1,
                grh1=grh1,
                bsh1=bsh1,
                monotonic_clock_ns=self._monotonic_clock_ns,
                epoch_clock_ms=self._wall_clock_epoch_ms,
                config=self._config,
            )

    def reserve_reply(
        self,
        state_key: ConversationStateKey,
        request_id: str,
        request_hash: str,
        replay_eligible: bool,
    ) -> transaction_records.ReservationResultV1:
        with self._lock:
            _ = self._cleanup_locked(self._monotonic_clock_ns())
            replay = coordinator_reservations.existing_reservation_result(
                self._root,
                state_key=state_key,
                request_id=request_id,
                request_hash=request_hash,
                replay_eligible=replay_eligible,
            )
            if replay is not None:
                return replay
            eviction = coordinator_retention.issued_capacity_eviction_candidate(
                self._root, self._config.issued_response_capacity
            )
            if eviction is not None:
                self._publish("cleanup", None, (), eviction)
            candidate, reservation = coordinator_reservations.reserve_reply_candidate(
                self._root,
                state_key=state_key,
                request_id=request_id,
                request_hash=request_hash,
                replay_eligible=replay_eligible,
                monotonic_clock_ns=self._monotonic_clock_ns,
                epoch_clock_ms=self._wall_clock_epoch_ms,
                config=self._config,
            )
            if candidate is None:
                return reservation
            self._publish("reply_reserve", state_key, (state_key,), candidate)
            return reservation

    def abort_reply(
        self, state_key: ConversationStateKey, request_id: str, owner_token: bytes
    ) -> None:
        with self._lock:
            candidate = coordinator_reservations.abort_reply_candidate(
                self._root, state_key, request_id, owner_token
            )
            self._publish("reply_abort", state_key, (state_key,), candidate)

    def commit_reply(
        self, proposal: transaction_records.ReplyCommitProposalV1
    ) -> ReplyResponse:
        with self._lock:
            candidate, response = coordinator_reply_commit.commit_reply_candidate(
                self._root,
                proposal,
                monotonic_clock_ns=self._monotonic_clock_ns,
                epoch_clock_ms=self._wall_clock_epoch_ms,
                config=self._config,
            )
            self._publish(
                "reply_commit", proposal.state_key, (proposal.state_key,), candidate
            )
            return response

    def prepare_feedback(
        self,
        state_key: ConversationStateKey,
        feedback: feedback_schemas.ActionFeedbackRequestV2,
    ) -> effect_records.FeedbackPreparationV1:
        with self._lock:
            _ = self._cleanup_locked(self._monotonic_clock_ns())
            record = coordinator_feedback.resolve_feedback_record(
                self._issued_store, self._root, state_key, feedback
            )
            candidate, prepared = coordinator_feedback.prepare_feedback_candidate(
                self._root,
                state_key=state_key,
                feedback=feedback,
                record=record,
                config=self._config,
                monotonic_clock_ns=self._monotonic_clock_ns,
            )
            if candidate is not None:
                self._publish("feedback_prepare", state_key, (state_key,), candidate)
            return prepared

    def commit_feedback(
        self, prepared: effect_records.PreparedFeedbackV1
    ) -> effect_records.FeedbackCommitResultV1:
        with self._lock:
            try:
                candidate, result = coordinator_feedback.commit_feedback_candidate(
                    self._root,
                    prepared=prepared,
                    store=self._issued_store,
                    monotonic_clock_ns=self._monotonic_clock_ns,
                    epoch_clock_ms=self._wall_clock_epoch_ms,
                    config=self._config,
                )
            except coordinator_errors.CoordinatorError as error:
                if coordinator_feedback.requires_prepared_token_drop(error):
                    candidate = coordinator_feedback.drop_prepared_token_candidate(
                        self._root, prepared
                    )
                    self._publish(
                        "feedback_commit",
                        prepared.state_key,
                        (prepared.state_key,),
                        candidate,
                    )
                raise
            self._publish(
                "feedback_commit", prepared.state_key, (prepared.state_key,), candidate
            )
            return result

    def _publish(
        self,
        operation: audit_types.ReplyStateJournalOperationV1,
        initiating_state_key: ConversationStateKey | None,
        affected_state_keys: tuple[ConversationStateKey, ...],
        candidate: CoordinatorStateRootV1,
    ) -> None:
        candidate = prepare_root_for_publication(
            candidate, self._config.state_revision_capacity
        )
        if candidate.root_revision != self._root.root_revision + 1:
            raise coordinator_errors.CoordinatorError("state_store_unavailable")
        journal = coordinator_audit.build_journal(
            operation=operation,
            initiating_state_key=initiating_state_key,
            affected_state_keys=affected_state_keys,
            prior_root=self._root,
            candidate_root=candidate,
        )
        try:
            if self._fault_hook is not None:
                self._fault_hook(operation)
            self._root = candidate
        except RuntimeError:
            self._last_journal = replace(journal, outcome="rolled_back")
            raise
        self._last_journal = replace(journal, outcome="committed")

    def cleanup(self) -> int:
        with self._lock:
            return self._cleanup_locked(self._monotonic_clock_ns())

    def _cleanup_locked(self, now_monotonic_ns: int) -> int:
        candidate, removed = coordinator_cleanup.cleanup_candidate(
            self._root,
            now_monotonic_ns=now_monotonic_ns,
            now_epoch_ms=self._wall_clock_epoch_ms(),
            config=self._config,
        )
        if candidate is not None:
            self._publish("cleanup", None, (), candidate)
        return removed
