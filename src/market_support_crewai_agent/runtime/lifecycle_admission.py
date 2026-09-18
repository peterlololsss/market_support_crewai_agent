from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime import lifecycle_history
from market_support_crewai_agent.runtime.context.action_history import (
    recent_action_summaries_from_snapshot,
)
from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    KernelReplyRequestV1,
)
from market_support_crewai_agent.runtime.observability.runtime_trace import (
    trace_event,
    trace_span,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    RecallModeV1,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
    state_admission_hash_v1,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.coordinator_protocol import (
    ReplyTurnStateCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    NewReplyReservationV1,
    TurnAdmissionSnapshotV1,
)


@dataclass(frozen=True, slots=True)
class TurnAdmissionV1:
    snapshot: TurnAdmissionSnapshotV1
    policy: PolicyManifestV2
    history: list[ConversationMessage]
    action_history: tuple[RecentExecutedActionSummaryViewV1, ...]


def admit_turn(
    coordinator: ReplyTurnStateCoordinatorV1,
    *,
    request: KernelReplyRequestV1,
    state_key: ConversationStateKey,
    reservation: NewReplyReservationV1,
    scope_authority: BusinessScopeAuthorityV1,
    group_recall_mode: RecallModeV1,
) -> TurnAdmissionV1:
    core = compile_policy_authority_core_v1(
        request,
        scope_authority,
        group_recall_mode=group_recall_mode,
    )
    with trace_span("state.admission_snapshot"):
        snapshot = coordinator.read_turn_admission_snapshot(
            state_key,
            state_admission_hash_v1(core),
            core.effective_grants_hash,
            scope_authority.business_scope_hash,
        )
    if (
        snapshot.revision_epoch != reservation.revision_epoch
        or snapshot.state_revision != reservation.state_revision
    ):
        raise CoordinatorError("conversation_state_changed")
    ledger_summary_v2 = policy_ledger_summary_v1(
        recent_artifact_types=(),
        recent_executed_count=sum(
            row.status == "executed" for row in snapshot.ledger_rows
        ),
    )
    policy_v2 = PolicyManifestV2.from_core(core, ledger_summary_v2)
    history = lifecycle_history.history_from_snapshot(snapshot.turns)
    action_history: tuple[RecentExecutedActionSummaryViewV1, ...] = (
        recent_action_summaries_from_snapshot(
            snapshot.ledger_rows,
            snapshot_at_epoch_ms=snapshot.snapshot_at_epoch_ms,
        )
    )
    trace_event(
        "state.history_loaded",
        history_count=len(history),
        action_history_count=len(action_history),
    )
    return TurnAdmissionV1(
        snapshot=snapshot,
        policy=policy_v2,
        history=history,
        action_history=action_history,
    )
