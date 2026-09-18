from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, final

from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.coordinator_protocol import (
    ReplyTurnStateCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    ActionLedgerRecordV2,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    NewReplyReservationV1,
    ReplyCommitProposalV1,
    TurnAdmissionSnapshotV1,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from market_support_crewai_agent.settings_model import Settings
from tests.unit.context.action_history_snapshot_rows import action_history_row

if TYPE_CHECKING:
    from market_support_crewai_agent.runtime.evidence.scope_authority import (
        BusinessScopeAuthorityV1,
    )
    from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
    from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
        CrewAIAgentAdapterV1,
    )
    from market_support_crewai_agent.runtime.lifecycle_contracts import (
        AlignmentAgentFactoryV1,
        CandidateResponseBuilderV1,
        ComposerAgentFactoryV1,
    )
    from market_support_crewai_agent.runtime.planning_flow import PlannerAgentFactoryV1
    from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
    from market_support_crewai_agent.runtime.policy.ontology_models import (
        DomainContextV1,
    )
    from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
    from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
    from market_support_crewai_agent.runtime.prompts.invocation_journal import (
        TurnLlmInvocationJournalV1,
    )
    from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
    from market_support_crewai_agent.runtime.rendering.v2_composer import V2Composer
    from market_support_crewai_agent.runtime.state.conversation_store import (
        ConversationMessage,
    )
    from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
    from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
        ReplyAlignmentVerifier,
    )


@final
class SnapshotCoordinatorSpy:
    def __init__(
        self, state_key: ConversationStateKey, row: ActionLedgerRecordV2
    ) -> None:
        self.state_key: ConversationStateKey = state_key
        self.rows: tuple[ActionLedgerRecordV2, ...] = (row,)
        self.snapshot_reads: int = 0
        self.committed: bool = False

    def reserve_reply(
        self,
        state_key: ConversationStateKey,
        request_id: str,
        request_hash: str,
        replay_eligible: bool,
    ) -> NewReplyReservationV1:
        return NewReplyReservationV1(
            state_key=state_key,
            request_id=request_id,
            request_hash=request_hash,
            replay_eligible=replay_eligible,
            owner_token=b"owner",
            revision_epoch=b"epoch-012345678",
            state_revision=3,
        )

    def read_turn_admission_snapshot(
        self,
        state_key: ConversationStateKey,
        par1: str,
        grh1: str,
        bsh1: str,
    ) -> TurnAdmissionSnapshotV1:
        self.snapshot_reads += 1
        return TurnAdmissionSnapshotV1(
            state_key=state_key,
            par1=par1,
            grh1=grh1,
            bsh1=bsh1,
            ledger_rows=self.rows,
            turns=(),
            pending_clarification=None,
            revision_epoch=b"epoch-012345678",
            state_revision=3,
            snapshot_at_epoch_ms=2_001_000,
            snapshot_at_monotonic_ns=1,
            pending_reply=False,
        )

    def commit_reply(self, proposal: ReplyCommitProposalV1) -> ReplyResponse:
        self.committed = True
        return proposal.response

    def abort_reply(
        self,
        state_key: ConversationStateKey,
        request_id: str,
        owner_token: bytes,
    ) -> None:
        del state_key, request_id, owner_token


@dataclass(frozen=True, slots=True)
class _UnusedPlannerFactory:
    def build_planner_agent(self) -> CrewAIAgentAdapterV1:
        raise AssertionError("planner factory is not used by this lifecycle scenario")


@dataclass(frozen=True, slots=True)
class _UnusedComposerFactory:
    def build_composer_agent(
        self,
        stage: Literal["knowledge_composer", "smalltalk_composer"],
    ) -> CrewAIAgentAdapterV1:
        raise AssertionError(f"composer factory is not used for {stage}")


@dataclass(frozen=True, slots=True)
class _UnusedAlignmentFactory:
    def build_alignment_verifier_agent(self) -> CrewAIAgentAdapterV1:
        raise AssertionError("alignment factory is not used by this lifecycle scenario")


class LifecycleRuntimeSpy:
    def __init__(self, coordinator: SnapshotCoordinatorSpy, settings: Settings) -> None:
        self.coordinator: ReplyTurnStateCoordinatorV1 = coordinator
        self.snapshot_coordinator: SnapshotCoordinatorSpy = coordinator
        self.settings: Settings = settings
        self.planner_agent_factory: PlannerAgentFactoryV1 = _UnusedPlannerFactory()
        self.composer_agent_factory: ComposerAgentFactoryV1 = _UnusedComposerFactory()
        self.alignment_agent_factory: AlignmentAgentFactoryV1 = (
            _UnusedAlignmentFactory()
        )
        self.alignment_verifier: ReplyAlignmentVerifier | None = None
        self.v2_composer: V2Composer | None = None
        self.captured_action_history: (
            Sequence[RecentExecutedActionSummaryViewV1] | None
        ) = None

    @property
    def candidate_response_builder_v1(self) -> CandidateResponseBuilderV1:
        return self._build_candidate_response

    async def _build_candidate_response(
        self,
        *,
        request: KernelReplyRequestV1,
        domain_context: DomainContextV1,
        policy: PolicyManifestV2,
        model_family: ModelFamily,
        intent_gate: IntentGateResult,
        history: list[ConversationMessage],
        action_history: Sequence[RecentExecutedActionSummaryViewV1],
        prompt_programs: list[PromptProgram],
        llm_executions: list[dict[str, JsonValue]],
        scope_authority: BusinessScopeAuthorityV1,
        state_key_ref: str,
        llm_journal: TurnLlmInvocationJournalV1,
    ) -> V2AttemptResult:
        del (
            request,
            policy,
            model_family,
            intent_gate,
            history,
            prompt_programs,
            llm_executions,
            scope_authority,
            state_key_ref,
            llm_journal,
        )
        from market_support_crewai_agent.runtime.evidence.grounding import (
            CanonicalEvidenceExecutionResultV1,
        )
        from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
            AdapterPreflightSnapshot,
        )
        from market_support_crewai_agent.runtime.planning.models import (
            ComplianceDecisionV1,
            ExecutionPlanV2,
        )
        from market_support_crewai_agent.runtime.v2_attempt import (
            V2AttemptResult,
            V2ReplyValidationResult,
        )

        self.captured_action_history = action_history
        self.snapshot_coordinator.rows = (
            action_history_row(received_at_epoch_ms=1_999_000),
        )
        plan = ExecutionPlanV2.model_construct(
            execution_plan_id="epl1:" + "0" * 64,
            origin="deterministic",
            user_need="snapshot action history test",
            artifact_kind="knowledge_answer",
            response_mode="knowledge_answer",
            compliance=ComplianceDecisionV1(
                is_compliant=True, reason_code="unknown", reason=""
            ),
            units=(),
            selected_manifest_refs=(),
            adapter_resolves=(),
            action_intents=(),
            guardrail_decisions=(),
            confidence=1.0,
            plan_spec=None,
        )
        return V2AttemptResult(
            plan=plan,
            evidence=CanonicalEvidenceExecutionResultV1(
                preflight=AdapterPreflightSnapshot.empty(),
                canonical_facts=(),
                resolve_bindings=(),
                groundings=(),
                domain_context=domain_context,
            ),
            response=ReplyResponse(
                reply=PrimaryReply(kind="answer", text="ok", mentions=[]),
                actions=[],
            ),
            reply_validation=V2ReplyValidationResult(valid=True),
            reason_code="action_ready",
        )
