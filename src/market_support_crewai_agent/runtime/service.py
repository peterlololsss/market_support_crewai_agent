from __future__ import annotations

from collections.abc import Sequence

from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.context.payload_store import (
    ScopedContextPayloadStoreV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
)
from market_support_crewai_agent.runtime.integrations.crewai.agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.lifecycle import run_reply_turn
from market_support_crewai_agent.runtime.lifecycle_contracts import (
    AlignmentAgentFactoryV1,
    CandidateResponseBuilderV1,
    ComposerAgentFactoryV1,
)
from market_support_crewai_agent.runtime.pipeline import (
    CandidatePlanBuilderV1,
    build_candidate_response,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.planning_flow import PlannerAgentFactoryV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology_models import DomainContextV1
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.recall.document_qa_recall import (
    DocumentQaSearchService,
)
from market_support_crewai_agent.runtime.recall.service import (
    QuestionRecallService,
)
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.rendering.v2_composer import V2Composer
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
    ConversationStore,
)
from market_support_crewai_agent.runtime.state.coordinator_protocol import (
    ReplyTurnStateCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.turn import (
    RuntimeDeps,
    build_runtime_deps,
)
from market_support_crewai_agent.runtime.v2_attempt import (
    V2AttemptResult,
    build_candidate_from_plan_v2,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerifier,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings


class CrewAIReplyRuntime:
    """CrewAI runtime boundary used by the FastAPI transport layer."""

    def __init__(
        self,
        settings: Settings | RuntimeDeps,
        conversation_store: ConversationStore | None = None,
        action_ledger: ActionLedger | None = None,
        preflight_service: AdapterPreflightService | None = None,
        evidence_executor: EvidenceExecutor | None = None,
        context_payload_store: ScopedContextPayloadStoreV1 | None = None,
        coordinator: ReplyStateTransactionCoordinatorV1 | None = None,
        internal_company_knowledge_gateway: InternalCompanyKnowledgeGatewayV1
        | None = None,
        document_cache_config: DocumentMcpCacheConfigV1 | None = None,
        alignment_verifier: ReplyAlignmentVerifier | None = None,
        v2_composer: V2Composer | None = None,
    ) -> None:
        deps = (
            settings
            if isinstance(settings, RuntimeDeps)
            else build_runtime_deps(
                settings=settings,
                conversation_store=conversation_store,
                action_ledger=action_ledger,
                preflight_service=preflight_service,
                evidence_executor=evidence_executor,
                context_payload_store=context_payload_store,
                coordinator=coordinator,
                internal_company_knowledge_gateway=internal_company_knowledge_gateway,
                document_cache_config=document_cache_config,
                alignment_verifier=alignment_verifier,
                v2_composer=v2_composer,
            )
        )
        self.settings: Settings = deps.settings
        self.conversation_store: ConversationStore = deps.conversation_store
        self.action_ledger: ActionLedger = deps.action_ledger
        self.preflight_service: AdapterPreflightService = deps.preflight_service
        self.evidence_executor: EvidenceExecutor = deps.evidence_executor
        self.qa_search_service: DocumentQaSearchService = DocumentQaSearchService(
            deps.settings
        )
        self.question_recall_service: QuestionRecallService = QuestionRecallService(
            deps.settings
        )
        self.alignment_verifier: ReplyAlignmentVerifier | None = deps.alignment_verifier
        agent_factory = CrewAIAgentFactory(deps.settings)
        self.planner_agent_factory: PlannerAgentFactoryV1 = agent_factory
        self.composer_agent_factory: ComposerAgentFactoryV1 = agent_factory
        self.alignment_agent_factory: AlignmentAgentFactoryV1 = agent_factory
        self.locator_safety: LocatorSafetyClassifierV1 = (
            LocatorSafetyClassifierV1.from_settings(deps.settings)
        )
        self.coordinator: ReplyTurnStateCoordinatorV1 = deps.coordinator
        self.document_cache_config: DocumentMcpCacheConfigV1 | None = (
            deps.document_cache_config
        )
        self.v2_composer: V2Composer | None = deps.v2_composer

    async def reply(self, envelope: VerifiedRequestEnvelopeV1) -> ReplyResponse:
        return await run_reply_turn(self, envelope)

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
        return await build_candidate_response(
            self,
            request=request,
            domain_context=domain_context,
            policy=policy,
            model_family=model_family,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            scope_authority=scope_authority,
            state_key_ref=state_key_ref,
            llm_journal=llm_journal,
        )

    @property
    def candidate_response_builder_v1(self) -> CandidateResponseBuilderV1:
        return self._build_candidate_response

    @property
    def candidate_from_plan_builder_v2(self) -> CandidatePlanBuilderV1:
        return self._build_candidate_from_plan_v2

    async def _build_candidate_from_plan_v2(
        self,
        *,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
        scope_authority: BusinessScopeAuthorityV1,
        state_key_ref: str,
        plan: ExecutionPlanV2,
        recall_state: RecallTurnStateV1 | None = None,
    ) -> V2AttemptResult:
        return await build_candidate_from_plan_v2(
            self,
            request=request,
            policy=policy,
            scope_authority=scope_authority,
            state_key_ref=state_key_ref,
            plan=plan,
            recall_state=recall_state,
        )


async def build_reply(
    envelope: VerifiedRequestEnvelopeV1 | ReplyRequestV2,
    *,
    settings: Settings | None = None,
) -> ReplyResponse:
    if not isinstance(envelope, VerifiedRequestEnvelopeV1):
        raise TypeError("build_reply requires VerifiedRequestEnvelopeV1")
    runtime = CrewAIReplyRuntime(settings or get_settings())
    return await runtime.reply(envelope)
