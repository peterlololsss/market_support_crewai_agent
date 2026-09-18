from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIKickoffOutputV1,
    CrewAILlmAdapterV1,
)
from market_support_crewai_agent.runtime.pipeline import (
    CandidatePlanBuilderV1,
    build_candidate_response,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.planning_flow import PlannerAgentFactoryV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaMatch,
)
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
)
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.v2_attempt import (
    V2AttemptResult,
    V2ReplyValidationResult,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from market_support_crewai_agent.settings_model import Settings

PipelineContext = tuple[
    VerifiedRequestEnvelopeV1,
    PolicyManifestV2,
    BusinessScopeAuthorityV1,
]


async def _planner_agent_kickoff_stub(
    prompt: str,
    response_format: type[BaseModel],
) -> CrewAIKickoffOutputV1:
    del prompt, response_format
    raise AssertionError("planner_agent_stub_not_executed")


def _planner_agent_stub() -> CrewAIAgentAdapterV1:
    llm = CrewAILlmAdapterV1(
        provider="test",
        model="test",
        api_key=None,
        base_url=None,
        client_params={},
        max_tokens=None,
        max_output_tokens=None,
        timeout=None,
        temperature=None,
        top_p=None,
        top_k=None,
        stop_sequences=None,
        thinking_config=None,
    )
    return CrewAIAgentAdapterV1(
        role="planner",
        llm=llm,
        kickoff_async=_planner_agent_kickoff_stub,
    )


class StaticCollectorProtocol(Protocol):
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> ApprovedStaticRecallCollectionV1: ...


class DocumentCollectorProtocol(Protocol):
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> KnowledgeQaMatch: ...


class StaticCollector:
    def __init__(self, result: ApprovedStaticRecallCollectionV1) -> None:
        self.result: ApprovedStaticRecallCollectionV1 = result
        self.calls: int = 0

    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> ApprovedStaticRecallCollectionV1:
        del request, policy
        self.calls += 1
        return self.result


class DocumentCollector:
    def __init__(self, result: KnowledgeQaMatch) -> None:
        self.result: KnowledgeQaMatch = result
        self.calls: int = 0

    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> KnowledgeQaMatch:
        del request, policy
        self.calls += 1
        return self.result


class RecallPipelineRuntime:
    def __init__(
        self,
        static: StaticCollectorProtocol,
        document: DocumentCollectorProtocol,
    ) -> None:
        self.settings: Settings = Settings.model_construct()
        self.question_recall_service: StaticCollectorProtocol = static
        self.qa_search_service: DocumentCollectorProtocol = document
        self.plans: list[ExecutionPlanV2] = []
        self.recall_states: list[RecallTurnStateV1 | None] = []
        self.planner_agent_factory: PlannerAgentFactoryV1 = self

    @property
    def candidate_from_plan_builder_v2(self) -> CandidatePlanBuilderV1:
        return self._build_candidate_from_plan_v2

    def build_planner_agent(self) -> CrewAIAgentAdapterV1:
        return _planner_agent_stub()

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
        del policy, state_key_ref
        self.plans.append(plan)
        self.recall_states.append(recall_state)
        return V2AttemptResult(
            plan=plan,
            evidence=CanonicalEvidenceExecutionResultV1(
                preflight=AdapterPreflightSnapshot.empty(),
                canonical_facts=(),
                resolve_bindings=(),
                groundings=(),
                domain_context=DomainContextV1Builder().build(
                    request,
                    scope_authority=scope_authority,
                ),
            ),
            response=ReplyResponse(
                reply=PrimaryReply(kind="answer", text="shortcut answer"),
                actions=[],
            ),
            reply_validation=V2ReplyValidationResult(valid=True),
            reason_code="question_recall_hit",
            recall_state=recall_state,
        )


async def run_pipeline(
    runtime: RecallPipelineRuntime,
    context: PipelineContext,
) -> V2AttemptResult:
    envelope, policy, scope = context
    return await build_candidate_response(
        runtime,
        request=envelope.request,
        domain_context=DomainContextV1Builder().build(
            envelope.request,
            scope_authority=scope,
        ),
        policy=policy,
        model_family="generic",
        intent_gate=IntentGateResult(artifact_hint="unclear"),
        history=[],
        action_history=[],
        prompt_programs=[],
        llm_executions=[],
        scope_authority=scope,
        state_key_ref=envelope.state_key_ref,
    )
