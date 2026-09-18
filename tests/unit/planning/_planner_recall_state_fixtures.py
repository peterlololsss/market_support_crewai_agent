from __future__ import annotations

from dataclasses import dataclass
from typing import final

from market_support_crewai_agent.runtime.context.stage_inputs import (
    PlannerPromptInputV1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.planning_flow import (
    PlanCandidateBuilderV1,
    PlannerAgentFactoryV1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.v2_attempt import (
    V2AttemptResult,
    V2ReplyValidationResult,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_agent_adapter


@dataclass(frozen=True, slots=True)
class _FakeProfile:
    stage: str = "planner_intent"


@dataclass(frozen=True, slots=True)
class FakeProgram:
    name: str
    profile: _FakeProfile = _FakeProfile()
    scene_key: str = "wecom_group.v1"


@final
class PlannerRuntime:
    def __init__(self) -> None:
        self.settings: Settings = Settings.model_construct(
            llm_timeout_seconds=1.0,
            planner_transient_retry_attempts=0,
            planner_transient_retry_base_seconds=0.0,
            planner_llm_provider="gemini",
            planner_llm_model="gemini-planner",
            planner_llm_base_url="https://planner.test/v1beta",
            llm_provider="openai",
            llm_model="generic-composer",
            llm_base_url="https://composer.test/v1",
            llm_api_key="generic-test-key",
        )
        self.planner_agent_factory: PlannerAgentFactoryV1 = self
        self.primary_agent: CrewAIAgentAdapterV1 = make_agent_adapter(role="primary")
        self.received_states: list[RecallTurnStateV1] = []

    @property
    def candidate_from_plan_builder_v2(self) -> PlanCandidateBuilderV1:
        return self._build_candidate_from_plan_v2

    def build_planner_agent(self) -> CrewAIAgentAdapterV1:
        return self.primary_agent

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
        if recall_state is None:
            raise AssertionError("recall_state_required")
        self.received_states.append(recall_state)
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
                reply=PrimaryReply(kind="answer", text="planner answer"),
                actions=[],
            ),
            reply_validation=V2ReplyValidationResult(valid=True),
            reason_code="question_recall_hit",
            recall_state=recall_state,
        )


@dataclass(frozen=True, slots=True)
class KickoffObservation:
    agent: CrewAIAgentAdapterV1
    program: FakeProgram
    planner_input: PlannerPromptInputV1
    settings: Settings | None


@dataclass(frozen=True, slots=True)
class FakeFrame:
    name: str
