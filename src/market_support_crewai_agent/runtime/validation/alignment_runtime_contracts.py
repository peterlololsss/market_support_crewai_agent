from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.common_view_models import (
    HistoryTurnViewV1,
)
from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.context.retry_view_models import (
    ComposerRetryOverlayV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
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
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
    ReplyAlignmentVerifier,
)
from market_support_crewai_agent.settings_model import Settings


class ComposerAgentFactoryV1(Protocol):
    def build_composer_agent(
        self,
        stage: Literal["knowledge_composer", "smalltalk_composer"],
    ) -> CrewAIAgentAdapterV1: ...


class AlignmentAgentFactoryV1(Protocol):
    def build_alignment_verifier_agent(self) -> CrewAIAgentAdapterV1: ...


@dataclass(frozen=True, slots=True)
class RuntimeAlignmentContextV1:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    model_family: ModelFamily
    intent_gate: IntentGateResult
    history: Sequence[ConversationMessage]
    action_history: Sequence[RecentExecutedActionSummaryViewV1]
    prompt_programs: list[PromptProgram]
    llm_executions: list[dict[str, JsonValue]]
    alignment_verdicts: list[ReplyAlignmentVerdict]
    alignment_remediations: list[dict[str, JsonValue]]
    scope_authority: BusinessScopeAuthorityV1
    state_key_ref: str
    now: datetime


class AlignmentRuntimeV1(Protocol):
    @property
    def settings(self) -> Settings: ...

    @property
    def alignment_verifier(self) -> ReplyAlignmentVerifier | None: ...

    @property
    def planner_agent_factory(self) -> PlannerAgentFactoryV1: ...

    @property
    def composer_agent_factory(self) -> ComposerAgentFactoryV1: ...

    @property
    def alignment_agent_factory(self) -> AlignmentAgentFactoryV1: ...


@runtime_checkable
class AlignmentPlanningFlowV1(Protocol):
    async def build_candidate_via_planner(
        self,
        runtime: AlignmentRuntimeV1,
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
        alignment_verdict: ReplyAlignmentVerdict | None,
        alignment_attempt: int,
        recall_state: RecallTurnStateV1,
        llm_journal: TurnLlmInvocationJournalV1 | None = None,
    ) -> V2AttemptResult: ...


@runtime_checkable
class V2AttemptTransportV1(Protocol):
    async def build_candidate_from_plan_v2(
        self,
        runtime: AlignmentRuntimeV1,
        *,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
        scope_authority: BusinessScopeAuthorityV1,
        plan: ExecutionPlanV2,
        state_key_ref: str | None = None,
        recall_state: RecallTurnStateV1 | None = None,
        alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
    ) -> V2AttemptResult: ...

    async def recompose_candidate_v2(
        self,
        runtime: AlignmentRuntimeV1,
        *,
        candidate: V2AttemptResult,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
        scope_authority: BusinessScopeAuthorityV1,
        history: tuple[HistoryTurnViewV1, ...],
        retry_overlay: ComposerRetryOverlayV1,
    ) -> V2AttemptResult: ...
