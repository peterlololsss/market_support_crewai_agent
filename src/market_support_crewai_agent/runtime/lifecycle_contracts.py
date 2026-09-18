from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.planning_flow import PlannerAgentFactoryV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology_models import DomainContextV1
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
from market_support_crewai_agent.runtime.state.coordinator_protocol import (
    ReplyTurnStateCoordinatorV1,
)
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
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


class CandidateResponseBuilderV1(Protocol):
    async def __call__(
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
    ) -> V2AttemptResult: ...


class ReplyTurnRuntimeV1(Protocol):
    settings: Settings
    coordinator: ReplyTurnStateCoordinatorV1
    planner_agent_factory: PlannerAgentFactoryV1
    composer_agent_factory: ComposerAgentFactoryV1
    alignment_agent_factory: AlignmentAgentFactoryV1
    alignment_verifier: ReplyAlignmentVerifier | None
    v2_composer: V2Composer | None

    @property
    def candidate_response_builder_v1(self) -> CandidateResponseBuilderV1: ...
