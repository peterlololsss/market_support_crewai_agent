from __future__ import annotations

from collections.abc import Sequence

from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology_models import DomainContextV1
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.recall.flow import (
    PreplannerRecallRuntimeV1,
    collect_preplanner_recall,
)
from market_support_crewai_agent.runtime.recall.turn_state import (
    PreplannerRecallTransitionV1,
    RecallTurnStateV1,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from tests.integration.runtime._recall_pipeline_fixtures import planner_attempt_result
from tests.integration.runtime._recall_pipeline_runtime_fixtures import (
    RecallPipelineRuntime,
)


class RecallCollectionRecorder:
    def __init__(self) -> None:
        self.states: list[RecallTurnStateV1] = []

    async def __call__(
        self,
        runtime_arg: PreplannerRecallRuntimeV1,
        *,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
        scope_authority: BusinessScopeAuthorityV1,
    ) -> PreplannerRecallTransitionV1:
        transition = await collect_preplanner_recall(
            runtime_arg,
            request=request,
            policy=policy,
            scope_authority=scope_authority,
        )
        self.states.append(transition.turn_state)
        return transition


class PlannerStateRecorder:
    def __init__(self) -> None:
        self.states: list[RecallTurnStateV1] = []

    async def __call__(
        self,
        _runtime: RecallPipelineRuntime,
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
        alignment_verdict: ReplyAlignmentVerdict | None = None,
        alignment_attempt: int = 0,
        recall_state: RecallTurnStateV1,
        llm_journal: TurnLlmInvocationJournalV1 | None = None,
    ) -> V2AttemptResult:
        del (
            _runtime,
            domain_context,
            model_family,
            intent_gate,
            history,
            action_history,
            prompt_programs,
            llm_executions,
            state_key_ref,
            alignment_verdict,
            alignment_attempt,
            llm_journal,
        )
        self.states.append(recall_state)
        return planner_attempt_result(
            request=request,
            policy=policy,
            scope_authority=scope_authority,
            recall_state=recall_state,
        )
