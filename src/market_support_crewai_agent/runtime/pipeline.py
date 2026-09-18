from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from pydantic import JsonValue
from market_support_crewai_agent.runtime.policy.ontology_models import DomainContextV1
from market_support_crewai_agent.runtime.planning.direct_send import (
    match_direct_send_command,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.planning.input_policy import (
    match_input_policy,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.prompts.context import (
    IntentGateResult,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.planning_flow import (
    PlannerRuntimeV1,
    build_candidate_via_planner,
)
from market_support_crewai_agent.runtime.recall.flow import (
    collect_preplanner_recall,
)
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.recall.flow import PreplannerRecallRuntimeV1
from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.observability.runtime_trace import (
    trace_event,
    trace_span,
)
from market_support_crewai_agent.runtime.v2_attempt import V2AttemptResult
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)


class CandidatePlanBuilderV1(Protocol):
    async def __call__(
        self,
        *,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
        scope_authority: BusinessScopeAuthorityV1,
        state_key_ref: str,
        plan: ExecutionPlanV2,
        recall_state: RecallTurnStateV1 | None = None,
    ) -> V2AttemptResult: ...


class CandidateResponseRuntimeV1(
    PreplannerRecallRuntimeV1,
    PlannerRuntimeV1,
    Protocol,
):
    pass


async def _build_candidate_from_plan(
    runtime: CandidateResponseRuntimeV1,
    *,
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
    state_key_ref: str,
    plan: ExecutionPlanV2,
    recall_state: RecallTurnStateV1 | None = None,
) -> V2AttemptResult:
    builder = runtime.candidate_from_plan_builder_v2
    return await builder(
        request=request,
        policy=policy,
        scope_authority=scope_authority,
        state_key_ref=state_key_ref,
        plan=plan,
        recall_state=recall_state,
    )


async def build_candidate_response(
    runtime: CandidateResponseRuntimeV1,
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
    recall_state: RecallTurnStateV1 | None = None,
    llm_journal: TurnLlmInvocationJournalV1 | None = None,
) -> V2AttemptResult:
    with trace_span("input_policy.match"):
        input_policy = match_input_policy(request, policy, scope_authority)
    if input_policy.matched and input_policy.plan is not None:
        trace_event(
            "input_policy.matched",
            status=input_policy.status,
            reason_code=input_policy.reason_code,
            rule_id=input_policy.rule_id,
        )
        return await _build_candidate_from_plan(
            runtime,
            request=request,
            policy=policy,
            scope_authority=scope_authority,
            state_key_ref=state_key_ref,
            plan=input_policy.plan,
        )

    with trace_span("direct_send.match"):
        direct_send = match_direct_send_command(request, policy, scope_authority)
    if direct_send.matched and direct_send.plan is not None:
        trace_event(
            "direct_send.matched",
            status=direct_send.status,
            reason_code=direct_send.reason_code,
            pattern_id=direct_send.pattern_id,
        )
        return await _build_candidate_from_plan(
            runtime,
            request=request,
            policy=policy,
            scope_authority=scope_authority,
            state_key_ref=state_key_ref,
            plan=direct_send.plan,
        )

    shortcut_plan = None
    if recall_state is None:
        with trace_span("recall.collect"):
            recall = await collect_preplanner_recall(
                runtime,
                request=request,
                policy=policy,
                scope_authority=scope_authority,
            )
        active_recall_state = recall.turn_state
        shortcut_plan = recall.shortcut_plan
    else:
        active_recall_state = recall_state
    trace_event(
        "recall.ready",
        mode=active_recall_state.outcome.mode,
        decision=active_recall_state.outcome.decision,
        candidate_count=len(active_recall_state.outcome.candidates),
        trace_hash=active_recall_state.outcome.trace_hash,
    )
    if shortcut_plan is not None:
        return await _build_candidate_from_plan(
            runtime,
            request=request,
            policy=policy,
            scope_authority=scope_authority,
            state_key_ref=state_key_ref,
            plan=shortcut_plan,
            recall_state=active_recall_state,
        )

    if llm_journal is None:
        return await build_candidate_via_planner(
            runtime,
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
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
            recall_state=active_recall_state,
        )
    return await build_candidate_via_planner(
        runtime,
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
        alignment_verdict=alignment_verdict,
        alignment_attempt=alignment_attempt,
        recall_state=active_recall_state,
        llm_journal=llm_journal,
    )
