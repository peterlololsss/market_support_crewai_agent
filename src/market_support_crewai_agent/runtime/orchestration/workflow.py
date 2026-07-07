from __future__ import annotations

from pydantic import JsonValue
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning.direct_send import (
    match_direct_send_command,
)
from market_support_crewai_agent.runtime.domain.planning.input_policy import (
    match_input_policy,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.profiles import ModelFamily
from market_support_crewai_agent.runtime.llm.prompting.context import (
    IntentGateResult,
)
from market_support_crewai_agent.runtime.orchestration.planning_workflow import (
    build_candidate_via_planner,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.state.runtime_trace import (
    trace_event,
    trace_span,
)
from market_support_crewai_agent.runtime.turn import AttemptResult
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import ReplyRequest


async def build_candidate_response(
    runtime,
    *,
    request: ReplyRequest,
    domain_context: DomainContext,
    policy: PolicyManifest,
    model_family: ModelFamily,
    intent_gate: IntentGateResult,
    history: list[ConversationMessage],
    action_history: list[ActionLedgerRecord],
    prompt_programs: list[PromptProgram],
    llm_executions: list[dict[str, JsonValue]],
    alignment_verdict: ReplyAlignmentVerdict | None = None,
    alignment_attempt: int = 0,
) -> AttemptResult:
    with trace_span("input_policy.match"):
        input_policy = match_input_policy(request, policy)
    if input_policy.matched and input_policy.plan is not None:
        trace_event(
            "input_policy.matched",
            status=input_policy.status,
            reason_code=input_policy.reason_code,
            rule_id=input_policy.rule_id,
        )
        return await runtime._build_candidate_from_plan(
            request=request,
            domain_context=domain_context,
            policy=policy,
            model_family=model_family,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            plan=input_policy.plan,
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
        )

    with trace_span("direct_send.match"):
        direct_send = match_direct_send_command(request, policy)
    if direct_send.matched and direct_send.plan is not None:
        trace_event(
            "direct_send.matched",
            status=direct_send.status,
            reason_code=direct_send.reason_code,
            pattern_id=direct_send.pattern_id,
        )
        return await runtime._build_candidate_from_plan(
            request=request,
            domain_context=domain_context,
            policy=policy,
            model_family=model_family,
            intent_gate=intent_gate,
            history=history,
            action_history=action_history,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            plan=direct_send.plan,
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
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
        alignment_verdict=alignment_verdict,
        alignment_attempt=alignment_attempt,
    )
