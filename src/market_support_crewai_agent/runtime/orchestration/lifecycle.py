from __future__ import annotations

import logging

from pydantic import JsonValue

from market_support_crewai_agent.runtime.domain.ontology import DomainContextBuilder
from market_support_crewai_agent.runtime.domain.policy import (
    compile_policy,
    ledger_summary_from_action_history,
)
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.router import (
    model_family_from_settings,
    route_intent,
)
from market_support_crewai_agent.runtime.orchestration.reply_history import (
    compact_assistant_result,
)
from market_support_crewai_agent.runtime.orchestration.direct_message import (
    direct_outbound_ready,
)
from market_support_crewai_agent.runtime.orchestration.attempt_validation import (
    reply_validation_error_summary,
    skip_alignment_verifier,
)
from market_support_crewai_agent.runtime.state.runtime_trace import (
    RuntimeTrace,
    trace_event,
    trace_span,
    use_runtime_trace,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.validation.reply_validator import (
    ReplyContractError,
)
from market_support_crewai_agent.runtime.validation.request_input_guard import (
    validate_reply_request_input,
)
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse

logger = logging.getLogger(__name__)


async def run_reply_turn(runtime, request: ReplyRequest) -> ReplyResponse:
    trace = RuntimeTrace(
        context={
            "context_id": request.context_id,
            "conversation_key": request.conversation_key,
        }
    )
    with use_runtime_trace(trace):
        try:
            with trace_span("request.validate"):
                validate_reply_request_input(request, runtime.settings)
                if not runtime.settings.llm_api_key:
                    raise AgentRuntimeError("YANFU_LLM_API_KEY is not configured")

            with trace_span("state.load_history"):
                history = runtime.conversation_store.get_recent(
                    request.conversation_key
                )
                action_history = runtime.action_ledger.recent_executed_for_conversation(
                    request.conversation_key,
                    limit=20,
                )
            trace_event(
                "state.history_loaded",
                history_count=len(history),
                action_history_count=len(action_history),
            )

            with trace_span("domain.build_context"):
                domain_context = DomainContextBuilder().build(
                    request,
                    conversation_metadata={
                        "context_id": request.context_id,
                        "conversation_key": request.conversation_key,
                    },
                )

            model_family = model_family_from_settings(runtime.settings)
            outbound_messaging_enabled = (
                await direct_outbound_ready(runtime) if not request.is_group else False
            )
            with trace_span("policy.compile"):
                policy = compile_policy(
                    request,
                    ledger_summary=ledger_summary_from_action_history(action_history),
                    doc_mcp_enabled=bool(
                        runtime.settings.doc_mcp_enabled
                        and runtime.settings.doc_mcp_base_url
                    ),
                    doc_mcp_allowed_channel_types=runtime.settings.doc_mcp_allowed_channel_types,
                    outbound_messaging_enabled=outbound_messaging_enabled,
                )
            trace_event(
                "state.policy_compiled",
                allowed_capabilities=policy.allowed_capabilities,
                allowed_actions=policy.allowed_outbound_actions,
            )

            with trace_span("intent.route"):
                intent_gate = route_intent(
                    request,
                    policy,
                    history=history,
                )
            llm_executions: list[dict[str, JsonValue]] = []
            prompt_programs: list[PromptProgram] = []
            alignment_verdicts: list[ReplyAlignmentVerdict] = []
            alignment_remediations: list[dict[str, JsonValue]] = []

            with trace_span("candidate.build"):
                candidate = await runtime._build_candidate_response(
                    request=request,
                    domain_context=domain_context,
                    policy=policy,
                    model_family=model_family,
                    intent_gate=intent_gate,
                    history=history,
                    action_history=action_history,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                )
            if (
                candidate.reply_validation.valid
                and runtime.settings.reply_alignment_verifier_enabled
                and not skip_alignment_verifier(candidate)
            ):
                with trace_span("alignment.ensure"):
                    candidate = await runtime._ensure_aligned_response(
                        request=request,
                        domain_context=domain_context,
                        policy=policy,
                        model_family=model_family,
                        intent_gate=intent_gate,
                        history=history,
                        action_history=action_history,
                        prompt_programs=prompt_programs,
                        llm_executions=llm_executions,
                        alignment_verdicts=alignment_verdicts,
                        alignment_remediations=alignment_remediations,
                        candidate=candidate,
                    )
            trace_event(
                "state.candidate_ready",
                reply_kind=candidate.response.reply.kind,
                action_count=len(candidate.response.actions),
                reply_valid=candidate.reply_validation.valid,
                alignment_verdict_count=len(alignment_verdicts),
            )

            with trace_span("audit.record"):
                runtime._record_audit_trace(
                    request=request,
                    policy=policy,
                    plan=candidate.plan,
                    directive=candidate.directive,
                    plan_validation=candidate.plan_validation,
                    action_history=action_history,
                    domain_context=candidate.domain_context,
                    preflight=candidate.preflight,
                    evidence_facts=candidate.evidence_facts,
                    business_facts=candidate.business_facts,
                    answerability_assessment=candidate.answerability,
                    response=candidate.response,
                    reply_validation=candidate.reply_validation,
                    guardrail_decisions=candidate.guardrail_decisions,
                    intent_gate=intent_gate,
                    prompt_programs=prompt_programs,
                    llm_executions=llm_executions,
                    alignment_verdicts=alignment_verdicts,
                    alignment_remediations=alignment_remediations,
                    runtime_trace=trace.to_dict(),
                )
            if not candidate.reply_validation.valid:
                raise ReplyContractError(
                    "rendered reply failed validation: {}".format(
                        reply_validation_error_summary(candidate.reply_validation)
                    )
                )

            with trace_span("state.save_turn"):
                runtime.conversation_store.save_turn(
                    request.conversation_key,
                    request.message,
                    compact_assistant_result(
                        candidate.response,
                        candidate.plan,
                        candidate.pending_outbound_draft,
                    ),
                )
            return candidate.response
        finally:
            trace.log_trace(
                logger,
                context_id=request.context_id,
                conversation_key=request.conversation_key,
            )
    raise AssertionError("unreachable reply runtime path")
