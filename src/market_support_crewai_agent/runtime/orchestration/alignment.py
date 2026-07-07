from __future__ import annotations

from pydantic import JsonValue
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.profiles import ModelFamily
from market_support_crewai_agent.runtime.llm.prompting.context import (
    IntentGateResult,
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.llm.prompting.router import (
    select_prompt_program,
)
from market_support_crewai_agent.runtime.orchestration.crewai_io import (
    coerce_alignment_verdict,
    run_crewai_kickoff,
    safe_short_text,
)
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.orchestration.planner import record_llm_failure
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.state.runtime_trace import (
    trace_event,
    trace_span,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError, AttemptResult
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import (
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
)


async def verify_reply_alignment(
    runtime,
    *,
    request: ReplyRequest,
    policy: PolicyManifest,
    model_family: ModelFamily,
    intent_gate: IntentGateResult,
    history: list[ConversationMessage],
    action_history: list[ActionLedgerRecord],
    prompt_programs: list[PromptProgram],
    llm_executions: list[dict[str, JsonValue]],
    candidate: AttemptResult,
    attempt: int,
) -> ReplyAlignmentVerdict:
    if runtime.alignment_verifier is not None:
        with trace_span("alignment.external_verifier", attempt=attempt):
            verdict = await runtime.alignment_verifier.verify(
                request=request,
                domain_context=candidate.domain_context,
                plan=candidate.plan,
                directive=candidate.directive,
                evidence_facts=candidate.evidence_facts,
                business_facts=candidate.business_facts,
                response=candidate.response,
                guardrail_decisions=candidate.guardrail_decisions,
                attempt=attempt,
            )
        return ReplyAlignmentVerdict.model_validate(verdict)

    with trace_span("alignment.project_context", attempt=attempt):
        verifier_context = runtime._project_context(
            stage="alignment_verifier",
            request=request,
            domain_context=candidate.domain_context,
            policy=policy,
            intent_gate=intent_gate,
            execution_plan=candidate.plan,
            plan_validation=candidate.plan_validation,
            preflight=candidate.preflight,
            evidence_facts=candidate.evidence_facts,
            business_facts=candidate.business_facts,
            answerability_assessment=candidate.answerability,
            guardrail_decisions=candidate.guardrail_decisions,
            history=history,
            action_history=action_history,
            candidate_response=candidate.response,
            alignment_attempt=attempt,
        )
    with trace_span("alignment.assemble_prompt", attempt=attempt):
        verifier_program = select_prompt_program(
            PromptAssemblyContext(
                stage="alignment_verifier",
                model_family=model_family,
                request=request,
                model_visible_context=verifier_context,
                domain_context=candidate.domain_context,
                policy=policy,
                intent_gate=intent_gate,
                execution_plan=candidate.plan,
                plan_validation=candidate.plan_validation,
                preflight=candidate.preflight,
                evidence_facts=candidate.evidence_facts,
                business_facts=candidate.business_facts,
                answerability_assessment=candidate.answerability,
                guardrail_decisions=candidate.guardrail_decisions,
                history=history,
                action_history=action_history,
                candidate_response=candidate.response,
                alignment_attempt=attempt,
            )
        )
    prompt_programs.append(verifier_program)
    verifier_agent = None
    try:
        with trace_span("alignment.build_agent", attempt=attempt):
            verifier_agent = runtime._build_alignment_verifier_agent()
        result, verifier_execution = await run_crewai_kickoff(
            verifier_agent,
            verifier_program,
            timeout_seconds=runtime.settings.llm_timeout_seconds,
        )
        llm_executions.append(verifier_execution)
    except TimeoutError as exc:
        if verifier_agent is not None:
            record_llm_failure(verifier_agent, "alignment_verifier", "timeout")
        raise AgentRuntimeError("CrewAI alignment verifier timed out") from exc
    except Exception as exc:
        if verifier_agent is not None:
            record_llm_failure(
                verifier_agent,
                "alignment_verifier",
                safe_short_text(exc) or "alignment verifier failed",
            )
        raise AgentRuntimeError("CrewAI alignment verifier failed") from exc

    with trace_span("alignment.coerce_verdict", attempt=attempt):
        verdict = coerce_alignment_verdict(result)
    if verdict is None:
        if verifier_agent is not None:
            record_llm_failure(
                verifier_agent,
                "alignment_verifier",
                "invalid ReplyAlignmentVerdict contract",
            )
        raise AgentRuntimeError(
            "CrewAI alignment verifier returned an invalid ReplyAlignmentVerdict contract"
        )
    trace_event(
        "state.alignment_verdict",
        attempt=attempt,
        aligned=verdict.aligned,
        safe_to_return=verdict.safe_to_return,
        remediation=verdict.remediation,
        failure_code=verdict.failure_code,
    )
    return verdict


def fallback_attempt(
    runtime,
    candidate: AttemptResult,
    policy: PolicyManifest,
    *,
    kind: str,
    reason_code: str,
) -> AttemptResult:
    if candidate.plan.compliance.is_compliant is False:
        return candidate
    if kind == "clarification":
        directive = ResponseDirective(
            mode="clarification",
            reply_kind="clarification",
            text="我需要再确认一下你具体想要哪类内容。",
            reason_code=reason_code,
        )
    else:
        directive = ResponseDirective(
            mode="unable",
            reply_kind="unable_to_answer",
            text="老师，这个信息我这边暂时无法确认，先不回答避免信息不准确。",
            reason_code=reason_code,
        )
    response = ReplyResponse(
        reply=PrimaryReply(
            kind=directive.reply_kind,
            text=directive.text,
            mentions=[],
        ),
        actions=[],
    )
    return runtime._validated_attempt(
        plan=candidate.plan,
        plan_validation=candidate.plan_validation,
        preflight=candidate.preflight,
        evidence_facts=candidate.evidence_facts,
        business_facts=candidate.business_facts,
        domain_context=candidate.domain_context,
        answerability=candidate.answerability,
        directive=directive,
        response=response,
        policy=policy,
        guardrail_decisions=candidate.guardrail_decisions,
    )
