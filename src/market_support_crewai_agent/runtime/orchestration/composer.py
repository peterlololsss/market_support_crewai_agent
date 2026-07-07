from __future__ import annotations

from pydantic import JsonValue
from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.evidence.models import EvidenceFact
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.profiles import ModelFamily
from market_support_crewai_agent.runtime.llm.prompting.context import (
    IntentGateResult,
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.llm.prompting.router import (
    select_prompt_program,
)
from market_support_crewai_agent.runtime.llm.retry import RetryPolicy, run_with_retry
from market_support_crewai_agent.runtime.orchestration.crewai_io import (
    coerce_agent_response,
    coerce_composer_output,
    run_crewai_kickoff,
    safe_short_text,
)
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.orchestration.planner import record_llm_failure
from market_support_crewai_agent.runtime.orchestration.response_renderer import (
    render_directive,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.state.runtime_trace import (
    trace_event,
    trace_span,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.validation.reply_validator import (
    remove_pre_execution_send_claims,
)
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse


async def compose_or_render_response(
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
    plan: ExecutionPlan,
    plan_validation: PlanValidationResult,
    preflight: AdapterPreflightSnapshot,
    evidence_facts: list[EvidenceFact],
    business_facts: BusinessFacts,
    answerability: AnswerabilityAssessment,
    directive: ResponseDirective,
    alignment_verdict: ReplyAlignmentVerdict | None = None,
    alignment_attempt: int = 0,
    guardrail_decisions: list[GuardrailDecision] | None = None,
) -> tuple[ReplyResponse, ComposerReplyOutput | None]:
    if not directive.requires_knowledge_composer:
        with trace_span("reply.render_directive"):
            return (
                render_directive(
                    directive,
                    plan,
                    business_facts,
                    evidence_facts,
                ),
                None,
            )

    composer_stage = directive.composer_stage or "knowledge_composer"
    with trace_span("composer.project_context", stage=composer_stage):
        composer_context = runtime._project_context(
            stage=composer_stage,
            request=request,
            domain_context=domain_context,
            policy=policy,
            intent_gate=intent_gate,
            execution_plan=plan,
            plan_validation=plan_validation,
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            answerability_assessment=answerability,
            guardrail_decisions=guardrail_decisions or [],
            history=history,
            action_history=action_history,
            alignment_verdict=alignment_verdict,
            alignment_attempt=alignment_attempt,
        )
    with trace_span("composer.assemble_prompt", stage=composer_stage):
        composer_program = select_prompt_program(
            PromptAssemblyContext(
                stage=composer_stage,
                model_family=model_family,
                request=request,
                model_visible_context=composer_context,
                domain_context=domain_context,
                policy=policy,
                intent_gate=intent_gate,
                execution_plan=plan,
                plan_validation=plan_validation,
                preflight=preflight,
                evidence_facts=evidence_facts,
                business_facts=business_facts,
                answerability_assessment=answerability,
                guardrail_decisions=guardrail_decisions or [],
                history=history,
                action_history=action_history,
                alignment_verdict=alignment_verdict,
                alignment_attempt=alignment_attempt,
            )
        )
    prompt_programs.append(composer_program)
    try:
        with trace_span("composer.build_agent", stage=composer_stage):
            composer_agent = runtime._build_agent(composer_stage)
        result, composer_executions = await run_composer_kickoff_with_retry(
            composer_agent,
            composer_program,
            timeout_seconds=runtime.settings.llm_timeout_seconds,
            retry_attempts=runtime.settings.planner_transient_retry_attempts,
            base_delay_seconds=runtime.settings.planner_transient_retry_base_seconds,
        )
        llm_executions.extend(composer_executions)
    except TimeoutError as exc:
        raise AgentRuntimeError("CrewAI composer timed out") from exc
    except Exception as exc:
        raise AgentRuntimeError("CrewAI composer failed") from exc

    with trace_span("composer.coerce_response"):
        composer_output = coerce_composer_output(result)
        response = (
            composer_output.to_reply_response()
            if composer_output is not None
            else coerce_agent_response(result)
        )
    if response is None:
        record_llm_failure(
            composer_agent,
            composer_stage,
            "invalid ReplyResponse contract",
        )
        raise AgentRuntimeError(
            "CrewAI composer returned an invalid ReplyResponse contract"
        )
    if directive.mode == "action" and directive.action_intents:
        reply_text = remove_pre_execution_send_claims(response.reply.text)
        reply = response.reply.model_copy(update={"text": reply_text})
        if composer_output is not None:
            composer_output = composer_output.model_copy(update={"reply": reply})
        with trace_span("reply.render_action_from_composer"):
            rendered = render_directive(
                directive.model_copy(
                    update={
                        "text": reply_text,
                        "requires_knowledge_composer": False,
                        "composer_stage": None,
                    }
                ),
                plan,
                business_facts,
                evidence_facts,
            )
        return ReplyResponse(reply=reply, actions=rendered.actions), composer_output
    return response, composer_output


async def run_composer_kickoff_with_retry(
    composer_agent,
    composer_program: PromptProgram,
    *,
    timeout_seconds: float | None,
    retry_attempts: int,
    base_delay_seconds: float,
) -> tuple[object, list[dict[str, JsonValue]]]:
    executions: list[dict[str, JsonValue]] = []

    async def call():
        result, execution = await run_crewai_kickoff(
            composer_agent,
            composer_program,
            timeout_seconds=timeout_seconds,
        )
        executions.append(execution)
        return result

    try:
        result = await run_with_retry(
            call,
            policy=RetryPolicy(
                retry_attempts=retry_attempts,
                base_delay_seconds=base_delay_seconds,
            ),
            should_retry_exception=_composer_exception_retry_reason,
            on_retry=_trace_composer_retry,
        )
    except Exception as exc:
        record_llm_failure(
            composer_agent,
            composer_program.profile.stage,
            safe_short_text(exc) or "composer_call_failed",
        )
        raise
    return result, executions


def _composer_exception_retry_reason(exc: Exception) -> str | None:
    return safe_short_text(exc) or "composer_call_failed"


def _trace_composer_retry(attempt: int, delay_seconds: float, reason: str) -> None:
    trace_event(
        "composer.transient_retry",
        attempt=attempt,
        delay_ms=round(delay_seconds * 1000, 3),
        reason=reason,
    )
