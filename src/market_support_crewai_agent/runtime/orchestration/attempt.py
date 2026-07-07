from __future__ import annotations

from pydantic import JsonValue
from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
    plan_spec_for_execution_plan,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.evidence.models import EvidenceFact
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.prompting.profiles import ModelFamily
from market_support_crewai_agent.runtime.llm.prompting.context import IntentGateResult
from market_support_crewai_agent.runtime.orchestration.answerability_directives import (
    directive_from_answerability,
)
from market_support_crewai_agent.runtime.orchestration.decision import (
    DecisionEngine,
)
from market_support_crewai_agent.runtime.orchestration.guardrail_summary import (
    guardrail_decisions_from_directive,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.state.runtime_trace import (
    trace_event,
    trace_span,
)
from market_support_crewai_agent.runtime.orchestration.planner import (
    validation_error_summary,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError, AttemptResult
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityGate,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import ReplyRequest


async def build_candidate_from_plan(
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
    alignment_verdict: ReplyAlignmentVerdict | None = None,
    alignment_attempt: int = 0,
) -> AttemptResult:
    if plan.plan_spec is None:
        plan = plan.model_copy(
            update={
                "plan_spec": plan_spec_for_execution_plan(
                    plan,
                    domain_context=domain_context,
                )
            }
        )
    with trace_span("plan.validate"):
        plan_validation = validate_execution_plan(plan, policy)
    if not plan_validation.valid:
        raise AgentRuntimeError(
            "compiled execution plan failed validation: {}".format(
                validation_error_summary(plan_validation)
            )
        )

    with trace_span("evidence.execute"):
        evidence_result = await runtime.evidence_executor.execute(
            request,
            plan,
            policy,
            action_history=action_history,
        )
    trace_event(
        "state.evidence_collected",
        preflight_items=len(evidence_result.preflight.items),
        evidence_fact_count=len(evidence_result.evidence_facts),
        guardrail_decision_count=len(
            getattr(evidence_result, "guardrail_decisions", [])
        ),
    )
    return await runtime._build_candidate_from_evidence(
        request=request,
        domain_context=getattr(evidence_result, "domain_context", domain_context),
        policy=policy,
        model_family=model_family,
        intent_gate=intent_gate,
        history=history,
        action_history=action_history,
        prompt_programs=prompt_programs,
        llm_executions=llm_executions,
        plan=plan,
        plan_validation=plan_validation,
        preflight=evidence_result.preflight,
        evidence_facts=evidence_result.evidence_facts,
        business_facts=evidence_result.business_facts,
        guardrail_decisions=[
            *plan.guardrail_decisions,
            *getattr(evidence_result, "guardrail_decisions", []),
        ],
        alignment_verdict=alignment_verdict,
        alignment_attempt=alignment_attempt,
    )


async def build_candidate_from_evidence(
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
    guardrail_decisions: list[GuardrailDecision],
    alignment_verdict: ReplyAlignmentVerdict | None = None,
    alignment_attempt: int = 0,
) -> AttemptResult:
    with trace_span("answerability.assess"):
        answerability = AnswerabilityGate().assess(
            request=request,
            domain_context=domain_context,
            plan=plan,
            policy=policy,
            evidence_facts=evidence_facts,
        )
    with trace_span("decision.decide"):
        directive = DecisionEngine().decide(
            plan,
            business_facts,
            evidence_facts,
            request,
            policy,
            domain_context,
        )
    forced_directive = directive_from_answerability(answerability, plan)
    if forced_directive is not None:
        directive = forced_directive
    if (
        directive.mode == "clarification"
        and answerability.recommended_response_mode != "clarify"
    ):
        answerability = answerability.model_copy(
            update={
                "can_answer": False,
                "ambiguity": "other",
                "recommended_response_mode": "clarify",
                "user_facing_reason": directive.text
                or answerability.user_facing_reason,
            }
        )
    guardrail_decisions = [
        *guardrail_decisions,
        *guardrail_decisions_from_directive(directive, plan, business_facts),
    ]
    trace_event(
        "state.directive_selected",
        mode=directive.mode,
        reply_kind=directive.reply_kind,
        requires_knowledge_composer=directive.requires_knowledge_composer,
        composer_stage=directive.composer_stage,
    )
    response, composer_output = await runtime._compose_or_render_response(
        request=request,
        domain_context=domain_context,
        policy=policy,
        model_family=model_family,
        intent_gate=intent_gate,
        history=history,
        action_history=action_history,
        prompt_programs=prompt_programs,
        llm_executions=llm_executions,
        plan=plan,
        plan_validation=plan_validation,
        preflight=preflight,
        evidence_facts=evidence_facts,
        business_facts=business_facts,
        answerability=answerability,
        directive=directive,
        alignment_verdict=alignment_verdict,
        alignment_attempt=alignment_attempt,
        guardrail_decisions=guardrail_decisions,
    )
    with trace_span("reply.validate_attempt"):
        return runtime._validated_attempt(
            plan=plan,
            plan_validation=plan_validation,
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            domain_context=domain_context,
            answerability=answerability,
            directive=directive,
            response=response,
            policy=policy,
            guardrail_decisions=guardrail_decisions,
            composer_output=composer_output,
        )
