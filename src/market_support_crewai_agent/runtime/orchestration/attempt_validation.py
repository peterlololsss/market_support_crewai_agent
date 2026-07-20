from __future__ import annotations

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
from market_support_crewai_agent.runtime.orchestration.answerability_directives import (
    default_answerability_for_plan,
)
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.orchestration.response_ids import (
    ensure_response_ids,
)
from market_support_crewai_agent.runtime.state.runtime_trace import trace_event
from market_support_crewai_agent.runtime.turn import AttemptResult
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
    abstention_response_text,
)
from market_support_crewai_agent.runtime.validation.output_guard import output_guard
from market_support_crewai_agent.runtime.validation.reply_validator import (
    validate_reply,
)
from market_support_crewai_agent.schemas import PrimaryReply, ReplyResponse


def validated_attempt(
    runtime,
    *,
    plan: ExecutionPlan,
    plan_validation: PlanValidationResult,
    preflight: AdapterPreflightSnapshot,
    evidence_facts: list[EvidenceFact],
    business_facts: BusinessFacts,
    domain_context: DomainContext,
    directive: ResponseDirective,
    response: ReplyResponse,
    policy: PolicyManifest,
    guardrail_decisions: list[GuardrailDecision] | None = None,
    answerability: AnswerabilityAssessment | None = None,
    composer_output: ComposerReplyOutput | None = None,
) -> AttemptResult:
    response = ensure_response_ids(response)
    guardrail_decisions = list(guardrail_decisions or [])
    output_decision = output_guard(
        response=response,
        directive=directive,
        plan=plan,
        policy=policy,
        evidence_facts=evidence_facts,
        domain_context=domain_context,
        composer_output=composer_output,
    )
    guardrail_decisions.append(output_decision)
    if output_decision.outcome == "abstain" and response.reply.kind == "answer":
        directive = ResponseDirective(
            mode="unable",
            reply_kind="unable_to_answer",
            text=abstention_response_text(),
            reason_code=output_decision.reason_code,
        )
        response = ensure_response_ids(
            ReplyResponse(
                response_id=response.response_id,
                reply=PrimaryReply(
                    kind="unable_to_answer",
                    text=directive.text,
                    mentions=[],
                ),
                actions=[],
            )
        )
    reply_validation = validate_reply(
        response,
        directive,
        plan,
        business_facts,
        evidence_facts,
        policy,
        domain_context=domain_context,
        composer_output=composer_output,
        output_decision=output_decision,
    )
    trace_event(
        "state.reply_validated",
        valid=reply_validation.valid,
        issue_count=len(reply_validation.issues),
        error=reply_validation_error_summary(reply_validation)
        if not reply_validation.valid
        else "",
        output_guard_outcome=output_decision.outcome,
    )
    return AttemptResult(
        plan=plan,
        plan_validation=plan_validation,
        preflight=preflight,
        evidence_facts=evidence_facts,
        business_facts=business_facts,
        domain_context=domain_context,
        answerability=answerability or default_answerability_for_plan(plan),
        directive=directive,
        response=response,
        reply_validation=reply_validation,
        guardrail_decisions=guardrail_decisions,
        composer_output=composer_output,
    )


def _validation_error_summary(validation: PlanValidationResult) -> str:
    return "; ".join(issue.code for issue in validation.issues) or "unknown"


def skip_alignment_verifier(candidate: AttemptResult) -> bool:
    return any(
        decision.reason_code
        in {
            "direct_send_command_matched",
            "direct_message_composer",
            "material_pack_option_confirmation_required",
        }
        for decision in candidate.plan.guardrail_decisions
    )


def reply_validation_error_summary(validation) -> str:
    parts = []
    for issue in validation.issues:
        detail_values = [
            issue.metadata.get("contract_issue_code"),
            issue.metadata.get("unit_id"),
            issue.metadata.get("selected_capability_id"),
        ]
        detail = (
            ",".join(str(value) for value in detail_values if value) or issue.message
        )
        parts.append(f"{issue.code}({detail})")
    return "; ".join(dict.fromkeys(parts)) or "unknown"
