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
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.state.audit import build_audit_trace
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse


def record_audit_trace(
    runtime,
    *,
    request: ReplyRequest,
    policy: PolicyManifest,
    plan: ExecutionPlan,
    directive: ResponseDirective,
    plan_validation: PlanValidationResult,
    action_history: list[ActionLedgerRecord],
    domain_context: DomainContext,
    preflight: AdapterPreflightSnapshot,
    evidence_facts: list[EvidenceFact],
    business_facts: BusinessFacts,
    response: ReplyResponse,
    reply_validation,
    answerability_assessment: AnswerabilityAssessment | None = None,
    guardrail_decisions: list[GuardrailDecision] | None = None,
    intent_gate=None,
    prompt_programs: list[PromptProgram] | None = None,
    llm_executions: list[dict[str, JsonValue]] | None = None,
    alignment_verdicts: list[ReplyAlignmentVerdict] | None = None,
    alignment_remediations: list[dict[str, JsonValue]] | None = None,
    runtime_trace: dict[str, JsonValue] | None = None,
) -> None:
    runtime.audit_store.record(
        build_audit_trace(
            request=request,
            settings=runtime.settings,
            policy=policy,
            plan=plan,
            directive=directive,
            plan_validation=plan_validation,
            action_history=action_history,
            domain_context=domain_context,
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            answerability_assessment=answerability_assessment,
            response=response,
            reply_validation=reply_validation,
            guardrail_decisions=guardrail_decisions or [],
            intent_gate=intent_gate,
            prompt_programs=prompt_programs,
            llm_executions=llm_executions,
            alignment_verdicts=alignment_verdicts,
            alignment_remediations=alignment_remediations,
            runtime_trace=runtime_trace,
        )
    )
