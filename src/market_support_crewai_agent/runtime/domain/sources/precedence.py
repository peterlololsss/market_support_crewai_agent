from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities import CapabilityName
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.evidence.models import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    EvidenceSelection,
)
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    select_evidence_for_plan,
)


def select_evidence_for_capabilities(
    capabilities: list[CapabilityName] | tuple[CapabilityName, ...],
    evidence_facts: list[EvidenceFact],
) -> EvidenceSelection:
    plan = ExecutionPlan(
        user_need="select evidence",
        artifact_kind="knowledge_answer",
        response_mode="knowledge_answer",
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "evidence selection",
        },
        capabilities=list(capabilities),
        answer_capabilities=list(capabilities),
    )
    return select_evidence_for_plan(
        plan=plan,
        policy=compile_policy(None),
        evidence_facts=evidence_facts,
    )


def evidence_facts_for_plan(
    plan: ExecutionPlan,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
) -> list[EvidenceFact]:
    if not plan.answer_capabilities:
        return []
    return list(
        select_evidence_for_plan(
            plan=plan,
            policy=compile_policy(None),
            evidence_facts=evidence_facts,
            domain_context=domain_context,
        )
        .accepted
    )


def plan_has_knowledge_evidence(
    plan: ExecutionPlan,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
) -> bool:
    return bool(evidence_facts_for_plan(plan, evidence_facts, domain_context))
