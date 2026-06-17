from __future__ import annotations

from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    evidence_required_for_plan,
    retrieval_source_guard,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import evidence_id
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
    make_decision,
)
from market_support_crewai_agent.schemas import ReplyResponse


def output_guard(
    *,
    response: ReplyResponse,
    directive: object,
    plan: object,
    policy: PolicyManifest,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
    composer_output: ComposerReplyOutput | None = None,
) -> GuardrailDecision:
    mode = str(getattr(directive, "mode", "") or "")
    composer_stage = str(getattr(directive, "composer_stage", "") or "")
    if mode != "knowledge_answer" and composer_stage != "knowledge_composer":
        return make_decision("allow", "output", "output_guard_not_applicable")

    if response.reply.kind != "answer" or not response.reply.text.strip():
        return make_decision("allow", "output", "abstention_or_empty_answer_allowed")

    source_decision = retrieval_source_guard(
        plan=plan,
        policy=policy,
        evidence_facts=evidence_facts,
        domain_context=domain_context,
    )
    composer_evidence_decision = composer_evidence_use_decision(
        composer_output,
        plan,
        source_decision,
    )
    if composer_evidence_decision is not None:
        return composer_evidence_decision

    if source_decision.outcome != "allow":
        return source_decision.model_copy(update={"phase": "output"})

    return make_decision(
        "allow",
        "output",
        "output_claims_supported",
        evidence_required=evidence_required_for_plan(plan),
        evidence_seen=[evidence_id(fact) for fact in evidence_facts],
    )


def composer_evidence_use_decision(
    composer_output: ComposerReplyOutput | None,
    plan: object,
    source_decision: GuardrailDecision,
) -> GuardrailDecision | None:
    if composer_output is None or composer_output.response_mode != "answer":
        return None

    evidence_required = evidence_required_for_plan(plan)
    if evidence_required and not composer_output.evidence_ids:
        return make_decision(
            "abstain",
            "output",
            "composer_evidence_ids_missing",
            human_reason="Composer answer did not cite any allowed evidence IDs.",
            evidence_required=evidence_required,
            evidence_seen=source_decision.evidence_seen,
            source_scopes=source_decision.source_scopes,
            metadata={"claims": list(composer_output.claims)},
        )

    allowed = (
        set(source_decision.evidence_seen)
        if source_decision.outcome == "allow"
        else set()
    )
    invalid = [
        evidence_id_value
        for evidence_id_value in composer_output.evidence_ids
        if evidence_id_value not in allowed
    ]
    if invalid:
        return make_decision(
            "abstain",
            "output",
            "composer_evidence_id_not_allowed",
            human_reason="Composer answer cited evidence outside the allowed source scope.",
            evidence_required=evidence_required,
            evidence_seen=source_decision.evidence_seen,
            source_scopes=source_decision.source_scopes,
            metadata={
                "invalid_evidence_ids": invalid,
                "allowed_evidence_ids": sorted(allowed),
                "claims": list(composer_output.claims),
            },
        )
    return None
