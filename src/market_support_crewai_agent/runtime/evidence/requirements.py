from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, override

from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.planning.models import (
    CanonicalActionIntentV1,
    ExecutionDomainScopeV2,
    ExecutionPlanV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CapabilityManifestV2,
    ManifestRefV1,
)


@dataclass(frozen=True, slots=True)
class CanonicalGroundingContractError(ValueError):
    reason_code: str

    @override
    def __str__(self) -> str:
        return self.reason_code


class GroundingPlanBindingV1(Protocol):
    unit_id: str
    manifest_ref: ManifestRefV1
    answerability: Literal[
        "answer",
        "send",
        "clarify",
        "abstain",
        "refuse",
        "handoff",
        "smalltalk",
        "no_reply",
    ]
    scope: ExecutionDomainScopeV2
    evidence_query: str | None
    action_intents: tuple[CanonicalActionIntentV1, ...]


def canonical_evidence_requirements_satisfied_v1(
    facts: tuple[CanonicalEvidenceFactV1, ...],
    manifest: CapabilityManifestV2,
) -> bool:
    contract = manifest.evidence_contract
    admitted_fact_types = {fact.fact_type for fact in facts}
    admitted_artifact_types = {fact.artifact_type for fact in facts}
    if not set(contract.required_fact_types) <= admitted_fact_types:
        return False
    if contract.any_of_fact_types and not (
        set(contract.any_of_fact_types) & admitted_fact_types
    ):
        return False
    if not set(contract.required_artifact_types) <= admitted_artifact_types:
        return False
    return len(facts) >= contract.min_facts


def validate_grounding_plan_binding_v1(
    plan: ExecutionPlanV2,
    groundings: tuple[GroundingPlanBindingV1, ...],
) -> None:
    if len(plan.units) != len(groundings):
        raise CanonicalGroundingContractError("execution_grounding_unit_count_mismatch")
    for unit, grounding in zip(plan.units, groundings, strict=True):
        if (
            grounding.unit_id != unit.unit_id
            or grounding.manifest_ref != unit.manifest_ref
            or grounding.answerability != unit.answerability_policy
            or grounding.scope != unit.scope
            or grounding.evidence_query != unit.evidence_query
            or grounding.action_intents != unit.action_intents
        ):
            raise CanonicalGroundingContractError(
                "execution_grounding_plan_binding_mismatch"
            )
