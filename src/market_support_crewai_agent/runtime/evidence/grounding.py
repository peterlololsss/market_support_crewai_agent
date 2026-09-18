from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    UnitBusinessFactsV1,
)
from market_support_crewai_agent.runtime.decisions.business_facts import (
    derive_unit_business_facts_v1,
)
from market_support_crewai_agent.runtime.evidence.admission import (
    canonical_fact_admitted_for_unit_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
    CanonicalResolveBindingV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    RegisteredMediaBindingV1,
)
from market_support_crewai_agent.runtime.evidence.requirements import (
    CanonicalGroundingContractError,
    canonical_evidence_requirements_satisfied_v1,
    validate_grounding_plan_binding_v1,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.models import (
    CanonicalActionIntentV1,
    ExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology_models import DomainContextV1
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.schemas.base import StrictModel


class _FrozenGroundingModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ExecutionUnitGroundingV1(_FrozenGroundingModel):
    contract_version: Literal["execution-unit-grounding.v1"] = (
        "execution-unit-grounding.v1"
    )
    unit_id: str = Field(min_length=1, max_length=120)
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
    evidence_query: str | None = Field(default=None, max_length=200)
    action_intents: tuple[CanonicalActionIntentV1, ...] = Field(
        default=(), max_length=3
    )
    allowed_evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)
    allowed_evidence: tuple[CanonicalEvidenceFactV1, ...] = Field(
        default=(), max_length=32
    )
    business_facts: UnitBusinessFactsV1
    guardrail_decisions: tuple[GuardrailDecision, ...] = Field(
        default=(), max_length=32
    )

    @model_validator(mode="after")
    def _validate_evidence_binding(self) -> ExecutionUnitGroundingV1:
        evidence_ids = tuple(fact.evidence_id for fact in self.allowed_evidence)
        if self.allowed_evidence_ids != evidence_ids:
            raise CanonicalGroundingContractError(
                "execution_grounding_evidence_id_binding_mismatch"
            )
        if len(set(evidence_ids)) != len(evidence_ids):
            raise CanonicalGroundingContractError(
                "execution_grounding_duplicate_evidence_id"
            )
        if self.business_facts.evidence_fact_count != len(self.allowed_evidence):
            raise CanonicalGroundingContractError(
                "execution_grounding_business_fact_count_mismatch"
            )
        return self


@dataclass(frozen=True, slots=True)
class CanonicalEvidenceExecutionResultV1:
    preflight: AdapterPreflightSnapshot
    canonical_facts: tuple[CanonicalEvidenceFactV1, ...]
    resolve_bindings: tuple[CanonicalResolveBindingV1, ...]
    groundings: tuple[ExecutionUnitGroundingV1, ...]
    domain_context: DomainContextV1
    media_bindings: tuple[RegisteredMediaBindingV1, ...] = ()


def ground_execution_plan_v2(
    plan: ExecutionPlanV2,
    policy: PolicyManifestV2,
    canonical_facts: tuple[CanonicalEvidenceFactV1, ...],
    resolve_bindings: tuple[CanonicalResolveBindingV1, ...] = (),
    *,
    evaluation_epoch_seconds: int | None = None,
) -> tuple[ExecutionUnitGroundingV1, ...]:
    if len({fact.evidence_id for fact in canonical_facts}) != len(canonical_facts):
        raise CanonicalGroundingContractError(
            "execution_grounding_duplicate_canonical_evidence_id"
        )
    groundings = tuple(
        _ground_execution_unit_v2(
            unit,
            plan,
            policy,
            canonical_facts,
            resolve_bindings,
            evaluation_epoch_seconds=evaluation_epoch_seconds,
        )
        for unit in plan.units
    )
    validate_grounding_plan_binding_v1(plan, groundings)
    return groundings


def _ground_execution_unit_v2(
    unit: ExecutionPlanUnitV2,
    plan: ExecutionPlanV2,
    policy: PolicyManifestV2,
    canonical_facts: tuple[CanonicalEvidenceFactV1, ...],
    resolve_bindings: tuple[CanonicalResolveBindingV1, ...],
    *,
    evaluation_epoch_seconds: int | None,
) -> ExecutionUnitGroundingV1:
    if unit.manifest_ref not in policy.eligible_capabilities:
        raise CanonicalGroundingContractError(
            "execution_grounding_manifest_not_policy_eligible"
        )
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(unit.manifest_ref.manifest_id)
    if manifest.manifest_version != unit.manifest_ref.manifest_version:
        raise CanonicalGroundingContractError(
            "execution_grounding_manifest_version_mismatch"
        )
    candidate_facts = tuple(
        fact
        for fact in canonical_facts
        if canonical_fact_admitted_for_unit_v1(
            fact,
            unit,
            manifest,
            evaluation_epoch_seconds=evaluation_epoch_seconds,
        )
    )
    admitted_facts = (
        candidate_facts
        if canonical_evidence_requirements_satisfied_v1(candidate_facts, manifest)
        else ()
    )
    return ExecutionUnitGroundingV1(
        unit_id=unit.unit_id,
        manifest_ref=unit.manifest_ref,
        answerability=unit.answerability_policy,
        scope=unit.scope,
        evidence_query=unit.evidence_query,
        action_intents=unit.action_intents,
        allowed_evidence_ids=tuple(fact.evidence_id for fact in admitted_facts),
        allowed_evidence=admitted_facts,
        business_facts=derive_unit_business_facts_v1(
            unit,
            manifest,
            policy,
            admitted_facts,
            tuple(
                binding
                for binding in resolve_bindings
                if binding.evidence_id in {fact.evidence_id for fact in admitted_facts}
            ),
        ),
        guardrail_decisions=plan.guardrail_decisions,
    )
