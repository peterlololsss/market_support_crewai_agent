from __future__ import annotations

from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    DistributionEvidenceScopeIdentityV1,
    UnscopedEvidenceScopeIdentityV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.planning.models import (
    DistributionExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    UnscopedExecutionDomainScopeV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import CapabilityManifestV2
from market_support_crewai_agent.runtime.policy.capabilities.evidence_vocabulary import (
    EvidenceScopeMatchFieldV2,
)


def canonical_fact_admitted_for_unit_v1(
    fact: CanonicalEvidenceFactV1,
    unit: ExecutionPlanUnitV2,
    manifest: CapabilityManifestV2,
    *,
    evaluation_epoch_seconds: int | None,
) -> bool:
    contract = manifest.evidence_contract
    if fact.fact_type not in contract.allowed_fact_types:
        return False
    if fact.fact_type in contract.forbidden_fact_types:
        return False
    if contract.provenance_required and not canonical_fact_has_provenance_v1(fact):
        return False
    if (
        contract.allowed_source_types
        and fact.source_type not in contract.allowed_source_types
    ):
        return False
    if fact.source_type in contract.forbidden_source_types:
        return False
    if (
        contract.allowed_artifact_types
        and fact.artifact_type not in contract.allowed_artifact_types
    ):
        return False
    if not canonical_scope_matches_unit_v1(fact, unit, contract.required_scope_match):
        return False
    if not canonical_history_allowed_v1(fact, manifest, evaluation_epoch_seconds):
        return False
    if not canonical_staleness_allowed_v1(fact, manifest, evaluation_epoch_seconds):
        return False
    return not contract.citation_required or bool(fact.public_urls)


def canonical_fact_has_provenance_v1(fact: CanonicalEvidenceFactV1) -> bool:
    provenance = fact.provenance
    return bool(
        provenance.source_class
        and provenance.source_record_ref
        and provenance.producer_contract_version
        and provenance.retrieval_operation
        and provenance.scope_ref
    )


def canonical_history_allowed_v1(
    fact: CanonicalEvidenceFactV1,
    manifest: CapabilityManifestV2,
    evaluation_epoch_seconds: int | None,
) -> bool:
    history_sources = {
        "conversation_history",
        "user_message",
        "assistant_message",
        "history_summary",
    }
    if fact.source_type not in history_sources:
        return True
    contract = manifest.evidence_contract
    if not contract.allow_history:
        return False
    constraints = contract.history_constraints
    role_by_source = {
        "user_message": "user",
        "assistant_message": "assistant",
    }
    role = role_by_source.get(fact.source_type)
    if role is None or role not in constraints.allowed_roles:
        return False
    if constraints.max_turns is not None:
        return False
    if constraints.max_age_seconds is None:
        return True
    if evaluation_epoch_seconds is None or fact.observed_at_epoch_seconds is None:
        return False
    return (
        evaluation_epoch_seconds - fact.observed_at_epoch_seconds
        <= constraints.max_age_seconds
    )


def canonical_staleness_allowed_v1(
    fact: CanonicalEvidenceFactV1,
    manifest: CapabilityManifestV2,
    evaluation_epoch_seconds: int | None,
) -> bool:
    policy = manifest.evidence_contract.stale_data_policy
    if policy.require_observed_at and fact.observed_at_epoch_seconds is None:
        return False
    if policy.max_age_seconds is None:
        return True
    if evaluation_epoch_seconds is None or fact.observed_at_epoch_seconds is None:
        return False
    is_stale = (
        evaluation_epoch_seconds - fact.observed_at_epoch_seconds
        > policy.max_age_seconds
    )
    return not is_stale or policy.on_stale == "allow"


def canonical_scope_matches_unit_v1(
    fact: CanonicalEvidenceFactV1,
    unit: ExecutionPlanUnitV2,
    required_scope_match: tuple[EvidenceScopeMatchFieldV2, ...],
) -> bool:
    match unit.scope, fact.scope:
        case (
            DistributionExecutionDomainScopeV2(
                business_scope_ref=business_scope_ref,
                channel_kind=channel_kind,
                product_ids=product_ids,
            ) as unit_scope,
            DistributionEvidenceScopeIdentityV1(
                business_scope_ref=fact_business_scope_ref,
                channel_kind=fact_channel_kind,
                product_ids=fact_product_ids,
            ) as fact_scope,
        ):
            full_scope_matches = (
                business_scope_ref == fact_business_scope_ref
                and channel_kind == fact_channel_kind
                and product_ids == fact_product_ids
            )
            if not full_scope_matches:
                return False
            return required_distribution_scope_matches_v1(
                unit_scope,
                fact_scope,
                unit.artifact_kind,
                required_scope_match,
            )
        case (UnscopedExecutionDomainScopeV2(), UnscopedEvidenceScopeIdentityV1()):
            return (
                not required_scope_match
                and fact.artifact_type == fact.scope.artifact_type
            )
        case _:
            return False


def required_distribution_scope_matches_v1(
    unit: DistributionExecutionDomainScopeV2,
    fact_scope: DistributionEvidenceScopeIdentityV1,
    unit_artifact_type: str,
    required_scope_match: tuple[EvidenceScopeMatchFieldV2, ...],
) -> bool:
    for field in required_scope_match:
        match field:
            case "channel_id" | "channel_kind" | "product_ids":
                continue
            case "product_id":
                if (
                    len(unit.product_ids) != 1
                    or unit.product_ids != fact_scope.product_ids
                ):
                    return False
            case "material_pack_option":
                if unit.material_pack_option != fact_scope.material_option:
                    return False
            case "time_range":
                if (
                    unit.time_range is None
                    or unit.time_range.period != fact_scope.period
                ):
                    return False
            case "artifact_type":
                if unit_artifact_type != fact_scope.artifact_type:
                    return False
    return True
