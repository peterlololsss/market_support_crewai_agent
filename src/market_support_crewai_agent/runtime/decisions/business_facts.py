from __future__ import annotations

from market_support_crewai_agent.runtime.decisions import (
    business_fact_models,
    business_fact_policy,
    business_fact_resolve,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
    CanonicalResolveBindingV1,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanUnitV2
from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestV2,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2


def derive_unit_business_facts_v1(
    unit: ExecutionPlanUnitV2,
    manifest: CapabilityManifestV2,
    effective_policy: PolicyManifestV2,
    admitted_facts: tuple[CanonicalEvidenceFactV1, ...],
    resolve_bindings: tuple[CanonicalResolveBindingV1, ...] = (),
) -> business_fact_models.UnitBusinessFactsV1:
    if len(admitted_facts) > 32:
        raise business_fact_models.BusinessFactsContractError(
            "unit_business_facts_evidence_limit_exceeded"
        )
    if unit.manifest_ref.manifest_id != manifest.manifest_id:
        raise business_fact_models.BusinessFactsContractError(
            "unit_business_facts_manifest_mismatch"
        )
    if unit.manifest_ref.manifest_version != manifest.manifest_version:
        raise business_fact_models.BusinessFactsContractError(
            "unit_business_facts_manifest_version_mismatch"
        )

    bindings = business_fact_resolve.validated_resolve_bindings_v1(
        admitted_facts, resolve_bindings
    )
    material_pack = business_fact_resolve.unit_resolvable_state_v1(
        unit, manifest, admitted_facts, bindings, "material_pack"
    )
    weekly_report = business_fact_resolve.unit_report_state_v1(
        unit, manifest, admitted_facts, bindings, "weekly_report"
    )
    monthly_report = business_fact_resolve.unit_report_state_v1(
        unit, manifest, admitted_facts, bindings, "monthly_report"
    )
    sales_mention = business_fact_resolve.unit_resolvable_state_v1(
        unit, manifest, admitted_facts, bindings, "sales_mention"
    )
    material_declared = business_fact_resolve.resolve_declared_or_admitted_v1(
        unit, manifest, "material_pack"
    )
    option_status = (
        material_pack.availability
        if material_declared and material_pack.source_record_ref is not None
        else "unknown"
    )
    return business_fact_models.UnitBusinessFactsV1(
        material_pack=material_pack,
        weekly_report=weekly_report,
        monthly_report=monthly_report,
        sales_mention=sales_mention,
        recent_executed_actions=business_fact_policy.unit_recent_actions_v1(
            unit, manifest, admitted_facts
        ),
        requested_material_pack_option_status=option_status,
        user_permission=business_fact_policy.unit_permission_v1(unit, effective_policy),
        evidence_fact_count=len(admitted_facts),
    )
