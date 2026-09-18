from __future__ import annotations

from typing import Literal

from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    AdapterEvidenceSourceRecordKeyV1,
    DistributionEvidenceScopeIdentityV1,
    EvidenceProvenanceCanonicalV1,
    UnscopedEvidenceScopeIdentityV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
    CanonicalResolveBindingV1,
    build_canonical_evidence_fact_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    EvidenceBooleanValueV1,
    ScalarEvidencePayloadCanonicalV1,
)
from market_support_crewai_agent.runtime.hashing import (
    evidence_scope_ref,
    evidence_source_record_ref,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.models import (
    DistributionExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
    UnscopedExecutionDomainScopeV2,
)
from market_support_crewai_agent.schemas.adapter import AdapterResolveResult
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType


def canonical_adapter_facts_v1(
    plan: ExecutionPlanV2,
    preflight: AdapterPreflightSnapshot,
) -> tuple[tuple[CanonicalEvidenceFactV1, ...], tuple[CanonicalResolveBindingV1, ...]]:
    results: dict[AdapterResolveType, tuple[AdapterResolveResult, str]] = {}
    for item in preflight.items:
        result = item.result
        if result is None or result.status != "resolved":
            continue
        resolve_ref = result.resolve_ref
        if resolve_ref is not None:
            results[item.resolve_type] = (result, resolve_ref)
    facts: list[CanonicalEvidenceFactV1] = []
    bindings: list[CanonicalResolveBindingV1] = []
    for unit in plan.units:
        for resolve in unit.adapter_resolves:
            resolved = results.get(resolve.resolve_type)
            if resolved is None:
                continue
            result, resolve_ref = resolved
            artifact_type = canonical_artifact_type_for_resolve(resolve.resolve_type)
            scope = canonical_scope_for_unit_v1(unit, artifact_type)
            source_record = AdapterEvidenceSourceRecordKeyV1(
                adapter_service_id="xiaoyan-adapter",
                adapter_contract_version=result.contract_version,
                resolve_type=resolve.resolve_type,
                opaque_resolve_ref=resolve_ref,
            )
            provenance = EvidenceProvenanceCanonicalV1(
                source_class="adapter",
                source_record_ref=evidence_source_record_ref(source_record),
                producer_contract_version=result.contract_version,
                retrieval_operation="adapter_resolve",
                scope_ref=evidence_scope_ref(scope),
                as_of_epoch_seconds=result.resolved_at,
                public_url_hashes=(),
            )
            fact = build_canonical_evidence_fact_v1(
                fact_type=canonical_resolvable_fact_type(resolve.resolve_type),
                source_type="adapter_resolve",
                artifact_type=artifact_type,
                resolve_type=resolve.resolve_type,
                payload=ScalarEvidencePayloadCanonicalV1(
                    value=EvidenceBooleanValueV1(value=True)
                ),
                scope=scope,
                provenance=provenance,
                observed_at_epoch_seconds=result.resolved_at,
            )
            facts.append(fact)
            bindings.append(
                CanonicalResolveBindingV1(
                    evidence_id=fact.evidence_id,
                    resolve_type=resolve.resolve_type,
                    source_record_ref=provenance.source_record_ref,
                    scope_ref=provenance.scope_ref,
                    resolve_ref=resolve_ref,
                    period=result.period,
                    report_date=result.report_date,
                )
            )
    fact_by_id = {fact.evidence_id: fact for fact in facts}
    binding_by_evidence_id = {binding.evidence_id: binding for binding in bindings}
    return tuple(fact_by_id.values()), tuple(binding_by_evidence_id.values())


def canonical_artifact_type_for_resolve(
    resolve_type: AdapterResolveType,
) -> Literal["material_pack", "weekly_report", "monthly_report", "adapter_context"]:
    match resolve_type:
        case "material_pack":
            return "material_pack"
        case "weekly_report":
            return "weekly_report"
        case "monthly_report":
            return "monthly_report"
        case "sales_mention":
            return "adapter_context"


def canonical_resolvable_fact_type(
    resolve_type: AdapterResolveType,
) -> Literal[
    "material_pack_resolvable",
    "weekly_report_resolvable",
    "monthly_report_resolvable",
    "sales_mention_resolvable",
]:
    match resolve_type:
        case "material_pack":
            return "material_pack_resolvable"
        case "weekly_report":
            return "weekly_report_resolvable"
        case "monthly_report":
            return "monthly_report_resolvable"
        case "sales_mention":
            return "sales_mention_resolvable"


def canonical_scope_for_unit_v1(
    unit: ExecutionPlanUnitV2,
    artifact_type: Literal[
        "material_pack", "weekly_report", "monthly_report", "adapter_context"
    ],
) -> DistributionEvidenceScopeIdentityV1 | UnscopedEvidenceScopeIdentityV1:
    match unit.scope:
        case DistributionExecutionDomainScopeV2(
            business_scope_ref=business_scope_ref,
            channel_kind=channel_kind,
            product_ids=product_ids,
            material_pack_option=material_pack_option,
            time_range=time_range,
        ):
            return DistributionEvidenceScopeIdentityV1(
                business_scope_ref=business_scope_ref,
                channel_kind=channel_kind,
                product_ids=product_ids,
                artifact_type=artifact_type,
                material_option=material_pack_option,
                period=time_range.period if time_range is not None else None,
            )
        case UnscopedExecutionDomainScopeV2():
            return UnscopedEvidenceScopeIdentityV1(artifact_type=artifact_type)
