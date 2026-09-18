from __future__ import annotations

from typing import Literal

from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    AvailabilityStatus,
    BusinessFactsContractError,
    CanonicalReportSectionV1,
    CanonicalReportStateV1,
    CanonicalResolvableStateV1,
    strict_iso_date_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
    CanonicalResolveBindingV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    EvidenceBooleanValueV1,
    ReportScopeSummaryCanonicalV1,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanUnitV2
from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestV2,
)
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType

_RESOLVE_FACT_TYPES_V1: dict[AdapterResolveType, str] = {
    "material_pack": "material_pack_resolvable",
    "weekly_report": "weekly_report_resolvable",
    "monthly_report": "monthly_report_resolvable",
    "sales_mention": "sales_mention_resolvable",
}


def validated_resolve_bindings_v1(
    facts: tuple[CanonicalEvidenceFactV1, ...],
    bindings: tuple[CanonicalResolveBindingV1, ...],
) -> dict[str, CanonicalResolveBindingV1]:
    fact_by_id = {fact.evidence_id: fact for fact in facts}
    output: dict[str, CanonicalResolveBindingV1] = {}
    for binding in bindings:
        fact = fact_by_id.get(binding.evidence_id)
        if fact is None:
            raise BusinessFactsContractError(
                "canonical_resolve_binding_evidence_not_admitted"
            )
        if binding.evidence_id in output:
            raise BusinessFactsContractError(
                "canonical_resolve_binding_duplicate_evidence"
            )
        if (
            binding.resolve_type != fact.resolve_type
            or binding.source_record_ref != fact.provenance.source_record_ref
            or binding.scope_ref != fact.provenance.scope_ref
        ):
            raise BusinessFactsContractError(
                "canonical_resolve_binding_provenance_mismatch"
            )
        output[binding.evidence_id] = binding
    return output


def unit_resolvable_state_v1(
    unit: ExecutionPlanUnitV2,
    manifest: CapabilityManifestV2,
    facts: tuple[CanonicalEvidenceFactV1, ...],
    bindings: dict[str, CanonicalResolveBindingV1],
    resolve_type: AdapterResolveType,
) -> CanonicalResolvableStateV1:
    if not resolve_declared_or_admitted_v1(unit, manifest, resolve_type):
        return CanonicalResolvableStateV1()
    fact = _first_resolvable_fact_v1(facts, resolve_type)
    if fact is None:
        return CanonicalResolvableStateV1()
    binding = bindings.get(fact.evidence_id)
    return CanonicalResolvableStateV1(
        availability=_availability_from_canonical_fact_v1(fact),
        reason_code="",
        material_pack_option=(
            fact.scope.material_option if resolve_type == "material_pack" else None
        ),
        source_record_ref=fact.provenance.source_record_ref,
        resolve_ref=(
            binding.resolve_ref
            if binding is not None and binding.resolve_type == resolve_type
            else None
        ),
    )


def unit_report_state_v1(
    unit: ExecutionPlanUnitV2,
    manifest: CapabilityManifestV2,
    facts: tuple[CanonicalEvidenceFactV1, ...],
    bindings: dict[str, CanonicalResolveBindingV1],
    resolve_type: Literal["weekly_report", "monthly_report"],
) -> CanonicalReportStateV1:
    base = unit_resolvable_state_v1(unit, manifest, facts, bindings, resolve_type)
    if not resolve_declared_or_admitted_v1(unit, manifest, resolve_type):
        return CanonicalReportStateV1()
    payload = _first_report_summary_v1(facts, resolve_type)
    if payload is None:
        binding = _binding_for_resolve_fact_v1(facts, bindings, resolve_type)
        return CanonicalReportStateV1(
            availability=base.availability,
            candidate_labels=base.candidate_labels,
            reason_code=base.reason_code,
            resolve_ref=base.resolve_ref,
            material_pack_option=base.material_pack_option,
            source_record_ref=base.source_record_ref,
            period=binding.period if binding is not None else None,
            report_date=(
                strict_iso_date_v1(binding.report_date) if binding is not None else None
            ),
        )
    return CanonicalReportStateV1(
        availability="available" if payload.available else base.availability,
        candidate_labels=base.candidate_labels,
        reason_code=payload.reason_code,
        resolve_ref=base.resolve_ref,
        material_pack_option=None,
        source_record_ref=base.source_record_ref,
        period=payload.period,
        report_date=strict_iso_date_v1(payload.report_date),
        period_start=strict_iso_date_v1(payload.period_start),
        period_end=strict_iso_date_v1(payload.period_end),
        period_label=payload.period_label,
        scope_complete=payload.scope_complete,
        expected_product_count=payload.expected_product_count,
        generated_product_count=payload.generated_product_count,
        missing_product_count=payload.missing_product_count,
        report_sections=tuple(
            CanonicalReportSectionV1(
                name=section.name,
                source_pdf_count=section.source_pdf_count,
                final_report_count=section.final_report_count,
                missing_product_count=section.missing_product_count,
            )
            for section in payload.sections
        ),
    )


def resolve_declared_or_admitted_v1(
    unit: ExecutionPlanUnitV2,
    manifest: CapabilityManifestV2,
    resolve_type: AdapterResolveType,
) -> bool:
    if any(resolve.resolve_type == resolve_type for resolve in unit.adapter_resolves):
        return True
    fact_type = _RESOLVE_FACT_TYPES_V1[resolve_type]
    contract = manifest.evidence_contract
    return (
        fact_type in contract.required_fact_types
        or fact_type in contract.any_of_fact_types
    )


def _first_resolvable_fact_v1(
    facts: tuple[CanonicalEvidenceFactV1, ...],
    resolve_type: AdapterResolveType,
) -> CanonicalEvidenceFactV1 | None:
    expected_fact_type = _RESOLVE_FACT_TYPES_V1[resolve_type]
    return next(
        (
            fact
            for fact in facts
            if fact.fact_type == expected_fact_type
            and fact.resolve_type == resolve_type
        ),
        None,
    )


def _binding_for_resolve_fact_v1(
    facts: tuple[CanonicalEvidenceFactV1, ...],
    bindings: dict[str, CanonicalResolveBindingV1],
    resolve_type: Literal["weekly_report", "monthly_report"],
) -> CanonicalResolveBindingV1 | None:
    fact = _first_resolvable_fact_v1(facts, resolve_type)
    if fact is None:
        return None
    binding = bindings.get(fact.evidence_id)
    if binding is None or binding.resolve_type != resolve_type:
        return None
    return binding


def _availability_from_canonical_fact_v1(
    fact: CanonicalEvidenceFactV1,
) -> AvailabilityStatus:
    if isinstance(fact.value, EvidenceBooleanValueV1):
        return "available" if fact.value.value else "unavailable"
    return "unknown"


def _first_report_summary_v1(
    facts: tuple[CanonicalEvidenceFactV1, ...],
    resolve_type: Literal["weekly_report", "monthly_report"],
) -> ReportScopeSummaryCanonicalV1 | None:
    material_type: Literal["weekly", "monthly"] = (
        "weekly" if resolve_type == "weekly_report" else "monthly"
    )
    return next(
        (
            fact.report_payload
            for fact in facts
            if fact.fact_type == "report_scope_summary"
            and fact.resolve_type == resolve_type
            and isinstance(fact.report_payload, ReportScopeSummaryCanonicalV1)
            and fact.report_payload.material_type == material_type
        ),
        None,
    )
