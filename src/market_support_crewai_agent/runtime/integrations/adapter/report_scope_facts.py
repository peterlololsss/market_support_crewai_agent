from __future__ import annotations

from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    AdapterEvidenceSourceRecordKeyV1,
    DistributionEvidenceScopeIdentityV1,
    EvidenceProvenanceCanonicalV1,
    UnscopedEvidenceScopeIdentityV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
    build_canonical_evidence_fact_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    EvidenceNullValueV1,
    EvidenceStringValueV1,
    ReportMatchEvidencePayloadCanonicalV1,
    ReportProductsEvidencePayloadCanonicalV1,
    ReportScopeMatchCanonicalV1,
    ReportScopeProductCanonicalV1,
    ReportScopeProductsCanonicalV1,
    ReportScopeSectionCanonicalV1,
    ReportScopeSummaryCanonicalV1,
    ReportSummaryEvidencePayloadCanonicalV1,
    ScalarEvidencePayloadCanonicalV1,
)
from market_support_crewai_agent.runtime.hashing import (
    evidence_scope_ref,
    evidence_source_record_ref,
)
from market_support_crewai_agent.runtime.integrations.adapter.report_scope_targets import (
    PAGE_SIZE,
    REPORT_SCOPE_CONTRACT_VERSION,
    ReportCommand,
    ReportTarget,
)
from market_support_crewai_agent.runtime.planning.models import (
    DistributionExecutionDomainScopeV2,
    UnscopedExecutionDomainScopeV2,
)
from market_support_crewai_agent.schemas.adapter import (
    AdapterReportScopeResult,
    ReportScopeProduct,
)


def summary_fact(
    target: ReportTarget,
    result: AdapterReportScopeResult,
) -> CanonicalEvidenceFactV1:
    scope = _fact_scope(target, result)
    payload = ReportScopeSummaryCanonicalV1(
        material_type=result.material_type,
        period=result.period,
        report_date=result.report_date,
        period_start=result.period_start,
        period_end=result.period_end,
        period_label=result.period_label,
        available=result.status == "resolved",
        reason_code=result.reason_code,
        scope_complete=result.scope_complete,
        expected_product_count=result.expected_product_count,
        generated_product_count=result.generated_product_count,
        missing_product_count=result.missing_product_count,
        sections=tuple(
            ReportScopeSectionCanonicalV1(
                name=section.name,
                source_pdf_count=section.expected_product_count,
                final_report_count=section.generated_product_count,
                missing_product_count=section.missing_product_count,
            )
            for section in result.report_sections
        ),
    )
    return build_canonical_evidence_fact_v1(
        fact_type="report_scope_summary",
        source_type="adapter_report_scope",
        artifact_type=target.resolve_type,
        resolve_type=target.resolve_type,
        payload=ReportSummaryEvidencePayloadCanonicalV1(
            value=EvidenceNullValueV1(),
            report_payload=payload,
        ),
        scope=scope,
        provenance=_provenance(target, result.contract_version, "summary", scope),
        observed_at_epoch_seconds=_observed_at(target),
    )


def match_fact(
    target: ReportTarget,
    result: AdapterReportScopeResult,
) -> CanonicalEvidenceFactV1:
    scope = _fact_scope(target, result)
    match_result = result.match
    payload = ReportScopeMatchCanonicalV1(
        status=match_result.status if match_result is not None else "not_found",
        query=(match_result.query if match_result is not None else target.query)
        or "unknown",
        match_type=match_result.match_type if match_result is not None else None,
        section=match_result.matched_section if match_result is not None else None,
        candidate_count=match_result.candidate_count if match_result is not None else 0,
        products=tuple(
            _canonical_product(product)
            for product in (match_result.products if match_result is not None else ())
        ),
        page=match_result.product_page if match_result is not None else 1,
        page_size=match_result.product_page_size if match_result is not None else 10,
    )
    return build_canonical_evidence_fact_v1(
        fact_type="report_scope_match",
        source_type="adapter_report_scope",
        artifact_type=target.resolve_type,
        resolve_type=target.resolve_type,
        payload=ReportMatchEvidencePayloadCanonicalV1(
            value=EvidenceNullValueV1(),
            report_payload=payload,
        ),
        scope=scope,
        provenance=_provenance(target, result.contract_version, "match", scope),
        observed_at_epoch_seconds=_observed_at(target),
    )


def products_fact(
    target: ReportTarget,
    result: AdapterReportScopeResult,
    products: list[ReportScopeProduct],
    total_count: int,
) -> CanonicalEvidenceFactV1:
    scope = _fact_scope(target, result)
    payload = ReportScopeProductsCanonicalV1(
        available=True,
        reason_code=result.reason_code,
        period=result.period,
        report_date=result.report_date,
        products=tuple(_canonical_product(product) for product in products),
        page=1,
        page_size=PAGE_SIZE,
        total_count=total_count,
        returned_count=len(products),
        full_product_list_in_projection=total_count <= len(products),
    )
    return build_canonical_evidence_fact_v1(
        fact_type="report_scope_products",
        source_type="adapter_report_scope",
        artifact_type=target.resolve_type,
        resolve_type=target.resolve_type,
        payload=ReportProductsEvidencePayloadCanonicalV1(
            value=EvidenceNullValueV1(),
            report_payload=payload,
        ),
        scope=scope,
        provenance=_provenance(target, result.contract_version, "list_products", scope),
        observed_at_epoch_seconds=_observed_at(target),
    )


def unavailable_fact(target: ReportTarget, reason_code: str) -> CanonicalEvidenceFactV1:
    scope = _fact_scope(target, None)
    return build_canonical_evidence_fact_v1(
        fact_type="report_scope_unavailable",
        source_type="adapter_report_scope",
        artifact_type=target.resolve_type,
        resolve_type=target.resolve_type,
        payload=ScalarEvidencePayloadCanonicalV1(
            value=EvidenceStringValueV1(value=reason_code)
        ),
        scope=scope,
        provenance=_provenance(
            target,
            REPORT_SCOPE_CONTRACT_VERSION,
            target.command,
            scope,
        ),
        observed_at_epoch_seconds=_observed_at(target),
    )


def _canonical_product(product: ReportScopeProduct) -> ReportScopeProductCanonicalV1:
    return ReportScopeProductCanonicalV1(
        product_name=product.product_name,
        portfolio_type=product.portfolio_type,
        report_section=product.report_section,
        source_pdf_status=product.source_pdf_status,
        final_report_status=product.final_report_status,
    )


def _fact_scope(
    target: ReportTarget,
    result: AdapterReportScopeResult | None,
) -> DistributionEvidenceScopeIdentityV1 | UnscopedEvidenceScopeIdentityV1:
    period = result.period if result is not None else target.period
    report_date = result.report_date if result is not None else None
    match target.unit.scope:
        case DistributionExecutionDomainScopeV2(
            business_scope_ref=business_scope_ref,
            channel_kind=channel_kind,
            product_ids=product_ids,
        ):
            return DistributionEvidenceScopeIdentityV1(
                business_scope_ref=business_scope_ref,
                channel_kind=channel_kind,
                product_ids=product_ids,
                artifact_type=target.resolve_type,
                period=period,
                report_date=report_date,
            )
        case UnscopedExecutionDomainScopeV2():
            return UnscopedEvidenceScopeIdentityV1(
                artifact_type=target.resolve_type,
                period=period,
                report_date=report_date,
            )


def _provenance(
    target: ReportTarget,
    contract_version: str,
    operation: ReportCommand,
    scope: DistributionEvidenceScopeIdentityV1 | UnscopedEvidenceScopeIdentityV1,
) -> EvidenceProvenanceCanonicalV1:
    result = target.preflight_result
    resolve_ref = (
        result.resolve_ref
        if result is not None and result.resolve_ref is not None
        else f"report-scope:{target.resolve_type}:unresolved"
    )
    source_record = AdapterEvidenceSourceRecordKeyV1(
        adapter_service_id="assistant-adapter",
        adapter_contract_version=contract_version,
        resolve_type=target.resolve_type,
        opaque_resolve_ref=resolve_ref,
    )
    return EvidenceProvenanceCanonicalV1(
        source_class="adapter",
        source_record_ref=evidence_source_record_ref(source_record),
        producer_contract_version=contract_version,
        retrieval_operation=f"adapter_report_scope.{operation}",
        scope_ref=evidence_scope_ref(scope),
        as_of_epoch_seconds=_observed_at(target),
        public_url_hashes=(),
    )


def _observed_at(target: ReportTarget) -> int | None:
    return (
        target.preflight_result.resolved_at
        if target.preflight_result is not None
        else None
    )
