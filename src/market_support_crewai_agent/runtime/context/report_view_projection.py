from __future__ import annotations

from typing import assert_never

from market_support_crewai_agent.runtime.context.report_view_models import (
    ReportScopeMatchViewV1,
    ReportScopeProductsViewV1,
    ReportScopeProductViewV1,
    ReportScopeSectionViewV1,
    ReportScopeSummaryViewV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    CanonicalEvidenceReportPayloadV1,
    ReportScopeMatchCanonicalV1,
    ReportScopeProductCanonicalV1,
    ReportScopeProductsCanonicalV1,
    ReportScopeSectionCanonicalV1,
    ReportScopeSummaryCanonicalV1,
)


def project_report_payload_view_v1(
    payload: CanonicalEvidenceReportPayloadV1 | None,
) -> (
    ReportScopeSummaryViewV1 | ReportScopeMatchViewV1 | ReportScopeProductsViewV1 | None
):
    match payload:
        case None:
            return None
        case ReportScopeSummaryCanonicalV1():
            return ReportScopeSummaryViewV1(
                material_type=payload.material_type,
                period=payload.period,
                report_date=payload.report_date,
                period_start=payload.period_start,
                period_end=payload.period_end,
                period_label=payload.period_label,
                available=payload.available,
                reason_code=payload.reason_code,
                scope_complete=payload.scope_complete,
                expected_product_count=payload.expected_product_count,
                generated_product_count=payload.generated_product_count,
                missing_product_count=payload.missing_product_count,
                sections=tuple(
                    _project_report_section(section) for section in payload.sections
                ),
            )
        case ReportScopeMatchCanonicalV1():
            return ReportScopeMatchViewV1(
                status=payload.status,
                query=payload.query,
                match_type=payload.match_type,
                section=payload.section,
                candidate_count=payload.candidate_count,
                products=tuple(
                    _project_report_product(product) for product in payload.products
                ),
                page=payload.page,
                page_size=payload.page_size,
            )
        case ReportScopeProductsCanonicalV1():
            return ReportScopeProductsViewV1(
                available=payload.available,
                reason_code=payload.reason_code,
                period=payload.period,
                report_date=payload.report_date,
                products=tuple(
                    _project_report_product(product) for product in payload.products
                ),
                page=payload.page,
                page_size=payload.page_size,
                total_count=payload.total_count,
                returned_count=payload.returned_count,
                products_are_paginated=payload.products_are_paginated,
                full_product_list_in_projection=(
                    payload.full_product_list_in_projection
                ),
            )
        case unreachable:
            assert_never(unreachable)


def _project_report_section(
    section: ReportScopeSectionCanonicalV1,
) -> ReportScopeSectionViewV1:
    return ReportScopeSectionViewV1(
        name=section.name,
        source_pdf_count=section.source_pdf_count,
        final_report_count=section.final_report_count,
        missing_product_count=section.missing_product_count,
    )


def _project_report_product(
    product: ReportScopeProductCanonicalV1,
) -> ReportScopeProductViewV1:
    return ReportScopeProductViewV1(
        product_name=product.product_name,
        portfolio_type=product.portfolio_type,
        report_section=product.report_section,
        source_pdf_status=product.source_pdf_status,
        final_report_status=product.final_report_status,
    )
