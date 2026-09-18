from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.context import models as context_models


def _product(index: int) -> context_models.ReportScopeProductViewV1:
    return context_models.ReportScopeProductViewV1(
        product_name=f"Product {index}",
        portfolio_type="market-neutral",
        report_section="Section A",
        source_pdf_status="found",
        final_report_status="generated",
    )


def _section(index: int) -> context_models.ReportScopeSectionViewV1:
    return context_models.ReportScopeSectionViewV1(
        name=f"Section {index}",
        source_pdf_count=1,
        final_report_count=1,
        missing_product_count=0,
    )


def test_report_summary_preserves_sixty_four_ordered_sections() -> None:
    # Given: one canonical-size report summary containing the maximum sections.
    sections = tuple(_section(index) for index in range(64))

    # When: the bounded summary view is constructed.
    view = context_models.ReportScopeSummaryViewV1(
        material_type="weekly",
        period="2026-W28",
        report_date="2026-07-17",
        period_start="2026-07-13",
        period_end="2026-07-17",
        period_label="Week 28",
        available=True,
        reason_code="ok",
        scope_complete=True,
        expected_product_count=64,
        generated_product_count=64,
        missing_product_count=0,
        sections=sections,
    )

    # Then: section order and all bounded count facts are retained exactly.
    assert view.sections == sections
    assert view.generated_product_count == 64


def test_report_summary_rejects_sixty_five_sections() -> None:
    # Given: a report summary exceeding the explicit 64-section boundary.
    sections = tuple(_section(index) for index in range(65))

    # When/Then: strict construction rejects instead of clipping the list.
    with pytest.raises(ValidationError):
        context_models.ReportScopeSummaryViewV1(
            material_type="weekly",
            available=True,
            reason_code="ok",
            sections=sections,
        )


def test_report_match_accepts_fifty_products_and_rejects_fifty_one() -> None:
    # Given: exact-boundary and over-boundary report match product tuples.
    products = tuple(_product(index) for index in range(50))

    # When: the exact-boundary match is constructed.
    view = context_models.ReportScopeMatchViewV1(
        status="ambiguous",
        query="Product",
        candidate_count=50,
        products=products,
        page=1,
        page_size=50,
    )

    # Then: all 50 survive, while a 51st product is rejected without clipping.
    assert view.products == products
    with pytest.raises(ValidationError):
        context_models.ReportScopeMatchViewV1(
            status="ambiguous",
            query="Product",
            candidate_count=51,
            products=(*products, _product(50)),
            page=1,
            page_size=50,
        )


def test_report_products_marks_two_hundred_of_two_hundred_one_incomplete() -> None:
    # Given: the projection ceiling returns 200 of 201 adapter-ordered products.
    products = tuple(_product(index) for index in range(200))

    # When: the bounded product-list view records the incomplete result.
    view = context_models.ReportScopeProductsViewV1(
        available=True,
        reason_code="ok",
        period="2026-W28",
        report_date="2026-07-17",
        products=products,
        page=1,
        page_size=50,
        total_count=201,
        returned_count=200,
        products_are_paginated=True,
        full_product_list_in_projection=False,
    )

    # Then: the explicit completeness flag remains false.
    assert view.full_product_list_in_projection is False


def test_report_products_rejects_false_completeness_claim() -> None:
    # Given: only 200 of 201 products are present in the view.
    products = tuple(_product(index) for index in range(200))

    # When/Then: claiming a full list is rejected deterministically.
    with pytest.raises(ValidationError, match="report_products_completeness_mismatch"):
        context_models.ReportScopeProductsViewV1(
            available=True,
            reason_code="ok",
            products=products,
            page=1,
            page_size=50,
            total_count=201,
            returned_count=200,
            products_are_paginated=True,
            full_product_list_in_projection=True,
        )
