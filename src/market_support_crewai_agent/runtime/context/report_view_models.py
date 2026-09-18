from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.schemas.base import StrictModel


class _FrozenReportView(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReportScopeSectionViewV1(_FrozenReportView):
    name: str = Field(min_length=1, max_length=160)
    source_pdf_count: int = Field(ge=0)
    final_report_count: int = Field(ge=0)
    missing_product_count: int = Field(ge=0)


class ReportScopeProductViewV1(_FrozenReportView):
    product_name: str = Field(min_length=1, max_length=160)
    portfolio_type: str | None = Field(default=None, min_length=1, max_length=120)
    report_section: str = Field(min_length=1, max_length=160)
    source_pdf_status: Literal["found", "missing"]
    final_report_status: Literal["generated", "not_generated"]


class ReportScopeSummaryViewV1(_FrozenReportView):
    contract_version: Literal["report-scope-summary-view.v1"] = (
        "report-scope-summary-view.v1"
    )
    material_type: Literal["weekly", "monthly"]
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=40)
    period_start: str | None = Field(default=None, max_length=40)
    period_end: str | None = Field(default=None, max_length=40)
    period_label: str | None = Field(default=None, max_length=80)
    available: bool
    reason_code: str = Field(max_length=120)
    scope_complete: bool | None = None
    expected_product_count: int | None = Field(default=None, ge=0)
    generated_product_count: int | None = Field(default=None, ge=0)
    missing_product_count: int | None = Field(default=None, ge=0)
    sections: tuple[ReportScopeSectionViewV1, ...] = Field(max_length=64)


class ReportScopeMatchViewV1(_FrozenReportView):
    contract_version: Literal["report-scope-match-view.v1"] = (
        "report-scope-match-view.v1"
    )
    status: Literal["matched", "not_found", "ambiguous"]
    query: str = Field(min_length=1, max_length=200)
    match_type: Literal["section", "product"] | None = None
    section: str | None = Field(default=None, min_length=1, max_length=160)
    candidate_count: int = Field(ge=0)
    products: tuple[ReportScopeProductViewV1, ...] = Field(max_length=50)
    page: int = Field(ge=1, le=65_535)
    page_size: int = Field(ge=1, le=50)


class ReportScopeProductsViewV1(_FrozenReportView):
    contract_version: Literal["report-scope-products-view.v1"] = (
        "report-scope-products-view.v1"
    )
    available: bool
    reason_code: str = Field(max_length=120)
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=40)
    products: tuple[ReportScopeProductViewV1, ...] = Field(max_length=200)
    page: int = Field(ge=1, le=65_535)
    page_size: int = Field(ge=1, le=50)
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    products_are_paginated: Literal[True] = True
    full_product_list_in_projection: bool

    @model_validator(mode="after")
    def validate_completeness(self) -> ReportScopeProductsViewV1:
        if self.returned_count != len(self.products):
            raise ContextViewInvariantError("report_products_returned_count_mismatch")
        expected_complete = self.total_count <= self.returned_count
        if self.full_product_list_in_projection != expected_complete:
            raise ContextViewInvariantError("report_products_completeness_mismatch")
        return self
