from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import (
    AdapterReportScopeCommand,
    AdapterReportScopeMaterialType,
    AdapterResolveStatus,
    AdapterResolveType,
    AvailableArtifactType,
    ChannelType,
)


class AvailableArtifact(StrictModel):
    type: AvailableArtifactType
    options: list[str] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def clean_options(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            option = str(value).strip()
            if not option or option in seen:
                continue
            seen.add(option)
            output.append(option)
        return output

    @model_validator(mode="after")
    def validate_options_only_for_material_pack(self):
        if self.type != "material_pack" and self.options:
            raise ValueError(
                "available_artifacts options are only valid for material_pack"
            )
        return self


class AdapterResolveRequest(StrictModel):
    resolve_type: AdapterResolveType
    dist_name: str = Field(min_length=1)
    material_pack_option: str | None = None


class ReportScopeSection(StrictModel):
    name: str = Field(min_length=1)
    expected_product_count: int = Field(default=0, ge=0)
    generated_product_count: int = Field(default=0, ge=0)
    missing_product_count: int = Field(default=0, ge=0)


class ReportScopeProduct(StrictModel):
    product_name: str = Field(min_length=1)
    portfolio_type: str | None = None
    report_section: str = Field(default="unknown", min_length=1)
    source_pdf_status: Literal["found", "missing"]
    final_report_status: Literal["generated", "not_generated"]
    file_name: str | None = None


class AdapterResolveResult(StrictModel):
    contract_version: Literal["adapter-resolve"]
    resolve_type: AdapterResolveType
    status: AdapterResolveStatus
    display_name: str
    reason_code: str
    candidates: list[str] = Field(default_factory=list)
    channel_type: ChannelType = "unknown"
    available_artifacts: list[AvailableArtifact]
    resolved_at: int
    detail: str | None = None
    resolve_ref: str | None = None
    material_pack_option: str | None = None
    period: str | None = None
    report_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_label: str | None = None
    source_trade_date: str | None = None
    scope_complete: bool | None = None
    expected_product_count: int | None = Field(default=None, ge=0)
    generated_product_count: int | None = Field(default=None, ge=0)
    missing_product_count: int | None = Field(default=None, ge=0)
    report_sections: list[ReportScopeSection] = Field(
        default_factory=list, max_length=64
    )

    @field_validator("available_artifacts")
    @classmethod
    def validate_available_artifacts_unique(
        cls, values: list[AvailableArtifact]
    ) -> list[AvailableArtifact]:
        _reject_duplicate_available_artifacts(values)
        return values

    @field_validator("detail")
    @classmethod
    def validate_detail_is_sanitized(cls, value: str | None) -> str | None:
        reject_raw_locator_text(value, "detail")
        return value

    @field_validator("resolve_ref")
    @classmethod
    def validate_resolve_ref_is_opaque(cls, value: str | None) -> str | None:
        reject_raw_locator_text(value, "resolve_ref")
        return value

    @model_validator(mode="after")
    def validate_resolved_has_ref(self):
        if self.status == "resolved" and not (self.resolve_ref or "").strip():
            raise ValueError("resolved adapter results must include resolve_ref")
        return self


class AdapterResolveBatchRequest(StrictModel):
    requests: list[AdapterResolveRequest] = Field(min_length=1, max_length=16)


class AdapterResolveBatchResult(StrictModel):
    contract_version: Literal["adapter-resolve-batch"]
    results: list[AdapterResolveResult]


class AdapterReportScopeRequest(StrictModel):
    material_type: AdapterReportScopeMaterialType
    dist_name: str = Field(min_length=1)
    command: AdapterReportScopeCommand
    period: str | None = None
    query: str | None = None
    page: int | None = Field(default=None, ge=1)
    page_size: int | None = Field(default=None, ge=1, le=50)
    section_name: str | None = None


class AdapterReportScopeMatch(StrictModel):
    status: Literal["matched", "not_found", "ambiguous"]
    query: str = ""
    match_type: Literal["section", "product"] | None = None
    matched_section: str | None = None
    candidate_count: int = Field(default=0, ge=0)
    products: list[ReportScopeProduct] = Field(default_factory=list, max_length=50)
    product_page: int = Field(default=1, ge=1)
    product_page_size: int = Field(default=20, ge=1, le=50)


class AdapterReportScopeResult(StrictModel):
    contract_version: Literal["adapter-report-scope"]
    material_type: AdapterReportScopeMaterialType
    dist_name: str
    period: str
    status: AdapterResolveStatus
    reason_code: str
    report_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_label: str | None = None
    detail: str | None = None
    schema_version: str | None = None
    source_trade_date: str | None = None
    scope_complete: bool | None = None
    expected_product_count: int | None = Field(default=None, ge=0)
    generated_product_count: int | None = Field(default=None, ge=0)
    missing_product_count: int | None = Field(default=None, ge=0)
    report_sections: list[ReportScopeSection] = Field(
        default_factory=list, max_length=64
    )
    match: AdapterReportScopeMatch | None = None
    products: list[ReportScopeProduct] = Field(default_factory=list, max_length=50)
    product_page: int | None = Field(default=None, ge=1)
    product_page_size: int | None = Field(default=None, ge=1, le=50)
    product_total_count: int | None = Field(default=None, ge=0)

    @field_validator("detail")
    @classmethod
    def validate_detail_is_sanitized(cls, value: str | None) -> str | None:
        reject_raw_locator_text(value, "detail")
        return value


def _reject_duplicate_available_artifacts(values: list[AvailableArtifact]) -> None:
    seen: set[str] = set()
    for artifact in values:
        if artifact.type in seen:
            raise ValueError(
                "available_artifacts must not contain duplicate artifact types"
            )
        seen.add(artifact.type)


def reject_raw_locator_text(value: str | None, field_name: str) -> None:
    if not value:
        return
    if (
        "://" in value
        or "/" in value
        or "\\" in value
        or "file:" in value
        or value.startswith("~")
    ):
        raise ValueError(f"{field_name} contains raw locator values")
