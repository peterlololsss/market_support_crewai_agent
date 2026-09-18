from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Annotated, ClassVar, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)


class CanonicalModelV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class CanonicalDecimalStringV1(RootModel[str]):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    @field_validator("root", mode="before")
    @classmethod
    def canonicalize(cls, value: Decimal | float | str) -> str:
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise ValueError("invalid_canonical_decimal") from error
        if not decimal_value.is_finite():
            raise ValueError("non_finite_canonical_decimal")
        normalized = format(decimal_value.normalize(), "f")
        if normalized.startswith("-") and Decimal(normalized) == 0:
            return "0"
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        if normalized.startswith("."):
            normalized = f"0{normalized}"
        if normalized.startswith("-."):
            normalized = f"-0{normalized[1:]}"
        integer, _separator, fractional = normalized.partition(".")
        significant = len(integer.lstrip("-0") + fractional)
        if significant > 30 or len(integer.lstrip("-")) > 30 or len(fractional) > 29:
            raise ValueError("canonical_decimal_precision_exceeded")
        if not normalized or normalized == "-0":
            return "0"
        return normalized


class EvidenceNullValueV1(CanonicalModelV1):
    kind: Literal["null"] = "null"


class EvidenceBooleanValueV1(CanonicalModelV1):
    kind: Literal["boolean"] = "boolean"
    value: bool


class EvidenceIntegerValueV1(CanonicalModelV1):
    kind: Literal["integer"] = "integer"
    value: int = Field(ge=-(2**63), le=2**63 - 1)


class EvidenceNumberValueV1(CanonicalModelV1):
    kind: Literal["number"] = "number"
    value: CanonicalDecimalStringV1


class EvidenceStringValueV1(CanonicalModelV1):
    kind: Literal["string"] = "string"
    value: str = Field(min_length=0, max_length=20_000)


class EvidenceContentValueV1(CanonicalModelV1):
    kind: Literal["content"] = "content"
    media_type: Literal["text/plain", "text/markdown", "application/json"]
    charset: Literal["utf-8"] = "utf-8"
    text: str = Field(min_length=0, max_length=1_000_000)


EvidenceValueCanonicalV1: TypeAlias = Annotated[
    EvidenceNullValueV1
    | EvidenceBooleanValueV1
    | EvidenceIntegerValueV1
    | EvidenceNumberValueV1
    | EvidenceStringValueV1
    | EvidenceContentValueV1,
    Field(discriminator="kind"),
]


class ReportScopeProductCanonicalV1(CanonicalModelV1):
    product_name: str = Field(min_length=1, max_length=160)
    portfolio_type: str | None = Field(default=None, min_length=1, max_length=120)
    report_section: str = Field(min_length=1, max_length=160)
    source_pdf_status: Literal["found", "missing"]
    final_report_status: Literal["generated", "not_generated"]


class ReportScopeSectionCanonicalV1(CanonicalModelV1):
    name: str = Field(min_length=1, max_length=160)
    source_pdf_count: int = Field(ge=0)
    final_report_count: int = Field(ge=0)
    missing_product_count: int = Field(ge=0)


class ReportScopeSummaryCanonicalV1(CanonicalModelV1):
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
    sections: tuple[ReportScopeSectionCanonicalV1, ...] = Field(max_length=64)


class ReportScopeMatchCanonicalV1(CanonicalModelV1):
    status: Literal["matched", "not_found", "ambiguous"]
    query: str = Field(min_length=1, max_length=200)
    match_type: Literal["section", "product"] | None = None
    section: str | None = Field(default=None, min_length=1, max_length=160)
    candidate_count: int = Field(ge=0)
    products: tuple[ReportScopeProductCanonicalV1, ...] = Field(max_length=50)
    page: int = Field(ge=1, le=65_535)
    page_size: int = Field(ge=1, le=50)


class ReportScopeProductsCanonicalV1(CanonicalModelV1):
    available: bool
    reason_code: str = Field(max_length=120)
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=40)
    products: tuple[ReportScopeProductCanonicalV1, ...] = Field(max_length=200)
    page: int = Field(ge=1, le=65_535)
    page_size: int = Field(ge=1, le=50)
    total_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    products_are_paginated: Literal[True] = True
    full_product_list_in_projection: bool

    @model_validator(mode="after")
    def validate_product_projection(self) -> ReportScopeProductsCanonicalV1:
        if self.returned_count != len(self.products):
            raise ValueError("report_products_returned_count_mismatch")
        if self.full_product_list_in_projection != (
            self.total_count <= self.returned_count
        ):
            raise ValueError("report_products_projection_completeness_mismatch")
        return self


class ScalarEvidencePayloadCanonicalV1(CanonicalModelV1):
    kind: Literal["scalar"] = "scalar"
    value: EvidenceValueCanonicalV1


class ReportSummaryEvidencePayloadCanonicalV1(CanonicalModelV1):
    kind: Literal["report_summary"] = "report_summary"
    value: EvidenceNullValueV1
    report_payload: ReportScopeSummaryCanonicalV1


class ReportMatchEvidencePayloadCanonicalV1(CanonicalModelV1):
    kind: Literal["report_match"] = "report_match"
    value: EvidenceNullValueV1
    report_payload: ReportScopeMatchCanonicalV1


class ReportProductsEvidencePayloadCanonicalV1(CanonicalModelV1):
    kind: Literal["report_products"] = "report_products"
    value: EvidenceNullValueV1
    report_payload: ReportScopeProductsCanonicalV1


class RecentExecutedActionEvidencePayloadCanonicalV1(CanonicalModelV1):
    kind: Literal["recent_executed_action"] = "recent_executed_action"
    value: EvidenceNullValueV1
    action_type: Literal[
        "send_material_pack", "send_weekly_report", "send_monthly_report"
    ]
    artifact_type: Literal["material_pack", "weekly_report", "monthly_report"]
    artifact_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:~-]{0,159}$",
    )
    material_option: str | None = Field(default=None, min_length=1, max_length=80)
    period: str | None = Field(default=None, min_length=1, max_length=40)
    report_date: str | None = Field(default=None, min_length=1, max_length=40)
    status_revision: int = Field(ge=1)
    response_id: str = Field(pattern=r"^resp-[0-9a-f]{32}$")
    action_id: str = Field(pattern=r"^act-[0-9a-f]{32}$")
    received_at_epoch_seconds: int = Field(ge=0)

    @field_validator("artifact_ref")
    @classmethod
    def reject_ambiguous_artifact_ref(cls, value: str | None) -> str | None:
        if value is not None and ".." in value:
            raise ValueError("canonical_executed_action_artifact_ref_invalid")
        return value

    @model_validator(mode="after")
    def validate_action_artifact_binding(
        self,
    ) -> RecentExecutedActionEvidencePayloadCanonicalV1:
        expected_artifact_type: Literal[
            "material_pack", "weekly_report", "monthly_report"
        ]
        match self.action_type:
            case "send_material_pack":
                expected_artifact_type = "material_pack"
            case "send_weekly_report":
                expected_artifact_type = "weekly_report"
            case "send_monthly_report":
                expected_artifact_type = "monthly_report"
        if self.artifact_type != expected_artifact_type:
            raise ValueError("canonical_executed_action_artifact_type_mismatch")
        match self.artifact_type:
            case "material_pack":
                if self.period is not None or self.report_date is not None:
                    raise ValueError(
                        "canonical_material_pack_action_report_metadata_forbidden"
                    )
            case "weekly_report" | "monthly_report":
                if self.material_option is not None:
                    raise ValueError(
                        "canonical_report_action_material_option_forbidden"
                    )
        return self


EvidencePayloadCanonicalV1: TypeAlias = Annotated[
    ScalarEvidencePayloadCanonicalV1
    | ReportSummaryEvidencePayloadCanonicalV1
    | ReportMatchEvidencePayloadCanonicalV1
    | ReportProductsEvidencePayloadCanonicalV1
    | RecentExecutedActionEvidencePayloadCanonicalV1,
    Field(discriminator="kind"),
]

CanonicalEvidenceReportPayloadV1: TypeAlias = (
    ReportScopeSummaryCanonicalV1
    | ReportScopeMatchCanonicalV1
    | ReportScopeProductsCanonicalV1
)
