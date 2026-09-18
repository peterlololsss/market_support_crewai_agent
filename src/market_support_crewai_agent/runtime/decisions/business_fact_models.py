from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar, Literal, override

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.evidence.canonical_values import (
    CanonicalModelV1,
)

AvailabilityStatus = Literal["available", "ambiguous", "unavailable", "unknown"]
UserPermissionStatus = Literal["allowed", "denied", "unknown"]


@dataclass(frozen=True, slots=True)
class BusinessFactsContractError(ValueError):
    reason_code: str

    @override
    def __str__(self) -> str:
        return self.reason_code


class _FrozenBusinessFactsModel(CanonicalModelV1):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, strict=True
    )


class CanonicalResolvableStateV1(_FrozenBusinessFactsModel):
    availability: AvailabilityStatus = "unknown"
    candidate_labels: tuple[str, ...] = Field(default=(), max_length=8)
    reason_code: str = Field(default="", max_length=120)
    resolve_ref: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:~-]{0,159}$",
    )
    material_pack_option: str | None = Field(default=None, min_length=1, max_length=80)
    source_record_ref: str | None = Field(
        default=None,
        pattern=r"^esr1:[0-9a-f]{64}$",
    )

    @field_validator("resolve_ref")
    @classmethod
    def _validate_resolve_ref(cls, value: str | None) -> str | None:
        return _opaque_adapter_ref_v1(value)

    @model_validator(mode="after")
    def _validate_candidate_labels(self) -> CanonicalResolvableStateV1:
        if len(set(self.candidate_labels)) != len(self.candidate_labels):
            raise BusinessFactsContractError(
                "canonical_resolvable_duplicate_candidate_label"
            )
        if any(not label or len(label) > 120 for label in self.candidate_labels):
            raise BusinessFactsContractError(
                "canonical_resolvable_invalid_candidate_label"
            )
        return self


class CanonicalReportSectionV1(_FrozenBusinessFactsModel):
    name: str = Field(min_length=1, max_length=160)
    source_pdf_count: int = Field(ge=0)
    final_report_count: int = Field(ge=0)
    missing_product_count: int = Field(ge=0)


class CanonicalReportStateV1(CanonicalResolvableStateV1):
    period: str | None = Field(default=None, max_length=40)
    report_date: date | None = None
    period_start: date | None = None
    period_end: date | None = None
    period_label: str | None = Field(default=None, max_length=80)
    scope_complete: bool | None = None
    expected_product_count: int | None = Field(default=None, ge=0)
    generated_product_count: int | None = Field(default=None, ge=0)
    missing_product_count: int | None = Field(default=None, ge=0)
    report_sections: tuple[CanonicalReportSectionV1, ...] = Field(
        default=(), max_length=64
    )


class CanonicalExecutedActionStateV1(_FrozenBusinessFactsModel):
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
    material_pack_option: str | None = Field(default=None, min_length=1, max_length=80)
    period: str | None = Field(default=None, min_length=1, max_length=40)
    report_date: date | None = None
    status_revision: int = Field(ge=1)
    received_at_epoch_seconds: int = Field(ge=0)
    response_id: str = Field(pattern=r"^resp-[0-9a-f]{32}$")
    action_id: str = Field(pattern=r"^act-[0-9a-f]{32}$")
    source_record_ref: str = Field(pattern=r"^esr1:[0-9a-f]{64}$")

    @field_validator("artifact_ref")
    @classmethod
    def _validate_artifact_ref(cls, value: str | None) -> str | None:
        return _opaque_adapter_ref_v1(value)

    @model_validator(mode="after")
    def _validate_action_artifact_binding(self) -> CanonicalExecutedActionStateV1:
        expected_artifact = {
            "send_material_pack": "material_pack",
            "send_weekly_report": "weekly_report",
            "send_monthly_report": "monthly_report",
        }[self.action_type]
        if self.artifact_type != expected_artifact:
            raise BusinessFactsContractError(
                "canonical_executed_action_artifact_type_mismatch"
            )
        if self.artifact_type == "material_pack":
            if self.period is not None or self.report_date is not None:
                raise BusinessFactsContractError(
                    "canonical_material_action_report_metadata_forbidden"
                )
        elif self.material_pack_option is not None:
            raise BusinessFactsContractError(
                "canonical_report_action_material_option_forbidden"
            )
        return self


class UnitBusinessFactsV1(_FrozenBusinessFactsModel):
    material_pack: CanonicalResolvableStateV1 = Field(
        default_factory=CanonicalResolvableStateV1
    )
    weekly_report: CanonicalReportStateV1 = Field(
        default_factory=CanonicalReportStateV1
    )
    monthly_report: CanonicalReportStateV1 = Field(
        default_factory=CanonicalReportStateV1
    )
    sales_mention: CanonicalResolvableStateV1 = Field(
        default_factory=CanonicalResolvableStateV1
    )
    recent_executed_actions: tuple[CanonicalExecutedActionStateV1, ...] = Field(
        default=(), max_length=20
    )
    requested_material_pack_option_status: AvailabilityStatus = "unknown"
    user_permission: UserPermissionStatus = "unknown"
    evidence_fact_count: int = Field(ge=0, le=32)


def _opaque_adapter_ref_v1(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        value != value.strip()
        or any(ord(character) < 32 for character in value)
        or "%" in value
        or "/" in value
        or "\\" in value
        or "://" in value
        or value.startswith((".", "~"))
        or ".." in value
    ):
        raise BusinessFactsContractError("invalid_opaque_adapter_ref")
    return value


def strict_iso_date_v1(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise BusinessFactsContractError("invalid_strict_iso_date") from error
