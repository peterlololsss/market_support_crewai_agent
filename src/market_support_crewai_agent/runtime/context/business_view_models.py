from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    AvailabilityStatus,
    UserPermissionStatus,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailOutcome,
    GuardrailPhase,
)
from market_support_crewai_agent.schemas.base import StrictModel

if TYPE_CHECKING:

    class RecentExecutedActionSummaryViewV1(Protocol):
        @property
        def action_type(
            self,
        ) -> Literal[
            "send_material_pack", "send_weekly_report", "send_monthly_report"
        ]: ...

        @property
        def artifact_type(
            self,
        ) -> Literal["material_pack", "weekly_report", "monthly_report"]: ...


class _FrozenBusinessView(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResolvableStateViewV1(_FrozenBusinessView):
    availability: AvailabilityStatus
    candidate_labels: tuple[
        Annotated[str, Field(min_length=1, max_length=120)], ...
    ] = Field(default=(), max_length=8)
    reason_code: str = Field(default="", max_length=120)
    source_available: bool
    resolve_ref_available: bool
    material_pack_option: str | None = Field(default=None, max_length=80)

    @field_validator("candidate_labels")
    @classmethod
    def validate_candidate_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ContextViewInvariantError("business_view_candidate_labels_not_unique")
        if any(any(ord(character) < 32 for character in value) for value in values):
            raise ContextViewInvariantError(
                "business_view_candidate_label_control_character"
            )
        return values


class ReportStateViewV1(ResolvableStateViewV1):
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=10)
    period_start: str | None = Field(default=None, max_length=10)
    period_end: str | None = Field(default=None, max_length=10)
    period_label: str | None = Field(default=None, max_length=80)
    scope_complete: bool | None = None
    expected_product_count: int | None = Field(default=None, ge=0)
    generated_product_count: int | None = Field(default=None, ge=0)
    missing_product_count: int | None = Field(default=None, ge=0)
    report_section_labels: tuple[
        Annotated[str, Field(min_length=1, max_length=160)], ...
    ] = Field(default=(), max_length=64)

    @field_validator("report_date", "period_start", "period_end")
    @classmethod
    def validate_iso_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value


class GuardrailDecisionViewV1(_FrozenBusinessView):
    outcome: GuardrailOutcome
    phase: GuardrailPhase
    reason_code: str = Field(
        min_length=1,
        max_length=120,
        pattern=r"^[a-z][a-z0-9_]{0,119}$",
    )
    capability_id: str | None = Field(default=None, max_length=160)
    artifact_ids: tuple[Annotated[str, Field(min_length=1, max_length=160)], ...] = (
        Field(default=(), max_length=16)
    )
    evidence_required: tuple[
        Annotated[str, Field(min_length=1, max_length=160)], ...
    ] = Field(default=(), max_length=16)
    evidence_seen: tuple[Annotated[str, Field(min_length=1, max_length=160)], ...] = (
        Field(default=(), max_length=16)
    )

    @model_validator(mode="after")
    def validate_canonical_lists(self) -> GuardrailDecisionViewV1:
        for values in (
            self.artifact_ids,
            self.evidence_required,
            self.evidence_seen,
        ):
            if values != tuple(sorted(set(values))):
                raise ContextViewInvariantError("guardrail_view_ids_not_canonical")
        return self


class BusinessFactsViewV1(_FrozenBusinessView):
    contract_version: Literal["business-facts-view.v1"] = "business-facts-view.v1"
    material_pack: ResolvableStateViewV1
    weekly_report: ReportStateViewV1
    monthly_report: ReportStateViewV1
    sales_mention: ResolvableStateViewV1
    recent_executed_actions: tuple[RecentExecutedActionSummaryViewV1, ...] = Field(
        default=(), max_length=20
    )
    requested_material_pack_option_status: AvailabilityStatus
    user_permission: UserPermissionStatus
    evidence_fact_count: int = Field(ge=0, le=32)
