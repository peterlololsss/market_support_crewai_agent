from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal, TypeAlias, Union, assert_never

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.context.report_view_models import (
    ReportScopeMatchViewV1,
    ReportScopeProductsViewV1,
    ReportScopeSummaryViewV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceArtifactTypeV1,
    CanonicalEvidenceFactTypeV1,
    CanonicalEvidenceSourceTypeV1,
)
from market_support_crewai_agent.schemas.base import StrictModel


class _FrozenEvidenceView(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvidenceNullValueViewV1(_FrozenEvidenceView):
    kind: Literal["null"] = "null"


class EvidenceBooleanValueViewV1(_FrozenEvidenceView):
    kind: Literal["boolean"] = "boolean"
    value: bool


class EvidenceIntegerValueViewV1(_FrozenEvidenceView):
    kind: Literal["integer"] = "integer"
    value: int = Field(ge=-(2**63), le=2**63 - 1)


class EvidenceNumberValueViewV1(_FrozenEvidenceView):
    kind: Literal["number"] = "number"
    value: str = Field(min_length=1, max_length=64)

    @field_validator("value")
    @classmethod
    def validate_finite_number(cls, value: str) -> str:
        try:
            parsed = Decimal(value)
        except InvalidOperation as error:
            raise ContextViewInvariantError("evidence_view_number_invalid") from error
        if not parsed.is_finite():
            raise ContextViewInvariantError("evidence_view_number_not_finite")
        return value


class EvidenceStringValueViewV1(_FrozenEvidenceView):
    kind: Literal["string"] = "string"
    value: str = Field(max_length=1_000)


class EvidenceContentValueViewV1(_FrozenEvidenceView):
    kind: Literal["content"] = "content"
    media_type: Literal["text/plain", "text/markdown", "application/json"]
    charset: Literal["utf-8"] = "utf-8"
    text: str = Field(max_length=1_000_000)


EvidenceValueViewV1: TypeAlias = Annotated[
    Union[
        EvidenceNullValueViewV1,
        EvidenceBooleanValueViewV1,
        EvidenceIntegerValueViewV1,
        EvidenceNumberValueViewV1,
        EvidenceStringValueViewV1,
        EvidenceContentValueViewV1,
    ],
    Field(discriminator="kind"),
]

EvidenceReportPayloadViewV1: TypeAlias = Annotated[
    Union[
        ReportScopeSummaryViewV1,
        ReportScopeMatchViewV1,
        ReportScopeProductsViewV1,
    ],
    Field(discriminator="contract_version"),
]


class EvidenceScopeViewV1(_FrozenEvidenceView):
    kind: Literal["distribution", "unscoped"]
    channel_kind: Literal["bank", "non_bank", "unknown"] | None = None
    channel_match: bool | None = None
    artifact_type: CanonicalEvidenceArtifactTypeV1 | None = None
    material_option: str | None = Field(default=None, max_length=80)
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(default=None, max_length=10)
    product_count: int | None = Field(default=None, ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_scope_shape(self) -> EvidenceScopeViewV1:
        match self.kind:
            case "distribution":
                if self.channel_kind is None or self.channel_match is None:
                    raise ContextViewInvariantError(
                        "evidence_scope_distribution_channel_required"
                    )
            case "unscoped":
                if self.channel_kind is not None or self.channel_match is not None:
                    raise ContextViewInvariantError(
                        "evidence_scope_unscoped_channel_forbidden"
                    )
            case unreachable:
                assert_never(unreachable)
        return self

    @field_validator("report_date")
    @classmethod
    def validate_iso_report_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value


class EvidenceFactViewV1(_FrozenEvidenceView):
    contract_version: Literal["evidence-fact-view.v1"] = "evidence-fact-view.v1"
    evidence_id: str = Field(pattern=r"^eid1:[0-9a-f]{64}$")
    fact_type: CanonicalEvidenceFactTypeV1
    source_type: CanonicalEvidenceSourceTypeV1
    artifact_type: CanonicalEvidenceArtifactTypeV1
    value: EvidenceValueViewV1 | None = None
    report_payload: EvidenceReportPayloadViewV1 | None = None
    scope: EvidenceScopeViewV1
    observed_age_seconds: int | None = Field(default=None, ge=0)
    provenance_available: bool
    citation_available: bool
    public_urls: tuple[Annotated[str, Field(min_length=1, max_length=2_048)], ...] = (
        Field(default=(), max_length=8)
    )

    @field_validator("public_urls")
    @classmethod
    def validate_unique_urls(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ContextViewInvariantError("evidence_view_public_urls_not_unique")
        return values

    @model_validator(mode="after")
    def validate_payload_binding(self) -> EvidenceFactViewV1:
        report_types = {
            "report_scope_summary": ReportScopeSummaryViewV1,
            "report_scope_match": ReportScopeMatchViewV1,
            "report_scope_products": ReportScopeProductsViewV1,
        }
        expected_type = report_types.get(self.fact_type)
        if expected_type is None:
            if self.value is None or self.report_payload is not None:
                raise ContextViewInvariantError("evidence_view_scalar_payload_mismatch")
        elif self.value is not None or not isinstance(
            self.report_payload, expected_type
        ):
            raise ContextViewInvariantError("evidence_view_report_payload_mismatch")
        if (
            self.scope.artifact_type is not None
            and self.artifact_type != self.scope.artifact_type
        ):
            raise ContextViewInvariantError("evidence_view_artifact_scope_mismatch")
        if self.citation_available != bool(self.public_urls):
            raise ContextViewInvariantError(
                "evidence_view_citation_availability_mismatch"
            )
        return self
