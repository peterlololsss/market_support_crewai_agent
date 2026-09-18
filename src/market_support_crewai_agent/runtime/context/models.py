from __future__ import annotations

import json
from datetime import date
from typing import Annotated, ClassVar, Final, Literal

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from market_support_crewai_agent.runtime.context.business_view_models import (
    BusinessFactsViewV1,
    GuardrailDecisionViewV1,
    ReportStateViewV1,
    ResolvableStateViewV1,
)
from market_support_crewai_agent.runtime.context.common_view_models import (
    BusinessScopeViewV1,
    CurrentMessageViewV1,
    EffectivePolicyViewV1,
    HistoryTurnViewV1,
    IntentGateViewV1,
    MaterialPackOptionSummaryViewV1,
    PendingClarificationViewV1,
    RuntimeClockViewV1,
    ScenePresentationViewV1,
)
from market_support_crewai_agent.runtime.context.evidence_view_models import (
    EvidenceBooleanValueViewV1,
    EvidenceContentValueViewV1,
    EvidenceFactViewV1,
    EvidenceIntegerValueViewV1,
    EvidenceNullValueViewV1,
    EvidenceNumberValueViewV1,
    EvidenceScopeViewV1,
    EvidenceStringValueViewV1,
    EvidenceValueViewV1,
)
from market_support_crewai_agent.runtime.context.grounding_projection_context import (
    GroundingProjectionContextV1,
)
from market_support_crewai_agent.runtime.context.plan_view_models import (
    ActionIntentViewV1,
    DistributionPlanScopeViewV1,
    PlanScopeViewV1,
    PlanTimeRangeViewV1,
    UnscopedPlanScopeViewV1,
    ValidatedPlanUnitViewV1,
    ValidatedPlanViewV1,
)
from market_support_crewai_agent.runtime.context.recall_view_models import (
    RecallPlannerViewV1,
)
from market_support_crewai_agent.runtime.context.report_view_models import (
    ReportScopeMatchViewV1,
    ReportScopeProductsViewV1,
    ReportScopeProductViewV1,
    ReportScopeSectionViewV1,
    ReportScopeSummaryViewV1,
)
from market_support_crewai_agent.runtime.context.response_view_models import (
    CandidateActionViewV1,
    CandidateMentionViewV1,
    CandidateReplyViewV1,
    EffectiveOutputCeilingsViewV1,
    PreflightFactViewV1,
    ResponseDirectiveViewV1,
)
from market_support_crewai_agent.runtime.context.retry_view_models import (
    ComposerRetryOverlayV1,
    PlannerRetryOverlayV1,
    PlanValidationIssueViewV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.schemas.base import StrictModel

__all__ = [
    "ActionIntentViewV1",
    "BusinessFactsViewV1",
    "BusinessScopeViewV1",
    "CandidateActionViewV1",
    "CandidateMentionViewV1",
    "CandidateReplyViewV1",
    "ComposerRetryOverlayV1",
    "CurrentMessageViewV1",
    "DistributionPlanScopeViewV1",
    "EffectiveOutputCeilingsViewV1",
    "EffectivePolicyViewV1",
    "EvidenceBooleanValueViewV1",
    "EvidenceContentValueViewV1",
    "EvidenceFactViewV1",
    "EvidenceIntegerValueViewV1",
    "EvidenceNullValueViewV1",
    "EvidenceNumberValueViewV1",
    "EvidenceScopeViewV1",
    "EvidenceStringValueViewV1",
    "EvidenceValueViewV1",
    "GroundingProjectionContextV1",
    "GuardrailDecisionViewV1",
    "HistoryTurnViewV1",
    "IntentGateViewV1",
    "MaterialPackOptionSummaryViewV1",
    "PendingClarificationViewV1",
    "PlanScopeViewV1",
    "PlanTimeRangeViewV1",
    "PlanValidationIssueViewV1",
    "PlannerRetryOverlayV1",
    "PreflightFactViewV1",
    "RecallPlannerViewV1",
    "ReportScopeMatchViewV1",
    "ReportScopeProductViewV1",
    "ReportScopeProductsViewV1",
    "ReportScopeSectionViewV1",
    "ReportScopeSummaryViewV1",
    "ReportStateViewV1",
    "ResolvableStateViewV1",
    "ResponseDirectiveViewV1",
    "RuntimeClockViewV1",
    "ScenePresentationViewV1",
    "UnscopedPlanScopeViewV1",
    "ValidatedPlanUnitViewV1",
    "ValidatedPlanViewV1",
]


class RecentExecutedActionSummaryViewV1(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["recent-action-summary-view.v1"] = (
        "recent-action-summary-view.v1"
    )
    action_type: Literal[
        "send_material_pack", "send_weekly_report", "send_monthly_report"
    ]
    status: Literal["executed"] = "executed"
    artifact_type: Literal["material_pack", "weekly_report", "monthly_report"]
    material_pack_option: str | None = Field(default=None, max_length=80)
    period: str | None = Field(default=None, max_length=40)
    report_date: str | None = Field(
        default=None,
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    age_seconds: int | None = Field(default=None, ge=0, le=2_592_000)

    @field_validator("material_pack_option", "period")
    @classmethod
    def _reject_control_characters(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 for character in value):
            raise ContextViewInvariantError("recent_action_summary_control_character")
        return value

    @field_validator("report_date")
    @classmethod
    def _validate_iso_report_date(cls, value: str | None) -> str | None:
        if value is not None:
            _ = date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _validate_action_artifact_binding(
        self,
    ) -> RecentExecutedActionSummaryViewV1:
        expected_artifact_type = _ARTIFACT_TYPE_BY_ACTION[self.action_type]
        if self.artifact_type != expected_artifact_type:
            raise ContextViewInvariantError(
                "recent_action_summary_artifact_type_mismatch"
            )
        if self.artifact_type == "material_pack":
            if self.period is not None or self.report_date is not None:
                raise ContextViewInvariantError(
                    "recent_action_summary_report_metadata_forbidden"
                )
        elif self.material_pack_option is not None:
            raise ContextViewInvariantError(
                "recent_action_summary_material_option_forbidden"
            )
        return self


_ = BusinessFactsViewV1.model_rebuild(
    _types_namespace={
        "RecentExecutedActionSummaryViewV1": RecentExecutedActionSummaryViewV1,
    }
)


class UnitGroundingViewV1(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["unit-grounding-view.v1"] = "unit-grounding-view.v1"
    unit_id: str = Field(min_length=1, max_length=120)
    manifest_ref: ManifestRefV1
    answerability: Literal[
        "answer",
        "send",
        "clarify",
        "abstain",
        "refuse",
        "handoff",
        "smalltalk",
        "no_reply",
    ]
    scope: PlanScopeViewV1
    evidence_query: str | None = Field(default=None, max_length=200)
    action_intents: tuple[ActionIntentViewV1, ...] = Field(default=(), max_length=3)
    allowed_evidence_ids: tuple[
        Annotated[str, Field(pattern=r"^eid1:[0-9a-f]{64}$")], ...
    ] = Field(default=(), max_length=32)
    allowed_evidence: tuple[EvidenceFactViewV1, ...] = Field(default=(), max_length=32)
    business_facts: BusinessFactsViewV1
    guardrail_decisions: tuple[GuardrailDecisionViewV1, ...] = Field(
        default=(), max_length=32
    )

    @model_validator(mode="after")
    def validate_evidence_binding(self) -> UnitGroundingViewV1:
        expected_ids = tuple(fact.evidence_id for fact in self.allowed_evidence)
        if self.allowed_evidence_ids != expected_ids:
            raise ContextViewInvariantError(
                "unit_grounding_view_evidence_binding_mismatch"
            )
        if len(set(expected_ids)) != len(expected_ids):
            raise ContextViewInvariantError("unit_grounding_view_duplicate_evidence_id")
        if self.business_facts.evidence_fact_count != len(self.allowed_evidence):
            raise ContextViewInvariantError(
                "unit_grounding_view_business_fact_count_mismatch"
            )
        return self


def prompt_json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


_ARTIFACT_TYPE_BY_ACTION: Final[
    dict[
        Literal["send_material_pack", "send_weekly_report", "send_monthly_report"],
        Literal["material_pack", "weekly_report", "monthly_report"],
    ]
] = {
    "send_material_pack": "material_pack",
    "send_weekly_report": "weekly_report",
    "send_monthly_report": "monthly_report",
}
