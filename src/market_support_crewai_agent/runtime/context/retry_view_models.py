from __future__ import annotations

from typing import ClassVar, Literal, TypeAlias

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.planning.models import (
    PlanValidationCode,
    PlanValidationSeverity,
)
from market_support_crewai_agent.schemas.base import StrictModel

PlanValidationIssueCodeViewV1: TypeAlias = (
    PlanValidationCode
    | Literal[
        "plan_spec_schema_invalid",
        "plan_spec_manifest_contract_mismatch",
        "plan_spec_scope_mismatch",
        "plan_spec_scope_value_not_allowed",
        "plan_spec_answerability_not_allowed",
        "plan_spec_evidence_query_not_allowed",
        "plan_spec_step_contract_mismatch",
        "plan_spec_risk_flag_mismatch",
        "plan_spec_time_range_not_projectable",
        "plan_unit_grounding_mismatch",
    ]
)


class _FrozenRetryView(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class PlanValidationIssueViewV1(_FrozenRetryView):
    code: PlanValidationIssueCodeViewV1
    severity: PlanValidationSeverity = "error"
    message: str = Field(min_length=1, max_length=300)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ContextViewInvariantError("plan_validation_message_control_character")
        return value


class PlannerRetryOverlayV1(_FrozenRetryView):
    contract_version: Literal["planner-retry-overlay.v1"] = "planner-retry-overlay.v1"
    attempt: int = Field(ge=1, le=3)
    phase: Literal["schema_repair", "alignment_replan"]
    phase_attempt: int = Field(ge=1, le=2)
    issues: tuple[PlanValidationIssueViewV1, ...] = Field(max_length=8)
    feedback: str | None = Field(default=None, max_length=300)

    @field_validator("feedback")
    @classmethod
    def validate_feedback(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 for character in value):
            raise ContextViewInvariantError("planner_retry_feedback_control_character")
        return value

    @model_validator(mode="after")
    def validate_attempt_mapping(self) -> PlannerRetryOverlayV1:
        actual = (self.attempt, self.phase, self.phase_attempt)
        allowed = (
            (1, "schema_repair", 1),
            (2, "alignment_replan", 1),
            (3, "alignment_replan", 2),
        )
        if actual not in allowed:
            raise ContextViewInvariantError("planner_retry_attempt_mapping_invalid")
        return self


class ComposerRetryOverlayV1(_FrozenRetryView):
    contract_version: Literal["composer-retry-overlay.v1"] = "composer-retry-overlay.v1"
    attempt: int = Field(ge=1, le=2)
    feedback: str = Field(min_length=1, max_length=300)

    @field_validator("feedback")
    @classmethod
    def validate_feedback(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ContextViewInvariantError("composer_retry_feedback_control_character")
        return value
