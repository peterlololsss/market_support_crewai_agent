from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Annotated, ClassVar, Literal, assert_never

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.context.policy_view_models import (
    BusinessScopeViewV1 as BusinessScopeViewV1,
)
from market_support_crewai_agent.runtime.context.policy_view_models import (
    EffectivePolicyViewV1 as EffectivePolicyViewV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.schemas.base import StrictModel


class _FrozenCommonView(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class CurrentMessageViewV1(_FrozenCommonView):
    contract_version: Literal["current-message-view.v1"] = "current-message-view.v1"
    text: str = Field(min_length=1)


class HistoryTurnViewV1(_FrozenCommonView):
    contract_version: Literal["history-turn-view.v1"] = "history-turn-view.v1"
    role: Literal["user", "assistant"]
    text: str = Field(max_length=1_200)
    age_seconds: int | None = Field(default=None, ge=0, le=2_592_000)


class RelativeYearsViewV1(_FrozenCommonView):
    current: int
    last: int
    two_years_ago: int


class RuntimeClockViewV1(_FrozenCommonView):
    contract_version: Literal["runtime-clock-view.v1"] = "runtime-clock-view.v1"
    timezone: Literal["Asia/Shanghai"] = "Asia/Shanghai"
    current_date: str = Field(
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    current_datetime: str = Field(min_length=25)
    weekday: int = Field(ge=1, le=7)
    relative_years: RelativeYearsViewV1

    @field_validator("current_date")
    @classmethod
    def validate_current_date(cls, value: str) -> str:
        _ = date.fromisoformat(value)
        return value

    @field_validator("current_datetime")
    @classmethod
    def validate_current_datetime(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value)
        if not value.endswith("+08:00") or parsed.utcoffset() != timedelta(hours=8):
            raise ContextViewInvariantError("runtime_clock_timezone_offset_mismatch")
        return value

    @model_validator(mode="after")
    def validate_snapshot_consistency(self) -> RuntimeClockViewV1:
        current_date = date.fromisoformat(self.current_date)
        current_datetime = datetime.fromisoformat(self.current_datetime)
        expected_years = (
            current_date.year,
            current_date.year - 1,
            current_date.year - 2,
        )
        actual_years = (
            self.relative_years.current,
            self.relative_years.last,
            self.relative_years.two_years_ago,
        )
        if current_datetime.date() != current_date:
            raise ContextViewInvariantError("runtime_clock_date_mismatch")
        if self.weekday != current_date.isoweekday():
            raise ContextViewInvariantError("runtime_clock_weekday_mismatch")
        if actual_years != expected_years:
            raise ContextViewInvariantError("runtime_clock_relative_years_mismatch")
        return self


class PendingClarificationViewV1(_FrozenCommonView):
    contract_version: Literal["pending-clarification-view.v1"] = (
        "pending-clarification-view.v1"
    )
    kind: Literal["material_pack_option", "report_scope", "destination", "other"]
    slots: tuple[Annotated[str, Field(min_length=1, max_length=120)], ...] = Field(
        max_length=8
    )
    question: str = Field(max_length=400)
    age_turns: int = Field(ge=0, le=12)

    @field_validator("slots")
    @classmethod
    def validate_slots(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))):
            raise ContextViewInvariantError("pending_clarification_slots_not_canonical")
        if any(_has_control_characters(value) for value in values):
            raise ContextViewInvariantError(
                "pending_clarification_slot_control_character"
            )
        return values

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if _has_control_characters(value):
            raise ContextViewInvariantError(
                "pending_clarification_question_control_character"
            )
        return value


class MaterialPackOptionSummaryViewV1(_FrozenCommonView):
    contract_version: Literal["material-pack-option-summary.v1"] = (
        "material-pack-option-summary.v1"
    )
    total_count: int = Field(ge=0)
    exact_requested_option: str | None = Field(default=None, max_length=80)
    exact_match: bool | None = None
    bounded_page_available: bool

    @field_validator("exact_requested_option")
    @classmethod
    def validate_requested_option(cls, value: str | None) -> str | None:
        if value is not None and _has_control_characters(value):
            raise ContextViewInvariantError("material_option_control_character")
        return value


class ScenePresentationViewV1(_FrozenCommonView):
    contract_version: Literal["scene-presentation-view.v1"] = (
        "scene-presentation-view.v1"
    )
    scene: Literal["group", "direct"]
    audience: Literal["group", "individual"]
    conversation_name: str | None = Field(default=None, max_length=120)
    principal_name: str | None = Field(default=None, max_length=80)
    redacted: bool

    @field_validator("conversation_name", "principal_name")
    @classmethod
    def validate_names(cls, value: str | None) -> str | None:
        if value is not None and _has_control_characters(value):
            raise ContextViewInvariantError("scene_presentation_control_character")
        return value

    @model_validator(mode="after")
    def validate_scene_allowlist(self) -> ScenePresentationViewV1:
        match self.scene:
            case "group":
                if self.audience != "group":
                    raise ContextViewInvariantError(
                        "scene_presentation_audience_mismatch"
                    )
            case "direct":
                if self.audience != "individual":
                    raise ContextViewInvariantError(
                        "scene_presentation_audience_mismatch"
                    )
                if self.conversation_name is not None:
                    raise ContextViewInvariantError(
                        "direct_presentation_conversation_name_forbidden"
                    )
            case unreachable:
                assert_never(unreachable)
        return self


class IntentGateViewV1(_FrozenCommonView):
    contract_version: Literal["intent-gate-view.v1"] = "intent-gate-view.v1"
    artifact_hint: Literal[
        "material_pack",
        "weekly_report",
        "monthly_report",
        "knowledge_answer",
        "human_support",
        "refusal",
        "unclear",
        "smalltalk",
    ]
    outbound_action_hint: bool
    material_pack_option_count: int = Field(ge=0)
    compliance_hint: Literal["clean", "risky", "blocked", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


def _has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 for character in value)
