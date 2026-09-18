from __future__ import annotations

import re
from datetime import date
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from market_support_crewai_agent.schemas.adapter import reject_raw_locator_text
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.conversation import (
    ConversationIdentityV1,
    validate_prefixed_opaque_ref,
)
from market_support_crewai_agent.schemas.feedback_sanitization import (
    SanitizedAdapterResultV1,
)
from market_support_crewai_agent.schemas.type_ids import (
    ActionExecutionStatus,
    ActionExecutionType,
)


class ActionFeedbackArtifactBase(StrictModel):
    resolve_ref: str | None = None
    artifact_ref: str | None = None

    @field_validator("artifact_ref")
    @classmethod
    def validate_opaque_artifact_ref(cls, value: str | None) -> str | None:
        reject_raw_locator_text(value, "artifact_ref")
        return value

    @field_validator("resolve_ref")
    @classmethod
    def validate_opaque_resolve_ref(cls, value: str | None) -> str | None:
        reject_raw_locator_text(value, "resolve_ref")
        return value


class MaterialPackFeedbackArtifact(ActionFeedbackArtifactBase):
    type: Literal["material_pack"]
    option: str | None = None


class WeeklyReportFeedbackArtifact(ActionFeedbackArtifactBase):
    type: Literal["weekly_report"]
    period: str | None = None
    report_date: str | None = None


class MonthlyReportFeedbackArtifact(ActionFeedbackArtifactBase):
    type: Literal["monthly_report"]
    period: str | None = None
    report_date: str | None = None


ActionFeedbackArtifact = Annotated[
    MaterialPackFeedbackArtifact
    | WeeklyReportFeedbackArtifact
    | MonthlyReportFeedbackArtifact,
    Field(discriminator="type"),
]


class ActionFeedbackResponse(StrictModel):
    status: Literal["accepted"]
    stored: int


def _validate_v2_feedback_artifact(
    *,
    action_type: ActionExecutionType,
    status: ActionExecutionStatus,
    artifact: ActionFeedbackArtifact,
) -> None:
    expected_type = {
        "send_material_pack": "material_pack",
        "send_weekly_report": "weekly_report",
        "send_monthly_report": "monthly_report",
    }[action_type]
    if artifact.type != expected_type:
        raise ValueError("feedback artifact does not match action type")
    _validate_v2_opaque_adapter_ref(artifact.resolve_ref, "resolve_ref", required=True)
    artifact_ref = artifact.artifact_ref
    match status:
        case "executed":
            _validate_v2_opaque_adapter_ref(artifact_ref, "artifact_ref", required=True)
        case "failed" | "skipped":
            if artifact_ref is not None:
                raise ValueError("non-executed send feedback forbids artifact_ref")
    match artifact:
        case MaterialPackFeedbackArtifact(option=option):
            if option is not None and (not option or len(option) > 80):
                raise ValueError(
                    "material feedback option must be 1-80 chars when present"
                )
        case WeeklyReportFeedbackArtifact(period=period, report_date=report_date):
            _validate_v2_report_metadata(period, report_date)
        case MonthlyReportFeedbackArtifact(period=period, report_date=report_date):
            _validate_v2_report_metadata(period, report_date)


def _validate_v2_opaque_adapter_ref(
    value: str | None,
    field_name: str,
    *,
    required: bool,
) -> None:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return
    if (
        not value
        or len(value) > 160
        or value != value.strip()
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:~-]{0,159}", value)
        or ".." in value
        or value.startswith((".", "~"))
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{field_name} must be an opaque adapter ref")
    reject_raw_locator_text(value, field_name)


def _validate_v2_report_metadata(period: str | None, report_date: str | None) -> None:
    if period is None or not period or len(period) > 40:
        raise ValueError("report feedback requires period of 1-40 chars")
    if report_date is None or not report_date:
        raise ValueError("report feedback requires report_date")
    try:
        parsed = date.fromisoformat(report_date)
    except ValueError as exc:
        raise ValueError("report_date must be strict ISO YYYY-MM-DD") from exc
    if parsed.isoformat() != report_date:
        raise ValueError("report_date must be strict ISO YYYY-MM-DD")


class ActionExecutionFeedbackV2(StrictModel):
    action_type: ActionExecutionType
    status: ActionExecutionStatus
    action_id: str | None = None
    artifact: ActionFeedbackArtifact | None = None
    adapter_result: SanitizedAdapterResultV1 = Field(
        default_factory=lambda: SanitizedAdapterResultV1({})
    )

    @field_validator("action_id")
    @classmethod
    def validate_action_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"act-[0-9a-f]{32}", value):
            raise ValueError("action_id must be a server-issued action id")
        return value

    @model_validator(mode="after")
    def validate_v2_effect_binding(self):
        match self.action_type:
            case "send_material_pack" | "send_weekly_report" | "send_monthly_report":
                if self.action_id is None or self.artifact is None:
                    raise ValueError("send feedback requires action_id and artifact")
                _validate_v2_feedback_artifact(
                    action_type=self.action_type,
                    status=self.status,
                    artifact=self.artifact,
                )
            case "send_text" | "mention_sales":
                if self.action_id is not None or self.artifact is not None:
                    raise ValueError(
                        "response-level feedback forbids action_id and artifact"
                    )
        return self


class ActionFeedbackRequestV2(StrictModel):
    contract_version: Literal["action-feedback.v2"]
    feedback_id: str
    request_id: str
    response_id: str = Field(pattern=r"^resp-[0-9a-f]{32}$")
    identity: ConversationIdentityV1
    executions: list[ActionExecutionFeedbackV2] = Field(max_length=5)

    @field_validator("feedback_id")
    @classmethod
    def validate_feedback_id(cls, value: str) -> str:
        return validate_prefixed_opaque_ref(
            value, prefix="fb:", field_name="feedback_id"
        )

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return validate_prefixed_opaque_ref(
            value, prefix="req:", field_name="request_id"
        )

    @model_validator(mode="after")
    def validate_execution_keys(self):
        action_ids = [
            item.action_id for item in self.executions if item.action_id is not None
        ]
        response_effects = [
            item.action_type for item in self.executions if item.action_id is None
        ]
        if len(set(action_ids)) != len(action_ids) or len(set(response_effects)) != len(
            response_effects
        ):
            raise ValueError("feedback execution effects must be unique")
        return self
