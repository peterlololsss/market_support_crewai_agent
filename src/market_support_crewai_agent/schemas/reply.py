from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from market_support_crewai_agent.schemas.adapter import reject_raw_locator_text
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import ReplyKind, ReplyMentionType


class ReplyMention(StrictModel):
    type: ReplyMentionType
    reason: str | None = None


class PrimaryReply(StrictModel):
    kind: ReplyKind
    text: str = ""
    mentions: list[ReplyMention] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_reply_shape(self):
        if self.kind == "no_reply" and (self.text.strip() or self.mentions):
            raise ValueError("no_reply must not include text or mentions")
        return self


class OutboundActionBase(StrictModel):
    action_id: str = ""


class SendActionBase(OutboundActionBase):
    resolve_ref: str = Field(min_length=1)

    @field_validator("resolve_ref")
    @classmethod
    def validate_resolve_ref_is_opaque(cls, value: str) -> str:
        reject_raw_locator_text(value, "resolve_ref")
        return value


class SendMaterialPackAction(SendActionBase):
    type: Literal["send_material_pack"]
    resolve_type: Literal["material_pack"]
    material_pack_option: str | None = None


class SendWeeklyReportAction(SendActionBase):
    type: Literal["send_weekly_report"]
    resolve_type: Literal["weekly_report"]
    period: str = Field(min_length=1)
    report_date: str = Field(min_length=1)


class SendMonthlyReportAction(SendActionBase):
    type: Literal["send_monthly_report"]
    resolve_type: Literal["monthly_report"]
    period: str = Field(min_length=1)
    report_date: str = Field(min_length=1)


OutboundAction = Annotated[
    SendMaterialPackAction | SendWeeklyReportAction | SendMonthlyReportAction,
    Field(discriminator="type"),
]


class ReplyResponse(StrictModel):
    contract_version: Literal["reply"] = "reply"
    response_id: str = ""
    reply: PrimaryReply
    actions: list[OutboundAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_no_reply_has_no_actions(self):
        if self.reply.kind == "no_reply" and self.actions:
            raise ValueError("no_reply must not include outbound actions")
        return self
