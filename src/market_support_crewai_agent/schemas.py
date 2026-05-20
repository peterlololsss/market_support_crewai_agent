from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


MaterialType = Literal["material", "weekly", "monthly"]
ChannelType = Literal["bank", "non_bank"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReplyRequest(StrictModel):
    context_id: str
    session_id: str
    message: str
    is_group: bool
    group_name: str | None = None
    dist_channel_name: str | None = None
    sender_nickname: str | None = None
    available_materials: list[MaterialType] = Field(
        default_factory=lambda: ["material", "weekly", "monthly"]
    )
    available_strategies: list[str] = Field(default_factory=list)
    channel_type: ChannelType | None = None


class SendTextAction(StrictModel):
    type: Literal["send_text"]
    text: str


class SendMaterialAction(StrictModel):
    type: Literal["send_material"]
    material_type: MaterialType
    strategy: str | None = None
    message: str | None = None


class MentionSalesAction(StrictModel):
    type: Literal["mention_sales"]
    reason: str
    message: str | None = None


class AskClarificationAction(StrictModel):
    type: Literal["ask_clarification"]
    text: str


class NoReplyAction(StrictModel):
    type: Literal["no_reply"]


ReplyAction = Annotated[
    Union[
        SendTextAction,
        SendMaterialAction,
        MentionSalesAction,
        AskClarificationAction,
        NoReplyAction,
    ],
    Field(discriminator="type"),
]


class ReplyResponse(StrictModel):
    text: str = ""
    actions: list[ReplyAction] = Field(default_factory=list)


class HealthResponse(StrictModel):
    status: Literal["ok"]
    service: str

