from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


MaterialType = Literal["material", "weekly", "monthly"]
ChannelType = Literal["bank", "non_bank"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReplyRequest(StrictModel):
    conversation_key: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    sender_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    is_group: bool
    context_id: str | None = None
    group_name: str = Field(min_length=1)
    dist_channel_name: str = Field(min_length=1)
    sender_nickname: str = Field(min_length=1)
    available_materials: list[MaterialType]
    available_strategies: list[str]
    channel_type: ChannelType


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
