from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from market_support_crewai_agent.schemas import (
    OutboundAction,
    PrimaryReply,
    ReplyResponse,
    StrictModel,
)

ComposerResponseMode = Literal["answer", "abstain", "clarify"]


class ComposerReplyOutput(StrictModel):
    contract_version: Literal["composer-reply"] = "composer-reply"
    response_id: str = ""
    response_mode: ComposerResponseMode
    claims: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=20)
    missing_inputs: list[str] = Field(default_factory=list, max_length=20)
    reply: PrimaryReply
    actions: list[OutboundAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_mode_matches_reply(self):
        if self.response_mode == "answer" and self.reply.kind != "answer":
            raise ValueError("response_mode=answer requires reply.kind=answer")
        if self.response_mode == "abstain" and self.reply.kind != "unable_to_answer":
            raise ValueError(
                "response_mode=abstain requires reply.kind=unable_to_answer"
            )
        if self.response_mode == "abstain" and (self.claims or self.evidence_ids):
            raise ValueError("response_mode=abstain must not include claims or evidence_ids")
        if self.response_mode == "clarify" and self.reply.kind != "clarification":
            raise ValueError("response_mode=clarify requires reply.kind=clarification")
        if self.response_mode == "clarify" and (self.claims or self.evidence_ids):
            raise ValueError("response_mode=clarify must not include claims or evidence_ids")
        if self.actions:
            raise ValueError("composer output must not include actions")
        return self

    def to_reply_response(self) -> ReplyResponse:
        return ReplyResponse(
            contract_version="reply",
            response_id=self.response_id,
            reply=self.reply,
            actions=[],
        )
