from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.effect_records import (
    FeedbackReceiptReplayV1,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from tests.helpers.reply_contract_requests import make_v2_payload


class ErrorDetail(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    code: str


class ErrorEnvelope(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    detail: ErrorDetail


def reply_response() -> ReplyResponse:
    return ReplyResponse(
        response_id="resp-" + "a" * 32,
        reply=PrimaryReply(kind="answer", text="ok"),
        actions=[],
    )


def direct_request() -> ReplyRequestV2:
    return ReplyRequestV2.model_validate(
        make_v2_payload(
            "介绍一下公司",
            identity={
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "direct",
                "tenant_ref": "tenant:test",
                "direct_thread_ref": "direct:route-test",
                "principal_ref": "principal:route-test",
            },
            presentation={
                "contract_version": "direct-presentation.v1",
                "principal_name": "test user",
            },
            business_scope={"kind": "unscoped"},
            grants={
                "contract_version": "principal-grants.v1",
                "read_capabilities": ["query_internal_company_info"],
                "outbound_actions": [],
                "mention_types": [],
            },
        )
    )


def empty_feedback(
    scene: Literal["group", "direct"],
) -> ActionFeedbackRequestV2:
    identity = (
        {
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": "tenant:route",
            "group_ref": "group:issued",
            "principal_ref": "principal:issued",
        }
        if scene == "group"
        else {
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:route",
            "direct_thread_ref": "direct:issued",
            "principal_ref": "principal:issued",
        }
    )
    return ActionFeedbackRequestV2.model_validate(
        {
            "contract_version": "action-feedback.v2",
            "feedback_id": "fb:route-empty",
            "request_id": "req:route-empty",
            "response_id": "resp-" + "a" * 32,
            "identity": identity,
            "executions": [],
        }
    )


def feedback_replay(
    state_key: ConversationStateKey,
    request: ActionFeedbackRequestV2,
) -> FeedbackReceiptReplayV1:
    return FeedbackReceiptReplayV1(
        state_key=state_key,
        feedback_id=request.feedback_id,
        feedback_hash="afh1:late",
        request_id=request.request_id,
        response_id=request.response_id,
        receipt_id="fbr1:late",
        issued_record_revision=1,
    )
