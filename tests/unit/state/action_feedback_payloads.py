from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import pytest
from fastapi.testclient import TestClient
from typing_extensions import NotRequired

from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    feedback_state_key_v2,
)
from market_support_crewai_agent.runtime.state.audit_records import (
    GroupAuditCommitProposalV1,
)
from market_support_crewai_agent.runtime.state.conversation_records import (
    AssistantTurnProposalV1,
    UserTurnProposalV1,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.transaction_records import (
    NewReplyReservationV1,
    ReplyCommitProposalV1,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.schemas.reply import (
    ReplyResponse,
    SendWeeklyReportAction,
)


class FeedbackIdentityPayload(TypedDict):
    contract_version: Literal["conversation-identity.v1"]
    surface: Literal["wecom"]
    scene: Literal["group"]
    tenant_ref: str
    group_ref: str
    principal_ref: str


class FeedbackArtifactPayload(TypedDict, total=False):
    type: str
    resolve_ref: str
    artifact_ref: str | None
    period: str
    report_date: str


class FeedbackAdapterResultPayload(TypedDict, total=False):
    ok: bool
    report_url: str


class FeedbackExecutionPayload(TypedDict):
    action_type: str
    status: str
    action_id: str
    artifact: FeedbackArtifactPayload
    adapter_result: FeedbackAdapterResultPayload
    material_type: NotRequired[str]


class FeedbackPayload(TypedDict):
    contract_version: Literal["action-feedback.v2"]
    feedback_id: str
    request_id: str
    response_id: str
    identity: FeedbackIdentityPayload
    executions: list[FeedbackExecutionPayload]


@dataclass(frozen=True, slots=True)
class IssuedWeeklyFeedbackFixture:
    coordinator: ReplyStateTransactionCoordinatorV1
    state_key: ConversationStateKey
    request_id: str
    payload: FeedbackPayload


@dataclass(frozen=True, slots=True)
class IssuedFeedbackRouteHarness:
    client: TestClient
    issued: IssuedWeeklyFeedbackFixture


def make_weekly_execution() -> FeedbackExecutionPayload:
    return {
        "action_type": "send_weekly_report",
        "status": "executed",
        "action_id": "act-" + "1" * 32,
        "artifact": {
            "type": "weekly_report",
            "resolve_ref": "weekly:resolve-ref",
            "artifact_ref": "weekly:opaque-ref",
            "period": "20260529",
            "report_date": "2026-05-29",
        },
        "adapter_result": {"ok": True},
    }


def make_feedback() -> FeedbackPayload:
    return {
        "contract_version": "action-feedback.v2",
        "feedback_id": "fb:feedback-1",
        "request_id": "req:feedback-1",
        "response_id": "resp-" + "1" * 32,
        "identity": {
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": "tenant:feedback",
            "group_ref": "group:feedback",
            "principal_ref": "principal:feedback",
        },
        "executions": [make_weekly_execution()],
    }


def make_feedback_request(
    payload: FeedbackPayload | None = None,
) -> ActionFeedbackRequestV2:
    return ActionFeedbackRequestV2.model_validate(payload or make_feedback())


def feedback_state_key() -> ConversationStateKey:
    return feedback_state_key_v2(
        make_feedback_request(),
        adapter_namespace="xiaoyan-wecom",
    )


def first_execution(payload: FeedbackPayload) -> FeedbackExecutionPayload:
    return payload["executions"][0]


def first_artifact(payload: FeedbackPayload) -> FeedbackArtifactPayload:
    return first_execution(payload)["artifact"]


def make_issued_weekly_feedback_fixture(
    *,
    request_id: str = "req:feedback-issued",
    feedback_id: str = "fb:feedback-issued",
) -> IssuedWeeklyFeedbackFixture:
    state_key = feedback_state_key()
    coordinator = ReplyStateTransactionCoordinatorV1()
    reservation = coordinator.reserve_reply(
        state_key,
        request_id,
        "rqh1:feedback-issued",
        True,
    )
    assert isinstance(reservation, NewReplyReservationV1)
    committed = coordinator.commit_reply(
        ReplyCommitProposalV1(
            state_key=state_key,
            request_id=reservation.request_id,
            request_hash=reservation.request_hash,
            replay_eligible=reservation.replay_eligible,
            owner_token=reservation.owner_token,
            expected_revision_epoch=reservation.revision_epoch,
            expected_state_revision=reservation.state_revision,
            response=ReplyResponse.model_validate(
                {
                    "reply": {"kind": "answer", "text": "已安排发送。"},
                    "actions": [
                        {
                            "type": "send_weekly_report",
                            "resolve_type": "weekly_report",
                            "resolve_ref": "weekly:resolve-ref",
                            "period": "20260529",
                            "report_date": "2026-05-29",
                        }
                    ],
                }
            ),
            pol1="pol1:feedback",
            par1="par1:feedback",
            grh1="grh1:feedback",
            bsh1="bsh1:feedback",
            user_turn=UserTurnProposalV1(text="请发周报"),
            assistant_turn=AssistantTurnProposalV1(
                text="已安排发送。",
                reply_kind="answer",
                clarification_requested=False,
            ),
            clarification=None,
            audit=GroupAuditCommitProposalV1(
                reason_code="action_ready",
                manifest_refs=("weekly_report.send@2026-07-15.1",),
                program_attempts=(),
            ),
        )
    )
    action = SendWeeklyReportAction.model_validate(committed.actions[0].model_dump())
    payload = make_feedback()
    payload["feedback_id"] = feedback_id
    payload["request_id"] = request_id
    payload["response_id"] = committed.response_id
    execution = first_execution(payload)
    execution["action_id"] = action.action_id
    artifact = first_artifact(payload)
    artifact["resolve_ref"] = action.resolve_ref
    artifact["period"] = action.period
    artifact["report_date"] = action.report_date
    return IssuedWeeklyFeedbackFixture(
        coordinator=coordinator,
        state_key=state_key,
        request_id=request_id,
        payload=payload,
    )


def make_issued_feedback_route_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str,
    request_id: str = "req:feedback-issued",
    feedback_id: str = "fb:feedback-issued",
) -> IssuedFeedbackRouteHarness:
    from market_support_crewai_agent.server import auth, main
    from market_support_crewai_agent.settings_model import Settings

    issued = make_issued_weekly_feedback_fixture(
        request_id=request_id,
        feedback_id=feedback_id,
    )
    monkeypatch.setattr(main, "get_reply_state_coordinator", lambda: issued.coordinator)
    route_settings = Settings(api_key=api_key, deployment_tenant_ref="tenant:feedback")
    monkeypatch.setattr(main, "get_settings", lambda: route_settings)
    monkeypatch.setattr(auth, "get_settings", lambda: route_settings)
    return IssuedFeedbackRouteHarness(
        client=TestClient(main.app, headers={"X-API-Key": api_key}),
        issued=issued,
    )


def assert_single_executed_feedback_effect(
    issued: IssuedWeeklyFeedbackFixture,
) -> None:
    snapshot = issued.coordinator.read_turn_admission_snapshot(
        issued.state_key,
        "par1:feedback",
        "grh1:feedback",
        "bsh1:feedback",
    )
    assert len(snapshot.ledger_rows) == 1
    assert snapshot.ledger_rows[0].status == "executed"


def assert_no_feedback_effects(issued: IssuedWeeklyFeedbackFixture) -> None:
    snapshot = issued.coordinator.read_turn_admission_snapshot(
        issued.state_key,
        "par1:feedback",
        "grh1:feedback",
        "bsh1:feedback",
    )
    assert snapshot.ledger_rows == ()
