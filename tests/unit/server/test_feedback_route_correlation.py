from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from market_support_crewai_agent.runtime.identity import ConversationStateKey
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
from market_support_crewai_agent.schemas.reply import (
    ReplyResponse,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.settings_model import Settings

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from market_support_crewai_agent.server import auth, main


def _state_key(tenant_ref: str = "tenant:route") -> ConversationStateKey:
    return ConversationStateKey(
        surface="wecom",
        adapter_namespace="xiaoyan-wecom",
        tenant_ref=tenant_ref,
        scene="group",
        subject_ref="group:issued",
        principal_ref="principal:issued",
    )


def _issued_response(
    coordinator: ReplyStateTransactionCoordinatorV1,
    state_key: ConversationStateKey,
    request_id: str,
) -> ReplyResponse:
    reservation = coordinator.reserve_reply(state_key, request_id, "rqh1:route", True)
    assert isinstance(reservation, NewReplyReservationV1)
    response = ReplyResponse.model_validate(
        {
            "reply": {"kind": "answer", "text": "已安排发送。"},
            "actions": [
                {
                    "type": "send_weekly_report",
                    "resolve_type": "weekly_report",
                    "resolve_ref": "weekly:issued",
                    "period": "20260717",
                    "report_date": "2026-07-17",
                }
            ],
        }
    )
    return coordinator.commit_reply(
        ReplyCommitProposalV1(
            state_key=state_key,
            request_id=reservation.request_id,
            request_hash=reservation.request_hash,
            replay_eligible=reservation.replay_eligible,
            owner_token=reservation.owner_token,
            expected_revision_epoch=reservation.revision_epoch,
            expected_state_revision=reservation.state_revision,
            response=response,
            pol1="pol1:route",
            par1="par1:route",
            grh1="grh1:route",
            bsh1="bsh1:route",
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


def _v2_payload(
    response: ReplyResponse,
    request_id: str,
    *,
    feedback_id: str = "fb:route-v2",
    tenant_ref: str = "tenant:route",
    group_ref: str = "group:issued",
    principal_ref: str = "principal:issued",
    action_id: str | None = None,
) -> dict[str, JsonValue]:
    action = response.actions[0]
    assert isinstance(action, SendWeeklyReportAction)
    return {
        "contract_version": "action-feedback.v2",
        "feedback_id": feedback_id,
        "request_id": request_id,
        "response_id": response.response_id,
        "identity": {
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": tenant_ref,
            "group_ref": group_ref,
            "principal_ref": principal_ref,
        },
        "executions": [
            {
                "action_type": "send_weekly_report",
                "status": "failed",
                "action_id": action.action_id if action_id is None else action_id,
                "artifact": {
                    "type": "weekly_report",
                    "resolve_ref": action.resolve_ref,
                    "period": action.period,
                    "report_date": action.report_date,
                },
            }
        ],
    }


def test_feedback_route_replays_same_v2_feedback_without_reapplying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: one issued action and a matching v2 feedback payload.
    coordinator = ReplyStateTransactionCoordinatorV1()
    state_key = _state_key()
    request_id = "req:route-correlation"
    response = _issued_response(coordinator, state_key, request_id)
    monkeypatch.setattr(main, "get_reply_state_coordinator", lambda: coordinator)
    route_settings = Settings(
        api_key="route-key",
        deployment_tenant_ref="tenant:route",
    )
    monkeypatch.setattr(main, "get_settings", lambda: route_settings)
    monkeypatch.setattr(auth, "get_settings", lambda: route_settings)
    client = TestClient(main.app)

    # When: the adapter retries the same feedback body.
    payload = _v2_payload(response, request_id)
    first = client.post(
        "/actions/feedback", json=payload, headers={"X-API-Key": "route-key"}
    )
    second = client.post(
        "/actions/feedback", json=payload, headers={"X-API-Key": "route-key"}
    )

    # Then: the first call stores one effect and the retry replays its receipt.
    assert first.status_code == 200
    assert first.json() == {"status": "accepted", "stored": 1}
    assert second.status_code == 200
    assert second.json() == {"status": "accepted", "stored": 0}
    snapshot = coordinator.read_turn_admission_snapshot(
        state_key,
        "par1:route",
        "grh1:route",
        "bsh1:route",
    )
    assert len(snapshot.ledger_rows) == 1
    assert snapshot.ledger_rows[0].status == "failed"


def test_v2_feedback_cannot_cross_correlate_response_and_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two immutable issued records in separate tenant states.
    coordinator = ReplyStateTransactionCoordinatorV1()
    first = _issued_response(coordinator, _state_key("tenant:first"), "req:first")
    second = _issued_response(coordinator, _state_key("tenant:second"), "req:second")
    payload = _v2_payload(
        first,
        "req:first",
        tenant_ref="tenant:first",
        action_id=second.actions[0].action_id,
    )
    monkeypatch.setattr(main, "get_reply_state_coordinator", lambda: coordinator)
    route_settings = Settings(api_key="route-key", deployment_tenant_ref="tenant:first")
    monkeypatch.setattr(main, "get_settings", lambda: route_settings)
    monkeypatch.setattr(auth, "get_settings", lambda: route_settings)
    client = TestClient(main.app)
    before = coordinator.root_revision()

    # When: a v2 payload combines an issued response and action from different states.
    result = client.post(
        "/actions/feedback", json=payload, headers={"X-API-Key": "route-key"}
    )

    # Then: the route fails closed before any coordinator mutation.
    assert result.status_code == 409
    assert result.json()["detail"]["code"] == "feedback_effect_mismatch"
    assert coordinator.root_revision() == before


def test_v2_feedback_wrong_principal_fails_identity_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an issued response bound to one principal.
    coordinator = ReplyStateTransactionCoordinatorV1()
    state_key = _state_key()
    response = _issued_response(coordinator, state_key, "req:wrong-principal")
    payload = _v2_payload(
        response,
        "req:wrong-principal",
        feedback_id="fb:wrong-principal",
        principal_ref="principal:other",
    )
    monkeypatch.setattr(main, "get_reply_state_coordinator", lambda: coordinator)
    route_settings = Settings(api_key="route-key", deployment_tenant_ref="tenant:route")
    monkeypatch.setattr(main, "get_settings", lambda: route_settings)
    monkeypatch.setattr(auth, "get_settings", lambda: route_settings)
    client = TestClient(main.app)
    before = coordinator.root_revision()

    # When: the feedback identity names another principal.
    result = client.post(
        "/actions/feedback", json=payload, headers={"X-API-Key": "route-key"}
    )

    # Then: correlation fails before state mutation.
    assert result.status_code == 409
    assert result.json()["detail"]["code"] == "feedback_identity_mismatch"
    assert coordinator.root_revision() == before


def test_unversioned_feedback_is_rejected_before_state_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a valid issued response and a payload without the active version marker.
    coordinator = ReplyStateTransactionCoordinatorV1()
    response = _issued_response(coordinator, _state_key(), "req:unversioned")
    payload = _v2_payload(response, "req:unversioned")
    _removed = payload.pop("contract_version")
    monkeypatch.setattr(main, "get_reply_state_coordinator", lambda: coordinator)
    route_settings = Settings(api_key="route-key", deployment_tenant_ref="tenant:route")
    monkeypatch.setattr(main, "get_settings", lambda: route_settings)
    monkeypatch.setattr(auth, "get_settings", lambda: route_settings)
    client = TestClient(main.app)
    before = coordinator.root_revision()

    # When: the request is submitted without `action-feedback.v2`.
    result = client.post(
        "/actions/feedback", json=payload, headers={"X-API-Key": "route-key"}
    )

    # Then: the route rejects it before resolving IDs or mutating state.
    assert result.status_code == 422
    assert result.json()["detail"]["code"] == "invalid_feedback_contract"
    assert coordinator.root_revision() == before
