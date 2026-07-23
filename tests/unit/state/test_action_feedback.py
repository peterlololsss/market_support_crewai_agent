from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.state.action_ledger import (
    ActionLedger,
    get_action_ledger,
)
from market_support_crewai_agent.schemas import ActionFeedbackRequest
from market_support_crewai_agent.schemas import PrimaryReply
from market_support_crewai_agent.server import main as server_main
from market_support_crewai_agent.server.main import app


client = TestClient(app)


def setup_function():
    get_action_ledger().clear()


def teardown_function():
    get_action_ledger().clear()


def make_feedback(**overrides):
    payload = {
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "context_id": "msg-1",
        "response_id": "resp-1",
        "executions": [
            {
                "action_type": "send_weekly_report",
                "status": "executed",
                "action_id": "act-1",
                "artifact": {
                    "type": "weekly_report",
                    "resolve_ref": "weekly:resolve-ref",
                    "artifact_ref": "weekly:opaque-ref",
                    "period": "20260529",
                    "report_date": "2026-05-29",
                },
                "adapter_result": {"ok": True},
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_action_feedback_route_records_execution_status():
    response = client.post("/actions/feedback", json=make_feedback())

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "stored": 1}

    records = get_action_ledger().recent_for_conversation(
        "wecom:group-1:sender-1",
    )
    assert len(records) == 1
    assert records[0].context_id == "msg-1"
    assert records[0].response_id == "resp-1"
    assert records[0].execution.action_id == "act-1"
    assert records[0].execution.status == "executed"
    assert records[0].execution.artifact.type == "weekly_report"


def test_action_feedback_requires_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")

    response = client.post("/actions/feedback", json=make_feedback())

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}
    assert get_action_ledger().count() == 0


def test_action_feedback_accepts_x_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")

    response = client.post(
        "/actions/feedback",
        json=make_feedback(),
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "stored": 1}
    assert get_action_ledger().count() == 1


def test_action_feedback_is_idempotent_for_retried_payload():
    first = client.post("/actions/feedback", json=make_feedback())
    second = client.post("/actions/feedback", json=make_feedback())

    assert first.status_code == 200
    assert first.json() == {"status": "accepted", "stored": 1}
    assert second.status_code == 200
    assert second.json() == {"status": "accepted", "stored": 0}
    assert get_action_ledger().count() == 1


def test_action_feedback_records_status_transition_as_new_execution():
    failed_payload = make_feedback()
    failed_payload["executions"][0]["status"] = "failed"

    failed = client.post("/actions/feedback", json=failed_payload)
    executed = client.post("/actions/feedback", json=make_feedback())

    assert failed.status_code == 200
    assert failed.json() == {"status": "accepted", "stored": 1}
    assert executed.status_code == 200
    assert executed.json() == {"status": "accepted", "stored": 1}
    records = get_action_ledger().recent_for_conversation(
        "wecom:group-1:sender-1",
    )
    assert [record.execution.status for record in records] == ["failed", "executed"]


def test_action_ledger_recent_executed_filters_skipped_and_failed():
    failed_payload = make_feedback()
    failed_payload["executions"][0]["status"] = "failed"
    skipped_payload = make_feedback()
    skipped_payload["executions"][0]["status"] = "skipped"

    client.post("/actions/feedback", json=failed_payload)
    client.post("/actions/feedback", json=skipped_payload)
    client.post("/actions/feedback", json=make_feedback())

    all_records = get_action_ledger().recent_for_conversation(
        "wecom:group-1:sender-1",
    )
    executed_records = get_action_ledger().recent_executed_for_conversation(
        "wecom:group-1:sender-1",
    )

    assert [record.execution.status for record in all_records] == [
        "failed",
        "skipped",
        "executed",
    ]
    assert [record.execution.status for record in executed_records] == ["executed"]


def test_action_feedback_accepts_empty_best_effort_payload():
    response = client.post(
        "/actions/feedback",
        json=make_feedback(executions=[]),
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "stored": 0}
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_unknown_execution_status():
    payload = make_feedback()
    payload["executions"][0]["status"] = "maybe"

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_unknown_action_type():
    payload = make_feedback()
    payload["executions"][0]["action_type"] = "send_batch_material"

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_prepare_outbound_feedback_preserves_confirmation_ref_for_next_dm_turn():
    payload = make_feedback()
    payload["executions"] = [
        {
            "action_type": "prepare_outbound_message",
            "status": "executed",
            "action_id": "act-prepare",
            "artifact": None,
            "adapter_result": {
                "ok": True,
                "confirmation_ref": "wecom-adapter-confirmation:abc123",
                "state": "prepared",
                "target": {"kind": "group", "name": "银河客户群"},
                "content": {"kind": "text"},
            },
        }
    ]

    feedback = ActionFeedbackRequest.model_validate(payload)

    assert feedback.executions[0].adapter_result["confirmation_ref"] == (
        "wecom-adapter-confirmation:abc123"
    )


def test_prepare_outbound_feedback_route_accepts_adapter_payload():
    payload = make_feedback()
    payload["executions"] = [
        {
            "action_type": "prepare_outbound_message",
            "status": "executed",
            "action_id": "act-prepare",
            "artifact": None,
            "adapter_result": {
                "ok": True,
                "action": "prepare_outbound_message",
                "confirmation_ref": "wecom-adapter-confirmation:abc123",
                "state": "feedback_pending",
                "expires_at": 1784529047,
                "replayed": False,
                "requires_feedback_ack": True,
                "target": {
                    "kind": "channel",
                    "name": "银河证券",
                    "resolve_ref": "outbound-target:" + "b" * 64,
                    "target_count": 1,
                },
                "content": {"kind": "text"},
            },
        }
    ]

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "stored": 1}


def test_execute_feedback_returns_agent_composed_primary_reply(monkeypatch):
    payload = make_feedback()
    payload["executions"] = [
        {
            "action_type": "execute_prepared_outbound_message",
            "status": "executed",
            "action_id": "act-execute",
            "artifact": None,
            "adapter_result": {
                "ok": True,
                "outcome": "partial",
                "target_count": 2,
                "attempted_count": 2,
                "accepted_count": 1,
                "failed_count": 1,
                "unattempted_count": 0,
            },
        }
    ]
    compose = AsyncMock(
        return_value=PrimaryReply(
            kind="answer",
            text="有一个目标已提交，另一个未成功。",
            mentions=[],
        )
    )
    monkeypatch.setattr(server_main, "build_action_feedback_reply", compose)

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "status": "accepted",
        "stored": 1,
        "reply": {
            "kind": "answer",
            "text": "有一个目标已提交，另一个未成功。",
            "mentions": [],
        },
    }
    compose.assert_awaited_once()


def test_action_feedback_rejects_raw_artifact_ref_locator():
    payload = make_feedback()
    payload["executions"][0]["artifact"]["artifact_ref"] = "https://example.invalid/weekly"

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_flat_artifact_fields():
    payload = make_feedback()
    payload["executions"][0]["material_type"] = "weekly"

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_requires_artifact_for_send_actions():
    payload = make_feedback()
    payload["executions"][0].pop("artifact")

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_wrong_artifact_for_action_type():
    payload = make_feedback()
    payload["executions"][0]["artifact"] = {"type": "material_pack"}

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_raw_adapter_result_locator():
    payload = make_feedback()
    payload["executions"][0]["adapter_result"] = {
        "ok": True,
        "report_url": "https://example.invalid/weekly",
    }

    response = client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_ledger_can_query_by_context_id():
    client.post("/actions/feedback", json=make_feedback(context_id="msg-1"))
    client.post("/actions/feedback", json=make_feedback(context_id="msg-2"))

    records = get_action_ledger().by_context_id("msg-2")

    assert len(records) == 1
    assert records[0].context_id == "msg-2"


def test_action_ledger_expires_old_records_and_rebuilds_dedupe_keys():
    clock = [datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)]
    ledger = ActionLedger(ttl_seconds=10, now_factory=lambda: clock[0])
    feedback = ActionFeedbackRequest.model_validate(make_feedback())

    assert ledger.record_feedback(feedback) == 1

    clock[0] += timedelta(seconds=11)

    assert ledger.cleanup_expired() == 1
    assert ledger.count() == 0
    assert ledger.record_feedback(feedback) == 1


def test_action_ledger_recent_executed_ignores_expired_records():
    clock = [datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)]
    ledger = ActionLedger(ttl_seconds=10, now_factory=lambda: clock[0])

    ledger.record_feedback(ActionFeedbackRequest.model_validate(make_feedback()))
    clock[0] += timedelta(seconds=10)

    assert ledger.recent_executed_for_conversation("wecom:group-1:sender-1") == []
    assert ledger.count() == 0
