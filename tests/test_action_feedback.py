from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.action_ledger import (
    ActionLedger,
    get_action_ledger,
)
from market_support_crewai_agent.schemas import ActionFeedbackRequest
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
                "action_type": "send_material",
                "status": "executed",
                "action_id": "act-1",
                "material_type": "weekly",
                "strategy": None,
                "material_id": "weekly:opaque-ref",
                "version": "20260529",
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
    assert records[0].execution.material_type == "weekly"


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


def test_action_feedback_rejects_raw_material_id_locator():
    payload = make_feedback()
    payload["executions"][0]["material_id"] = "https://example.invalid/weekly"

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
