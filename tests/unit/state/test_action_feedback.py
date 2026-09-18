from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.state.action_ledger import (
    ActionLedger,
    get_action_ledger,
)
from tests.unit.state.action_feedback_payloads import (
    assert_no_feedback_effects,
    assert_single_executed_feedback_effect,
    feedback_state_key,
    first_artifact,
    first_execution,
    make_issued_feedback_route_harness,
    make_feedback,
    make_feedback_request,
)

_TEST_API_KEY = "feedback-test-key"


@pytest.fixture(autouse=True)
def configure_feedback_route_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKET_AGENT_API_KEY", _TEST_API_KEY)
    monkeypatch.setenv("MARKET_AGENT_DEPLOYMENT_TENANT_REF", "tenant:feedback")
    monkeypatch.setenv("CREWAI_MAX_RETRY_LIMIT", "0")


@pytest.fixture
def feedback_client() -> TestClient:
    from market_support_crewai_agent.server.main import app

    return TestClient(app, headers={"X-API-Key": _TEST_API_KEY})


def setup_function() -> None:
    get_action_ledger().clear()


def teardown_function() -> None:
    get_action_ledger().clear()


def test_action_feedback_route_rejects_unknown_issued_response(
    feedback_client: TestClient,
) -> None:
    response = feedback_client.post("/actions/feedback", json=make_feedback())

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "issued_response_not_found"
    assert get_action_ledger().count() == 0


def test_action_feedback_route_rejects_unversioned_feedback_contract(
    feedback_client: TestClient,
) -> None:
    payload = make_feedback()
    _ = payload.pop("contract_version")

    response = feedback_client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_feedback_contract"
    assert get_action_ledger().count() == 0


def test_action_feedback_requires_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")
    from market_support_crewai_agent.server.main import app

    unauthenticated_client = TestClient(app, headers={"X-API-Key": _TEST_API_KEY})

    response = unauthenticated_client.post("/actions/feedback", json=make_feedback())

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}
    assert get_action_ledger().count() == 0


def test_action_feedback_unconfigured_auth_fails_before_body_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MARKET_AGENT_API_KEY", raising=False)
    from market_support_crewai_agent.server.main import app

    unauthenticated_client = TestClient(app)

    response = unauthenticated_client.post(
        "/actions/feedback",
        content=b"{",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "v2_adapter_auth_required"
    assert get_action_ledger().count() == 0


def test_action_feedback_accepts_x_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = make_issued_feedback_route_harness(
        monkeypatch,
        api_key="secret",
        request_id="req:x-api-key-issued",
        feedback_id="fb:x-api-key-issued",
    )

    response = harness.client.post(
        "/actions/feedback",
        json=harness.issued.payload,
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "stored": 1}
    assert_single_executed_feedback_effect(harness.issued)


def test_action_feedback_is_idempotent_for_retried_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = make_issued_feedback_route_harness(
        monkeypatch,
        api_key=_TEST_API_KEY,
    )

    first = harness.client.post("/actions/feedback", json=harness.issued.payload)
    second = harness.client.post("/actions/feedback", json=harness.issued.payload)

    assert first.status_code == 200
    assert first.json() == {"status": "accepted", "stored": 1}
    assert second.status_code == 200
    assert second.json() == {"status": "accepted", "stored": 0}
    assert_single_executed_feedback_effect(harness.issued)


def test_action_ledger_records_status_transition_as_new_execution() -> None:
    ledger = get_action_ledger()
    state_key = feedback_state_key()
    failed_payload = make_feedback()
    first_execution(failed_payload)["status"] = "failed"
    first_artifact(failed_payload)["artifact_ref"] = None

    failed = make_feedback_request(failed_payload)
    executed = make_feedback_request()

    assert ledger.record_feedback(failed, state_key) == 1
    assert ledger.record_feedback(executed, state_key) == 1
    records = ledger.recent_for_conversation(state_key)
    assert [record.execution.status for record in records] == ["failed", "executed"]


def test_action_ledger_recent_executed_filters_skipped_and_failed() -> None:
    ledger = get_action_ledger()
    state_key = feedback_state_key()
    failed_payload = make_feedback()
    first_execution(failed_payload)["status"] = "failed"
    first_artifact(failed_payload)["artifact_ref"] = None
    skipped_payload = make_feedback()
    first_execution(skipped_payload)["status"] = "skipped"
    first_artifact(skipped_payload)["artifact_ref"] = None

    assert ledger.record_feedback(make_feedback_request(failed_payload), state_key) == 1
    assert (
        ledger.record_feedback(make_feedback_request(skipped_payload), state_key) == 1
    )
    assert ledger.record_feedback(make_feedback_request(), state_key) == 1

    all_records = ledger.recent_for_conversation(state_key)
    executed_records = ledger.recent_executed_for_conversation(state_key)

    assert [record.execution.status for record in all_records] == [
        "failed",
        "skipped",
        "executed",
    ]
    assert [record.execution.status for record in executed_records] == ["executed"]


def test_action_feedback_accepts_empty_best_effort_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = make_issued_feedback_route_harness(
        monkeypatch,
        api_key=_TEST_API_KEY,
        request_id="req:empty-best-effort",
        feedback_id="fb:empty-best-effort",
    )
    payload = harness.issued.payload
    payload["executions"] = []

    response = harness.client.post("/actions/feedback", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "stored": 0}
    assert_no_feedback_effects(harness.issued)


def test_action_feedback_rejects_unknown_execution_status(
    feedback_client: TestClient,
) -> None:
    payload = make_feedback()
    first_execution(payload)["status"] = "maybe"

    response = feedback_client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_unknown_action_type(
    feedback_client: TestClient,
) -> None:
    payload = make_feedback()
    first_execution(payload)["action_type"] = "send_batch_material"

    response = feedback_client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_raw_artifact_ref_locator(
    feedback_client: TestClient,
) -> None:
    payload = make_feedback()
    first_artifact(payload)["artifact_ref"] = "https://example.invalid/weekly"

    response = feedback_client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_flat_artifact_fields(
    feedback_client: TestClient,
) -> None:
    payload = make_feedback()
    first_execution(payload)["material_type"] = "weekly"

    response = feedback_client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_requires_artifact_for_send_actions(
    feedback_client: TestClient,
) -> None:
    payload = make_feedback()
    _ = first_execution(payload).pop("artifact")

    response = feedback_client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_wrong_artifact_for_action_type(
    feedback_client: TestClient,
) -> None:
    payload = make_feedback()
    first_execution(payload)["artifact"] = {"type": "material_pack"}

    response = feedback_client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_feedback_rejects_raw_adapter_result_locator(
    feedback_client: TestClient,
) -> None:
    payload = make_feedback()
    first_execution(payload)["adapter_result"] = {
        "ok": True,
        "report_url": "https://example.invalid/weekly",
    }

    response = feedback_client.post("/actions/feedback", json=payload)

    assert response.status_code == 422
    assert get_action_ledger().count() == 0


def test_action_ledger_can_query_by_context_id() -> None:
    ledger = get_action_ledger()
    state_key = feedback_state_key()
    first_payload = make_feedback()
    first_payload["request_id"] = "req:msg-1"
    second_payload = make_feedback()
    second_payload["request_id"] = "req:msg-2"

    assert ledger.record_feedback(make_feedback_request(first_payload), state_key) == 1
    assert ledger.record_feedback(make_feedback_request(second_payload), state_key) == 1

    records = ledger.by_context_id("req:msg-2")

    assert len(records) == 1
    assert records[0].context_id == "req:msg-2"


def test_action_ledger_expires_old_records_and_rebuilds_dedupe_keys() -> None:
    clock = [datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)]
    ledger = ActionLedger(ttl_seconds=10, now_factory=lambda: clock[0])
    feedback = make_feedback_request()
    state_key = feedback_state_key()

    assert ledger.record_feedback(feedback, state_key) == 1

    clock[0] += timedelta(seconds=11)

    assert ledger.cleanup_expired() == 1
    assert ledger.count() == 0
    assert ledger.record_feedback(feedback, state_key) == 1


def test_action_ledger_recent_executed_ignores_expired_records() -> None:
    clock = [datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)]
    ledger = ActionLedger(ttl_seconds=10, now_factory=lambda: clock[0])
    state_key = feedback_state_key()

    assert ledger.record_feedback(make_feedback_request(), state_key) == 1
    clock[0] += timedelta(seconds=10)

    assert ledger.recent_executed_for_conversation(state_key) == []
    assert ledger.count() == 0
