from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.identity import (
    normalize_reply_request_v2,
    state_key_ref,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.server import adapter_compatibility, auth, main
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_payload


def _direct_request(
    request_id: str,
    principal_ref: str = "principal:alice",
) -> ReplyRequestV2:
    return ReplyRequestV2.model_validate(
        make_v2_payload(
            "T0请人工协助",
            request_id=request_id,
            identity={
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "direct",
                "tenant_ref": "tenant:primary",
                "direct_thread_ref": "direct:alice",
                "principal_ref": principal_ref,
            },
            presentation={
                "contract_version": "direct-presentation.v1",
                "principal_name": "Alice",
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


def _send_text_feedback(
    response_id: str,
    *,
    feedback_id: str,
    principal_ref: str = "principal:alice",
) -> ActionFeedbackRequestV2:
    return ActionFeedbackRequestV2.model_validate(
        {
            "contract_version": "action-feedback.v2",
            "feedback_id": feedback_id,
            "request_id": "req:direct-issued",
            "response_id": response_id,
            "identity": _direct_request("req:unused", principal_ref).identity,
            "executions": [{"action_type": "send_text", "status": "executed"}],
        }
    )


def test_flag_rollback_preserves_late_feedback_on_same_coordinator(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    # Given: the enabled route and real runtime issue a direct response.
    audit_key = base64.urlsafe_b64encode(b"a" * 32).decode("ascii").rstrip("=")
    enabled = Settings(
        api_key="service-key",
        deployment_tenant_ref="tenant:primary",
        internal_dm_enabled=True,
        direct_audit_hmac_key=audit_key,
        adapter_api_key="adapter-key",
        llm_api_key="test-key",
        reply_alignment_verifier_enabled=False,
    )
    coordinator = ReplyStateTransactionCoordinatorV1()
    runtime = CrewAIReplyRuntime(enabled, coordinator=coordinator)
    compatibility_calls: list[tuple[str, str]] = []

    class CompatibleAdapter:
        def assert_scene_compatible(self, scene: str, tenant_ref: str) -> None:
            compatibility_calls.append((scene, tenant_ref))

    adapter_compatibility.clear_compatibility_client_cache_for_testing()
    request.addfinalizer(
        adapter_compatibility.clear_compatibility_client_cache_for_testing
    )
    monkeypatch.setattr(main, "get_settings", lambda: enabled)
    monkeypatch.setattr(auth, "get_settings", lambda: enabled)
    monkeypatch.setattr(main, "get_reply_state_coordinator", lambda: coordinator)
    monkeypatch.setattr(
        adapter_compatibility, "new_compatibility_client", CompatibleAdapter
    )
    monkeypatch.setattr(main, "build_reply", runtime.reply)
    headers = {"X-API-Key": "service-key"}
    issued_result = TestClient(main.app).post(
        "/reply",
        json=_direct_request("req:direct-issued").model_dump(mode="json"),
        headers=headers,
    )
    assert issued_result.status_code == 200
    issued = ReplyResponse.model_validate_json(issued_result.content)
    state_key = normalize_reply_request_v2(
        _direct_request("req:direct-issued"),
        adapter_namespace="xiaoyan-wecom",
    ).state_key
    issued_record = coordinator.lookup_issued_response(
        state_key,
        issued.response_id,
    )
    revision_after_issue = coordinator.root_revision()
    assert runtime.coordinator is coordinator

    # When: a flag-only rollback rebuilds admission around the retained root.
    disabled = Settings(
        api_key="service-key",
        deployment_tenant_ref="tenant:primary",
        internal_dm_enabled=False,
    )
    monkeypatch.setattr(main, "get_settings", lambda: disabled)
    monkeypatch.setattr(auth, "get_settings", lambda: disabled)
    blocked = TestClient(main.app).post(
        "/reply",
        json=_direct_request("req:direct-blocked").model_dump(mode="json"),
        headers=headers,
    )

    # Then: no new turn mutates state, while late matching feedback remains valid.
    assert blocked.status_code == 503
    assert blocked.json()["detail"]["code"] == "internal_dm_disabled"
    assert coordinator.root_revision() == revision_after_issue
    assert (
        coordinator.lookup_issued_response(state_key, issued.response_id)
        == issued_record
    )
    assert state_key_ref(state_key).startswith("csk1:")

    late = TestClient(main.app).post(
        "/actions/feedback",
        json=_send_text_feedback(
            issued.response_id, feedback_id="fb:late-direct"
        ).model_dump(mode="json"),
        headers=headers,
    )
    assert late.status_code == 200
    assert late.json() == {"status": "accepted", "stored": 1}
    assert compatibility_calls == [("direct", "tenant:primary")]
    replay = TestClient(main.app).post(
        "/actions/feedback",
        json=_send_text_feedback(
            issued.response_id,
            feedback_id="fb:late-direct",
        ).model_dump(mode="json"),
        headers=headers,
    )
    assert replay.status_code == 200
    assert replay.json() == {"status": "accepted", "stored": 0}

    revision_after_late = coordinator.root_revision()
    mismatch = TestClient(main.app).post(
        "/actions/feedback",
        json=_send_text_feedback(
            issued.response_id,
            feedback_id="fb:cross-principal",
            principal_ref="principal:mallory",
        ).model_dump(mode="json"),
        headers=headers,
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["code"] == "feedback_identity_mismatch"
    assert coordinator.root_revision() == revision_after_late
