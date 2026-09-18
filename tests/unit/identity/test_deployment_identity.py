from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.identity.deployment import (
    DeploymentIdentityError,
    validate_deployment_identity,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    FeedbackReceiptReplayV1,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.server import adapter_compatibility, auth, main
from market_support_crewai_agent.settings_model import Settings


def test_validate_deployment_identity_returns_for_exact_canonical_match() -> None:
    validate_deployment_identity("tenant:primary", "tenant:primary")


@pytest.mark.parametrize("configured_tenant_ref", (None, ""))
def test_validate_deployment_identity_unavailable_when_configuration_missing(
    configured_tenant_ref: str | None,
) -> None:
    with pytest.raises(DeploymentIdentityError) as exc_info:
        validate_deployment_identity("tenant:primary", configured_tenant_ref)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "deployment_identity_unavailable"
    assert str(exc_info.value) == "deployment_identity_unavailable"


def test_validate_deployment_identity_mismatch_fails_closed() -> None:
    with pytest.raises(DeploymentIdentityError) as exc_info:
        validate_deployment_identity("tenant:other", "tenant:primary")

    assert exc_info.value.status_code == 403
    assert exc_info.value.code == "deployment_tenant_mismatch"
    assert str(exc_info.value) == "deployment_tenant_mismatch"


def test_direct_late_feedback_reaches_correlation_when_flag_off_without_adapter(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    coordinator = ReplyStateTransactionCoordinatorV1()
    captured_keys: list[ConversationStateKey] = []
    adapter_calls: list[None] = []
    feedback = ActionFeedbackRequestV2.model_validate(
        {
            "contract_version": "action-feedback.v2",
            "feedback_id": "fb:late-direct",
            "request_id": "req:late-direct",
            "response_id": "resp-" + "a" * 32,
            "identity": {
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "direct",
                "tenant_ref": "tenant:route",
                "direct_thread_ref": "direct:issued",
                "principal_ref": "principal:issued",
            },
            "executions": [],
        }
    )

    def capture_prepare(
        state_key: ConversationStateKey,
        request: ActionFeedbackRequestV2,
    ) -> FeedbackReceiptReplayV1:
        captured_keys.append(state_key)
        return FeedbackReceiptReplayV1(
            state_key=state_key,
            feedback_id=request.feedback_id,
            feedback_hash="afh1:late",
            request_id=request.request_id,
            response_id=request.response_id,
            receipt_id="fbr1:late",
            issued_record_revision=1,
        )

    def forbidden_adapter(*_args, **_kwargs) -> None:
        adapter_calls.append(None)
        raise AssertionError("late feedback must not instantiate an adapter")

    monkeypatch.setattr(coordinator, "prepare_feedback", capture_prepare)
    monkeypatch.setattr(main, "get_reply_state_coordinator", lambda: coordinator)
    route_settings = Settings(
        api_key="route-key",
        deployment_tenant_ref="tenant:route",
        internal_dm_enabled=False,
    )
    monkeypatch.setattr(main, "get_settings", lambda: route_settings)
    monkeypatch.setattr(auth, "get_settings", lambda: route_settings)
    adapter_compatibility.clear_compatibility_client_cache_for_testing()
    request.addfinalizer(
        adapter_compatibility.clear_compatibility_client_cache_for_testing
    )
    monkeypatch.setattr(
        adapter_compatibility, "AdapterResolveClient", forbidden_adapter
    )
    monkeypatch.setattr(
        "market_support_crewai_agent.runtime.integrations.adapter.client.AdapterResolveClient",
        forbidden_adapter,
    )

    result = TestClient(main.app).post(
        "/actions/feedback",
        json=feedback.model_dump(mode="json"),
        headers={"X-API-Key": "route-key"},
    )

    assert result.status_code == 200
    assert result.json() == {"status": "accepted", "stored": 0}
    assert captured_keys == [
        ConversationStateKey(
            surface="wecom",
            adapter_namespace="xiaoyan-wecom",
            tenant_ref="tenant:route",
            scene="direct",
            subject_ref="direct:issued",
            principal_ref="principal:issued",
        )
    ]
    assert adapter_calls == []
