from __future__ import annotations

from collections import Counter

import pytest
from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    VerifiedRequestEnvelopeV1,
    feedback_state_key_v2,
    normalize_reply_request_v2,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    FeedbackReceiptReplayV1,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.server import auth, main
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_payload
from tests.unit.server.scene_admission_support import (
    ErrorDetail,
    ErrorEnvelope,
    empty_feedback,
    feedback_replay,
    reply_response,
)


def _install_reply_zero_work_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> Counter[str]:
    calls: Counter[str] = Counter()

    def count_normalize(
        request: ReplyRequestV2,
        *,
        adapter_namespace: str,
    ) -> VerifiedRequestEnvelopeV1:
        calls["normalize"] += 1
        return normalize_reply_request_v2(
            request,
            adapter_namespace=adapter_namespace,
        )

    async def count_build(_envelope: VerifiedRequestEnvelopeV1) -> ReplyResponse:
        calls["build"] += 1
        return reply_response()

    def count_coordinator() -> None:
        calls["coordinator"] += 1

    monkeypatch.setattr(main, "normalize_reply_request_v2", count_normalize)
    monkeypatch.setattr(main, "build_reply", count_build)
    monkeypatch.setattr(main, "get_reply_state_coordinator", count_coordinator)
    return calls


@pytest.mark.parametrize(
    ("settings", "credential", "body", "expected_status", "expected_detail"),
    (
        (
            Settings(api_key=None, deployment_tenant_ref="tenant:test"),
            None,
            b"{",
            503,
            ErrorDetail(code="v2_adapter_auth_required"),
        ),
        (
            Settings(api_key="secret", deployment_tenant_ref="tenant:test"),
            "wrong",
            b"{",
            401,
            "unauthorized",
        ),
        (
            Settings(api_key="secret", deployment_tenant_ref="tenant:test"),
            "secret",
            b"{",
            422,
            ErrorDetail(code="invalid_request_contract"),
        ),
    ),
)
def test_reply_auth_and_schema_precedence_before_tenant_work(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    credential: str | None,
    body: bytes,
    expected_status: int,
    expected_detail: ErrorDetail | str,
) -> None:
    calls = _install_reply_zero_work_spies(monkeypatch)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    headers = {"content-type": "application/json"}
    if credential is not None:
        headers["X-API-Key"] = credential

    response = TestClient(main.app).post("/reply", content=body, headers=headers)

    assert response.status_code == expected_status
    match expected_detail:
        case ErrorDetail():
            assert (
                ErrorEnvelope.model_validate_json(response.content).detail
                == expected_detail
            )
        case str():
            assert response.json() == {"detail": expected_detail}
    assert calls == Counter()


_LEGACY_REPLY_PAYLOAD = {
    "conversation_key": "wecom:g:s",
    "group_id": "g",
    "sender_id": "s",
    "message": "hello",
    "is_group": True,
    "group_name": "g",
    "dist_channel_name": "d",
    "sender_nickname": "s",
    "available_artifacts": [],
    "channel_type": "bank",
}


@pytest.mark.parametrize(
    "payload",
    (
        _LEGACY_REPLY_PAYLOAD,
        {**make_v2_payload(), "contract_version": "reply-request.v1"},
        make_v2_payload(
            identity={
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "group",
                "tenant_ref": "tenant:test",
                "group_ref": "group:contract-probe",
                "principal_ref": "principal:sender-1",
            }
        ),
    ),
    ids=("legacy-payload", "unknown-version", "reserved-probe-group"),
)
def test_reply_rejects_retired_or_reserved_payloads_with_zero_work(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    calls = _install_reply_zero_work_spies(monkeypatch)
    settings = Settings(api_key="secret", deployment_tenant_ref="tenant:test")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "get_settings", lambda: settings)

    response = TestClient(main.app).post(
        "/reply",
        json=payload,
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 422
    assert ErrorEnvelope.model_validate_json(response.content).detail == ErrorDetail(
        code="invalid_request_contract"
    )
    assert calls == Counter()


@pytest.mark.parametrize(
    ("configured_tenant_ref", "expected_status", "expected_code"),
    (
        (None, 503, "deployment_identity_unavailable"),
        ("tenant:other", 403, "deployment_tenant_mismatch"),
    ),
)
def test_reply_tenant_unavailable_or_mismatch_has_zero_work(
    monkeypatch: pytest.MonkeyPatch,
    configured_tenant_ref: str | None,
    expected_status: int,
    expected_code: str,
) -> None:
    calls = _install_reply_zero_work_spies(monkeypatch)
    settings = Settings(
        api_key="secret",
        deployment_tenant_ref=configured_tenant_ref,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "get_settings", lambda: settings)

    response = TestClient(main.app).post(
        "/reply",
        json=make_v2_payload(),
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == expected_status
    assert (
        ErrorEnvelope.model_validate_json(response.content).detail.code == expected_code
    )
    assert calls == Counter()


def test_reply_matching_tenant_normalizes_once_and_preserves_full_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalized: list[VerifiedRequestEnvelopeV1] = []
    built: list[VerifiedRequestEnvelopeV1] = []

    def capture_normalize(
        request: ReplyRequestV2,
        *,
        adapter_namespace: str,
    ) -> VerifiedRequestEnvelopeV1:
        envelope = normalize_reply_request_v2(
            request,
            adapter_namespace=adapter_namespace,
        )
        normalized.append(envelope)
        return envelope

    async def capture_build(
        envelope: VerifiedRequestEnvelopeV1,
    ) -> ReplyResponse:
        built.append(envelope)
        return reply_response()

    settings = Settings(api_key="secret", deployment_tenant_ref="tenant:test")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "normalize_reply_request_v2", capture_normalize)
    monkeypatch.setattr(main, "build_reply", capture_build)

    response = TestClient(main.app).post(
        "/reply",
        json=make_v2_payload(),
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 200
    assert len(normalized) == 1
    assert built == normalized
    assert normalized[0].state_key.surface == "wecom"
    assert normalized[0].state_key.adapter_namespace == "assistant-wecom"
    assert normalized[0].state_key.tenant_ref == "tenant:test"
    assert normalized[0].state_key.scene == "group"
    assert normalized[0].state_key.subject_ref == "group:group-1"
    assert normalized[0].state_key.principal_ref == "principal:sender-1"
    assert normalized[0].state_key_ref


@pytest.mark.parametrize(
    ("configured_tenant_ref", "expected_status", "expected_code"),
    (
        (None, 503, "deployment_identity_unavailable"),
        ("tenant:other", 403, "deployment_tenant_mismatch"),
    ),
)
def test_v2_feedback_tenant_unavailable_or_mismatch_has_zero_work(
    monkeypatch: pytest.MonkeyPatch,
    configured_tenant_ref: str | None,
    expected_status: int,
    expected_code: str,
) -> None:
    calls: Counter[str] = Counter()
    coordinator = ReplyStateTransactionCoordinatorV1()
    before = coordinator.root_revision()

    def count_key(
        feedback: ActionFeedbackRequestV2,
        *,
        adapter_namespace: str,
    ) -> ConversationStateKey:
        calls["feedback_key"] += 1
        return feedback_state_key_v2(
            feedback,
            adapter_namespace=adapter_namespace,
        )

    def count_prepare(
        state_key: ConversationStateKey,
        request: ActionFeedbackRequestV2,
    ) -> FeedbackReceiptReplayV1:
        calls["prepare"] += 1
        return feedback_replay(state_key, request)

    def get_coordinator() -> ReplyStateTransactionCoordinatorV1:
        calls["coordinator"] += 1
        return coordinator

    monkeypatch.setattr(coordinator, "prepare_feedback", count_prepare)
    monkeypatch.setattr(main, "feedback_state_key_v2", count_key)
    monkeypatch.setattr(main, "get_reply_state_coordinator", get_coordinator)
    route_settings = Settings(
        api_key="route-key",
        deployment_tenant_ref=configured_tenant_ref,
    )
    monkeypatch.setattr(main, "get_settings", lambda: route_settings)
    monkeypatch.setattr(auth, "get_settings", lambda: route_settings)

    result = TestClient(main.app).post(
        "/actions/feedback",
        json=empty_feedback("group").model_dump(mode="json"),
        headers={"X-API-Key": "route-key"},
    )

    assert result.status_code == expected_status
    assert (
        ErrorEnvelope.model_validate_json(result.content).detail.code == expected_code
    )
    assert calls == Counter()
    assert coordinator.root_revision() == before
