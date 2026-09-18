from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Literal, final

import pytest
from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.identity import (
    VerifiedRequestEnvelopeV1,
    normalize_reply_request_v2,
)
from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
)
from market_support_crewai_agent.runtime.observability.direct_audit import (
    decode_direct_audit_hmac_key,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.server import adapter_compatibility, auth, main
from market_support_crewai_agent.settings_model import Settings
from tests.unit.server.scene_admission_support import (
    ErrorEnvelope,
    direct_request,
    reply_response,
)


def _audit_key() -> str:
    return base64.urlsafe_b64encode(b"a" * 32).decode("ascii").rstrip("=")


def _clear_compatibility_cache() -> None:
    clear_cache: Callable[[], None] = (
        adapter_compatibility.clear_compatibility_client_cache_for_testing
    )
    clear_cache()


@pytest.mark.parametrize(
    ("settings", "compatible", "expected_code", "expected_events"),
    (
        (
            Settings(
                api_key="secret",
                deployment_tenant_ref="tenant:test",
                internal_dm_enabled=False,
            ),
            True,
            "internal_dm_disabled",
            [],
        ),
        (
            Settings(
                api_key="secret",
                deployment_tenant_ref="tenant:test",
                internal_dm_enabled=True,
                adapter_api_key="adapter-secret",
            ),
            True,
            "internal_dm_audit_unavailable",
            ["audit"],
        ),
        (
            Settings(
                api_key="secret",
                deployment_tenant_ref="tenant:test",
                internal_dm_enabled=True,
                direct_audit_hmac_key="invalid",
                adapter_api_key="adapter-secret",
            ),
            True,
            "internal_dm_audit_unavailable",
            ["audit"],
        ),
        (
            Settings(
                api_key="secret",
                deployment_tenant_ref="tenant:test",
                internal_dm_enabled=True,
                direct_audit_hmac_key=_audit_key(),
            ),
            True,
            "internal_dm_adapter_auth_required",
            ["audit"],
        ),
        (
            Settings(
                api_key="secret",
                deployment_tenant_ref="tenant:test",
                internal_dm_enabled=True,
                direct_audit_hmac_key=_audit_key(),
                adapter_api_key="adapter-secret",
            ),
            False,
            "internal_dm_adapter_incompatible",
            ["audit", "factory", "compatibility"],
        ),
    ),
)
def test_direct_admission_failures_stop_before_later_work(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    compatible: bool,
    expected_code: str,
    expected_events: list[str],
) -> None:
    events: list[str] = []

    @final
    class CompatibilityClient:
        def assert_scene_compatible(
            self,
            scene: Literal["direct", "group"],
            tenant_ref: str,
        ) -> None:
            assert (scene, tenant_ref) == ("direct", "tenant:test")
            events.append("compatibility")
            if not compatible:
                raise AdapterClientError("incompatible")

    def capture_audit(key: str) -> bytes:
        events.append("audit")
        return decode_direct_audit_hmac_key(key)

    def compatibility_factory() -> CompatibilityClient:
        events.append("factory")
        return CompatibilityClient()

    async def build_trap(
        _envelope: VerifiedRequestEnvelopeV1,
    ) -> ReplyResponse:
        events.append("build")
        return reply_response()

    _clear_compatibility_cache()
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "decode_direct_audit_hmac_key", capture_audit)
    monkeypatch.setattr(
        adapter_compatibility, "new_compatibility_client", compatibility_factory
    )
    monkeypatch.setattr(main, "build_reply", build_trap)

    response = TestClient(main.app).post(
        "/reply",
        json=direct_request().model_dump(mode="json"),
        headers={"X-API-Key": "secret"},
    )

    assert response.status_code == 503
    assert (
        ErrorEnvelope.model_validate_json(response.content).detail.code == expected_code
    )
    assert events == expected_events
    _clear_compatibility_cache()


def test_enabled_internal_dm_admission_reuses_one_process_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    settings = Settings(
        api_key="secret",
        deployment_tenant_ref="tenant:test",
        internal_dm_enabled=True,
        direct_audit_hmac_key=_audit_key(),
        adapter_api_key="adapter-secret",
    )

    @final
    class CompatibilityClient:
        def assert_scene_compatible(
            self,
            scene: Literal["direct", "group"],
            tenant_ref: str,
        ) -> None:
            assert (scene, tenant_ref) == ("direct", "tenant:test")
            events.append("compatibility")

    def compatibility_factory() -> CompatibilityClient:
        events.append("factory")
        return CompatibilityClient()

    def capture_audit(key: str) -> bytes:
        events.append("audit")
        return decode_direct_audit_hmac_key(key)

    def capture_normalize(
        request: ReplyRequestV2,
        *,
        adapter_namespace: str,
    ) -> VerifiedRequestEnvelopeV1:
        events.append("normalize")
        return normalize_reply_request_v2(
            request,
            adapter_namespace=adapter_namespace,
        )

    async def capture_build(
        _envelope: VerifiedRequestEnvelopeV1,
    ) -> ReplyResponse:
        events.append("build")
        return reply_response()

    _clear_compatibility_cache()
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "decode_direct_audit_hmac_key", capture_audit)
    monkeypatch.setattr(
        adapter_compatibility, "new_compatibility_client", compatibility_factory
    )
    monkeypatch.setattr(main, "normalize_reply_request_v2", capture_normalize)
    monkeypatch.setattr(main, "build_reply", capture_build)

    client = TestClient(main.app)
    first = client.post(
        "/reply",
        json=direct_request().model_dump(mode="json"),
        headers={"X-API-Key": "secret"},
    )
    second = client.post(
        "/reply",
        json=direct_request().model_dump(mode="json"),
        headers={"X-API-Key": "secret"},
    )

    assert (first.status_code, second.status_code) == (200, 200)
    assert events == [
        "audit",
        "factory",
        "compatibility",
        "normalize",
        "build",
        "audit",
        "compatibility",
        "normalize",
        "build",
    ]
    _clear_compatibility_cache()
