from __future__ import annotations

import os

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

import anyio
import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.parsing import (
    DocumentEvidenceChunk,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime, build_reply
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.reply import (
    PrimaryReply,
    ReplyResponse,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_json import json_object
from tests.helpers.reply_contract_requests import make_v2_payload
from tests.helpers.reply_contract_runtime import (
    ReplyRequestBuilder,
    client,
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_request_contract_rejects_unknown_available_artifact_type():
    payload = make_v2_payload()
    payload["business_scope"] = {
        **json_object(payload["business_scope"]),
        "available_artifacts": [{"type": "calendar"}],
    }
    response = client.post(
        "/reply",
        json=payload,
    )

    assert response.status_code == 422
    with pytest.raises(ValidationError) as exc_info:
        _ = ReplyRequestV2.model_validate(payload)
    errors = exc_info.value.errors()
    assert any(
        error["loc"]
        == ("business_scope", "distribution", "available_artifacts", 0, "type")
        and error["type"] == "literal_error"
        for error in errors
    )


def test_request_contract_rejects_report_artifact_options():
    payload = make_v2_payload()
    payload["business_scope"] = {
        **json_object(payload["business_scope"]),
        "available_artifacts": [{"type": "weekly_report", "options": ["x"]}],
    }
    response = client.post(
        "/reply",
        json=payload,
    )

    assert response.status_code == 422
    with pytest.raises(ValidationError) as exc_info:
        _ = ReplyRequestV2.model_validate(payload)
    errors = exc_info.value.errors()
    error_text = str(errors)
    assert any(
        error["loc"] == ("business_scope", "distribution", "available_artifacts", 0)
        and error["type"] == "value_error"
        for error in errors
    )
    assert "options are only valid for material_pack" in error_text


def test_request_contract_rejects_removed_trigger_and_session_fields():
    for removed_field in ("session_id", "bot_mentioned", "trigger_reason"):
        payload = make_v2_payload()
        payload[removed_field] = "not allowed"
        response = client.post(
            "/reply",
            json=payload,
        )

        assert response.status_code == 422
        with pytest.raises(ValidationError) as exc_info:
            _ = ReplyRequestV2.model_validate(payload)
        errors = exc_info.value.errors()
        assert any(
            error["loc"] == (removed_field,) and error["type"] == "extra_forbidden"
            for error in errors
        )


def test_reply_response_rejects_unsupported_contract_version():
    from pydantic import ValidationError

    try:
        _ = ReplyResponse.model_validate(
            {
                "contract_version": "reply-versioned",
                "response_id": "resp-unsupported",
                "reply": {"kind": "answer", "text": "ok", "mentions": []},
                "actions": [],
            }
        )
    except ValidationError:
        return

    raise AssertionError("version-suffixed reply contract must not be accepted")


def test_report_action_requires_period_and_report_date():
    from pydantic import ValidationError

    try:
        _ = SendWeeklyReportAction.model_validate(
            {
                "type": "send_weekly_report",
                "action_id": "act-1",
                "resolve_type": "weekly_report",
                "resolve_ref": "weekly:ref",
            }
        )
    except ValidationError:
        return

    raise AssertionError("report send actions must include period and report_date")


def test_report_action_rejects_report_scope_selector():
    from pydantic import ValidationError

    try:
        _ = SendWeeklyReportAction.model_validate(
            {
                "type": "send_weekly_report",
                "action_id": "act-1",
                "resolve_type": "weekly_report",
                "resolve_ref": "weekly:ref",
                "report_scope": "channel_all",
                "period": "20260529",
                "report_date": "2026-05-29",
            }
        )
    except ValidationError:
        return

    raise AssertionError("report send actions must not include report_scope")


def test_build_reply_uses_custom_settings_for_default_runtime_services(
    monkeypatch: pytest.MonkeyPatch,
):
    settings = Settings(
        llm_api_key="test-key",
        doc_mcp_enabled=True,
        doc_mcp_base_url="http://doc-mcp.local:23000",
        agent_conversation_max_messages=3,
        reply_alignment_verifier_enabled=False,
    )
    runtime_settings: list[Settings] = []
    document_settings: list[Settings] = []
    conversation_settings: list[Settings] = []

    class CapturingDocumentClient:
        def __init__(self, settings: Settings) -> None:
            self.settings: Settings = settings
            document_settings.append(settings)

        async def fetch_context_async(
            self,
            request: KernelReplyRequestV1,
            *,
            evidence_query: str,
            cache_authority: DocumentMcpCacheAuthorityV1 | None = None,
        ) -> tuple[DocumentEvidenceChunk, ...]:
            del request, evidence_query, cache_authority
            return ()

    original_from_settings = ConversationStore.from_settings

    def capture_conversation_store(settings_value: Settings) -> ConversationStore:
        conversation_settings.append(settings_value)
        return original_from_settings(settings_value)

    async def fake_reply(
        self: CrewAIReplyRuntime,
        request: KernelReplyRequestV1,
    ) -> ReplyResponse:
        del request
        runtime_settings.append(self.settings)
        return ReplyResponse(
            response_id="resp-ok",
            reply=PrimaryReply(kind="answer", text="ok"),
            actions=[],
        )

    monkeypatch.setattr(
        "market_support_crewai_agent.runtime.turn.DocumentMcpClient",
        CapturingDocumentClient,
    )
    monkeypatch.setattr(ConversationStore, "from_settings", capture_conversation_store)
    monkeypatch.setattr(CrewAIReplyRuntime, "reply", fake_reply)

    response = anyio.run(
        lambda: build_reply(
            ReplyRequestBuilder("介绍一下中证1000").payload(), settings=settings
        )
    )

    assert response.response_id == "resp-ok"
    assert runtime_settings == [settings]
    assert document_settings == [settings]
    assert conversation_settings == [settings]


def test_build_reply_request_contract_rejects_raw_v2_before_dependencies(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {"deps": 0}

    def count_deps() -> None:
        calls["deps"] += 1
        raise AssertionError("runtime dependencies must not be constructed")

    monkeypatch.setattr(
        "market_support_crewai_agent.runtime.service.build_runtime_deps",
        count_deps,
    )
    request = ReplyRequestV2.model_validate(make_v2_payload())

    with pytest.raises(TypeError) as exc_info:
        _ = anyio.run(lambda: build_reply(request))

    assert str(exc_info.value) == "build_reply requires VerifiedRequestEnvelopeV1"
    assert calls == {"deps": 0}
