from __future__ import annotations

import base64

import pytest

from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_envelope


@pytest.mark.anyio
async def test_public_v2_runtime_reply_replays_without_legacy_attempt_stack() -> None:
    # Given: a direct v2 request with the deterministic V2 handoff path.
    hmac_key = base64.urlsafe_b64encode(b"l" * 32).decode("ascii").rstrip("=")
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            direct_audit_hmac_key=hmac_key,
            reply_alignment_verifier_enabled=False,
        ),
        coordinator=ReplyStateTransactionCoordinatorV1(),
    )
    envelope = make_v2_envelope(
        "T0请人工协助",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:no-legacy",
            "direct_thread_ref": "direct:no-legacy-attempt",
            "principal_ref": "principal:no-legacy",
        },
        presentation={"contract_version": "direct-presentation.v1"},
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    )

    # When: the public runtime executes the request twice.
    response = await runtime.reply(envelope)
    replay = await runtime.reply(envelope)

    # Then: the canonical V2 result is committed and replayed byte-identically.
    assert response.reply.kind == "human_handoff"
    assert replay == response
