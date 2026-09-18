from __future__ import annotations

import asyncio
import os
from pathlib import Path

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_json import json_object, str_value
from tests.helpers.reply_contract_preflight import EmptyPreflightService
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.helpers.reply_contract_runtime import next_request_id

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_group_t0_handoff_without_sales_preserves_legacy_parity_fixture():
    fixture = TypeAdapter(dict[str, JsonValue]).validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures"
            / "legacy_group_handoff_no_sales.v1.json"
        ).read_text(encoding="utf-8")
    )
    reply_fixture = json_object(fixture["reply"])
    envelope = make_v2_envelope(
        "这个T0可以做吗",
        request_id=next_request_id("group-no-sales"),
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": [],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )

    response = asyncio.run(runtime.reply(envelope))

    assert response.reply.kind == str_value(reply_fixture, "kind", "")
    assert response.reply.text == str_value(reply_fixture, "text", "")
    assert response.reply.mentions == []
    assert fixture["actions"] == []
    assert not response.actions
