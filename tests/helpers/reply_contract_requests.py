from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import JsonValue

from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    VerifiedRequestEnvelopeV1,
)
from tests.helpers.reply_contract_json import JsonInput, json_mapping


def make_v2_payload(
    message: str = "hello",
    **overrides: JsonInput,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "contract_version": "reply-request.v2",
        "request_id": "req:test-message-1",
        "message": message,
        "context_id": "ctx:msg-1",
        "identity": {
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": "tenant:test",
            "group_ref": "group:group-1",
            "principal_ref": "principal:sender-1",
        },
        "presentation": {
            "contract_version": "group-presentation.v1",
            "conversation_name": "test group",
            "principal_name": "test user",
        },
        "business_scope": {
            "kind": "distribution",
            "dist_channel_name": "test channel",
            "channel_type": "bank",
            "available_artifacts": [
                {"type": "material_pack", "options": []},
                {"type": "weekly_report"},
                {"type": "monthly_report"},
            ],
        },
        "grants": {
            "contract_version": "principal-grants.v1",
            "read_capabilities": [
                "resolve_material_pack",
                "resolve_weekly_report",
                "resolve_monthly_report",
                "resolve_sales_mention",
                "query_internal_company_info",
            ],
            "outbound_actions": [
                "send_material_pack",
                "send_weekly_report",
                "send_monthly_report",
            ],
            "mention_types": ["sales"],
        },
    }
    payload.update(json_mapping(overrides))
    return payload


def make_v2_envelope(
    message: str = "hello",
    **overrides: JsonInput,
) -> VerifiedRequestEnvelopeV1:
    from market_support_crewai_agent.runtime.identity import normalize_reply_request_v2
    from market_support_crewai_agent.schemas.conversation import ReplyRequestV2

    return normalize_reply_request_v2(
        ReplyRequestV2.model_validate(make_v2_payload(message, **overrides)),
        adapter_namespace="assistant-wecom",
    )


def make_state_key(
    message: str = "hello",
    **overrides: JsonInput,
) -> ConversationStateKey:
    return make_v2_envelope(message, **overrides).state_key


def assistant_history_with_pending(
    *,
    text: str,
    pending_plan: Mapping[str, JsonInput],
) -> str:
    return json.dumps(
        {
            "contract_version": "reply-runtime-history",
            "reply_response": {
                "contract_version": "reply",
                "response_id": "resp-history",
                "reply": {"kind": "clarification", "text": text, "mentions": []},
                "actions": [],
            },
            "pending_plan": json_mapping(pending_plan),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
