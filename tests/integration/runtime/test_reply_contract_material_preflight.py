from __future__ import annotations

import asyncio
import base64
import os

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from typing import override

import pytest

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_agents import install_fake_planner
from tests.helpers.reply_contract_plan_fixtures import make_support_plan_spec
from tests.helpers.reply_contract_preflight import (
    EmptyPreflightService,
    MissingWeeklyWithSalesPreflight,
    resolved_item,
)
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.helpers.reply_contract_runtime import (
    next_request_id,
    reply_test_settings,
)


@pytest.mark.parametrize(
    ("status", "expected_kind", "expected_action_count"),
    [
        ("ambiguous", "clarification", 0),
        ("resolved", "answer", 1),
        ("missing", "unable_to_answer", 0),
    ],
)
def test_material_action_preflight_statuses_produce_distinct_outcomes(
    status: str,
    expected_kind: str,
    expected_action_count: int,
):
    envelope = make_v2_envelope(
        "发一下材料",
        request_id=next_request_id(f"material-status-{status}"),
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "test channel",
            "channel_type": "bank",
            "available_artifacts": [
                {
                    "type": "material_pack",
                    "options": ["中证1000指增", "中证A500指增"],
                },
                {"type": "weekly_report"},
                {"type": "monthly_report"},
            ],
        },
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["resolve_material_pack"],
            "outbound_actions": ["send_material_pack"],
            "mention_types": [],
        },
    )

    class AmbiguousMaterialPreflight(AdapterPreflightService):
        def __init__(self) -> None:
            super().__init__()
            self.snapshot: AdapterPreflightSnapshot | None = None

        @override
        async def collect(
            self,
            request: KernelReplyRequestV1,
            resolve_types: list[AdapterResolveType] | None = None,
            resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
        ) -> AdapterPreflightSnapshot:
            del request, resolve_types, resolve_material_pack_options
            self.snapshot = AdapterPreflightSnapshot(
                items=[
                    resolved_item(
                        "material_pack",
                        status=status,
                        candidates=(
                            ["中证1000指增", "中证A500指增"]
                            if status == "ambiguous"
                            else []
                        ),
                        resolve_ref="material:ref" if status == "resolved" else None,
                        material_pack_option=(
                            "中证1000指增" if status == "resolved" else None
                        ),
                    )
                ]
            )
            return self.snapshot

    preflight = AmbiguousMaterialPreflight()
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=preflight,
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            request=envelope.request,
            selected_capability_id="material_pack.send",
            answerability_policy="send",
            artifact_kind="material_pack",
            action_intent="send",
            ambiguity_slots=[],
        ),
    )

    response = asyncio.run(runtime.reply(envelope))

    assert response.reply.kind == expected_kind
    assert len(response.actions) == expected_action_count
    assert preflight.snapshot is not None
    result = preflight.snapshot.items[0].result
    assert result is not None
    assert result.status == status
    assert result.candidates == (
        ["中证1000指增", "中证A500指增"] if status == "ambiguous" else []
    )


pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_runtime_hands_off_missing_action_when_sales_resolves():
    envelope = make_v2_envelope(
        "请帮我确认一下周报是否可用",
        request_id=next_request_id("sales-handoff"),
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "test channel",
            "channel_type": "bank",
            "available_artifacts": [{"type": "weekly_report"}],
        },
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["resolve_weekly_report", "resolve_sales_mention"],
            "outbound_actions": ["send_weekly_report"],
            "mention_types": ["sales"],
        },
    )
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=MissingWeeklyWithSalesPreflight(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            request=envelope.request,
            artifact_kind="human_support",
            action_intent="handoff",
            ambiguity_slots=[],
        ),
    )

    response = asyncio.run(runtime.reply(envelope))

    assert response.reply.kind == "human_handoff"
    assert response.reply.mentions[0].type == "sales"
    assert not response.actions


def test_direct_handoff_obeys_scene_ceiling_and_empty_grants():
    envelope = make_v2_envelope(
        "T0请人工协助",
        request_id=next_request_id("direct-handoff"),
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:test",
            "direct_thread_ref": "direct:thread-reply-contract",
            "principal_ref": "principal:direct-reply-contract",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "direct user",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": [],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
    audit_key = base64.urlsafe_b64encode(b"d" * 32).decode("ascii").rstrip("=")
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            direct_audit_hmac_key=audit_key,
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )

    response = asyncio.run(runtime.reply(envelope))

    assert response.reply.kind == "human_handoff"
    assert response.reply.mentions == []
    assert not response.actions
