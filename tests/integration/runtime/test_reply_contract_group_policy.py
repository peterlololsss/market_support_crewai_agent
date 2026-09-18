from __future__ import annotations

import asyncio
import os
from typing import override

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

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
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_agents import install_fake_planner
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.helpers.reply_contract_runtime import next_request_id

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_group_permissions_cannot_be_widened_by_artifacts_or_planner():
    envelope = make_v2_envelope(
        "请帮我确认一下周报是否可用",
        request_id=next_request_id("deny-all"),
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "test channel",
            "channel_type": "bank",
            "available_artifacts": [
                {"type": "weekly_report"},
                {"type": "monthly_report"},
            ],
        },
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": [],
            "outbound_actions": [],
            "mention_types": [],
        },
    )

    class RecordingPreflight(AdapterPreflightService):
        def __init__(self) -> None:
            super().__init__()
            self.calls: int = 0

        @override
        async def collect(
            self,
            request: KernelReplyRequestV1,
            resolve_types: list[AdapterResolveType] | None = None,
            resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
        ) -> AdapterPreflightSnapshot:
            del request, resolve_types, resolve_material_pack_options
            self.calls += 1
            return AdapterPreflightSnapshot.empty()

    preflight = RecordingPreflight()
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        preflight_service=preflight,
    )
    install_fake_planner(runtime, make_weekly_plan_spec(request=envelope.request))

    with pytest.raises(
        AgentRuntimeError, match="plan_spec_capability_not_policy_eligible"
    ):
        _ = asyncio.run(runtime.reply(envelope))

    assert preflight.calls == 0
