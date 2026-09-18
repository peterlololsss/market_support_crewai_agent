from __future__ import annotations

import asyncio
import os

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"


from market_support_crewai_agent.runtime.context.stage_inputs import (
    SanitizedAlignmentVerifierInputV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_agents import install_fake_planner
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.helpers.reply_contract_preflight import (
    CapturingEmptyPreflightService,
    CapturingResolvedMaterialPreflight,
    ResolvedMonthlyPreflight,
    ResolvedWeeklyPreflight,
)
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.helpers.reply_contract_runtime import (
    ReplyRequestBuilder,
    failing_crewai_agent,
    install_composer_agent_builder,
    install_planner_agent_builder,
    next_request_id,
    reply_test_settings,
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_runtime_deterministic_action_does_not_call_composer():
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    install_composer_agent_builder(
        runtime,
        lambda _stage: failing_crewai_agent(
            "composer",
            "composer should not run for action responses",
        ),
    )

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("请发周报").payload()))

    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_weekly_report"
    assert response.actions[0].resolve_ref == "weekly:ref"
    assert response.actions[0].period == "20260529"


def test_runtime_direct_weekly_command_bypasses_planner_and_composer():
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )

    install_planner_agent_builder(
        runtime,
        lambda: failing_crewai_agent(
            "planner",
            "LLM stages should not run for direct send commands",
        ),
    )
    install_composer_agent_builder(
        runtime,
        lambda _stage: failing_crewai_agent(
            "composer",
            "LLM stages should not run for direct send commands",
        ),
    )

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("发周报").payload()))

    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_weekly_report"
    assert response.actions[0].resolve_ref == "weekly:ref"


def test_runtime_direct_weekly_command_skips_alignment_verifier():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=True),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )

    class ShouldNotVerify:
        async def verify(
            self,
            input_value: SanitizedAlignmentVerifierInputV1,
        ) -> ReplyAlignmentVerdict:
            del input_value
            raise AssertionError(
                "alignment verifier should not run for direct send commands"
            )

    install_planner_agent_builder(
        runtime,
        lambda: failing_crewai_agent(
            "planner",
            "LLM stages should not run for direct send commands",
        ),
    )
    install_composer_agent_builder(
        runtime,
        lambda _stage: failing_crewai_agent(
            "composer",
            "LLM stages should not run for direct send commands",
        ),
    )
    runtime.alignment_verifier = ShouldNotVerify()

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("发周报").payload()))

    assert response.reply.kind == "answer"
    assert response.actions[0].type == "send_weekly_report"


def test_runtime_direct_monthly_command_bypasses_planner_and_composer():
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedMonthlyPreflight(),
    )

    install_planner_agent_builder(
        runtime,
        lambda: failing_crewai_agent(
            "planner",
            "LLM stages should not run for direct send commands",
        ),
    )
    install_composer_agent_builder(
        runtime,
        lambda _stage: failing_crewai_agent(
            "composer",
            "LLM stages should not run for direct send commands",
        ),
    )

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("发月报").payload()))

    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_monthly_report"
    assert response.actions[0].resolve_ref == "monthly:ref"


def test_runtime_direct_material_command_sends_when_no_options():
    preflight = CapturingResolvedMaterialPreflight()
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=preflight,
    )

    install_planner_agent_builder(
        runtime,
        lambda: failing_crewai_agent(
            "planner",
            "LLM stages should not run for direct send commands",
        ),
    )
    install_composer_agent_builder(
        runtime,
        lambda _stage: failing_crewai_agent(
            "composer",
            "LLM stages should not run for direct send commands",
        ),
    )

    response = asyncio.run(
        runtime.reply(
            make_v2_envelope(
                "发材料包",
                request_id=next_request_id("material-send"),
                business_scope={
                    "kind": "distribution",
                    "dist_channel_name": "test channel",
                    "channel_type": "bank",
                    "available_artifacts": [
                        {"type": "material_pack", "options": []},
                        {"type": "weekly_report"},
                    ],
                },
                grants={
                    "contract_version": "principal-grants.v1",
                    "read_capabilities": ["resolve_material_pack"],
                    "outbound_actions": ["send_material_pack"],
                    "mention_types": [],
                },
            )
        )
    )

    assert preflight.resolve_material_pack_options == {}
    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_material_pack"
    assert response.actions[0].resolve_ref == "material:ref"


def test_runtime_direct_material_command_confirms_when_options_exist():
    preflight = CapturingEmptyPreflightService()
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=preflight,
    )

    install_planner_agent_builder(
        runtime,
        lambda: failing_crewai_agent(
            "planner",
            "LLM stages should not run for direct confirmation",
        ),
    )
    install_composer_agent_builder(
        runtime,
        lambda _stage: failing_crewai_agent(
            "composer",
            "LLM stages should not run for direct confirmation",
        ),
    )

    response = asyncio.run(
        runtime.reply(
            make_v2_envelope(
                "发材料包",
                request_id=next_request_id("material-clarify"),
                business_scope={
                    "kind": "distribution",
                    "dist_channel_name": "test channel",
                    "channel_type": "bank",
                    "available_artifacts": [
                        {
                            "type": "material_pack",
                            "options": [
                                "中证1000指增",
                                "中证A500指增",
                            ],
                        },
                        {"type": "weekly_report"},
                    ],
                },
                grants={
                    "contract_version": "principal-grants.v1",
                    "read_capabilities": ["resolve_material_pack"],
                    "outbound_actions": ["send_material_pack"],
                    "mention_types": [],
                },
            )
        )
    )

    assert preflight.calls == [
        {"resolve_types": [], "resolve_material_pack_options": {}}
    ]
    assert response.reply.kind == "clarification"
    assert response.reply.text == "老师，麻烦确认一下需要哪一类材料，我再继续处理。"
    assert not response.actions
