from __future__ import annotations

import asyncio
import os

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from pydantic import BaseModel

from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from tests.helpers.crewai_adapter import make_completion_agent_adapter
from tests.helpers.reply_contract_agents import FakePlannerAgent, install_fake_planner
from tests.helpers.reply_contract_plan_fixtures import (
    make_monthly_plan_spec,
    make_weekly_plan_spec,
)
from tests.helpers.reply_contract_preflight import (
    CapturingResolvedWeeklyPreflight,
    ResolvedMonthlyPreflight,
    ResolvedWeeklyPreflight,
)
from tests.helpers.reply_contract_requests import (
    assistant_history_with_pending,
    make_v2_envelope,
)
from tests.helpers.reply_contract_runtime import (
    install_planner_agent_builder,
    next_request_id,
    reply_test_settings,
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_runtime_allows_mixed_question_plus_unqualified_monthly_send():
    envelope = make_v2_envelope(
        "在各个策略上的规模是怎么分布呢  然后发我个月报",
        request_id=next_request_id("mixed-monthly"),
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "示例银行",
            "channel_type": "bank",
            "available_artifacts": [
                {
                    "type": "material_pack",
                    "options": ["中证1000指增", "中证A500指增", "中证全指指增"],
                },
                {"type": "weekly_report"},
                {"type": "monthly_report"},
            ],
        },
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["resolve_monthly_report"],
            "outbound_actions": ["send_monthly_report"],
            "mention_types": [],
        },
    )
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedMonthlyPreflight(),
    )
    install_fake_planner(runtime, make_monthly_plan_spec(request=envelope.request))

    response = asyncio.run(runtime.reply(envelope))

    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_monthly_report"
    assert response.actions[0].resolve_ref == "monthly:ref"


def test_runtime_retries_report_query_clarification_as_invalid_plan():
    envelope = make_v2_envelope(
        "[adapter_allowed_read_capabilities: query_internal_company_info]\n周报",
        request_id=next_request_id("weekly-retry"),
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "test channel",
            "channel_type": "bank",
            "available_artifacts": [
                {
                    "type": "material_pack",
                    "options": ["option-a", "option-b", "option-c"],
                },
                {"type": "weekly_report"},
                {"type": "monthly_report"},
            ],
        },
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["resolve_weekly_report"],
            "outbound_actions": ["send_weekly_report"],
            "mention_types": [],
        },
    )
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    prompts: list[str] = []

    def completion(_response_format: type[BaseModel]) -> BaseModel:
        if len(prompts) == 1:
            return make_weekly_plan_spec(
                request=envelope.request,
                ambiguity_slots=["report_query"],
            )
        return make_weekly_plan_spec(request=envelope.request)

    install_planner_agent_builder(
        runtime,
        lambda: make_completion_agent_adapter(
            completion,
            role="planner",
            model="fake-planner",
            on_prompt=prompts.append,
        ),
    )

    response = asyncio.run(runtime.reply(envelope))

    assert response.reply.kind == "clarification"
    assert not response.actions
    assert len(prompts) == 1


def test_runtime_uses_planner_resolved_followup_for_weekly_action():
    envelope = make_v2_envelope(
        "中证1000的",
        request_id=next_request_id("weekly-followup"),
        business_scope={
            "kind": "distribution",
            "dist_channel_name": "test channel",
            "channel_type": "bank",
            "available_artifacts": [
                {
                    "type": "material_pack",
                    "options": ["中证1000指增", "中证A500指增", "中证全指指增"],
                },
                {"type": "weekly_report"},
                {"type": "monthly_report"},
            ],
        },
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["resolve_weekly_report"],
            "outbound_actions": ["send_weekly_report"],
            "mention_types": [],
        },
    )
    store = ConversationStore(max_messages=12)
    store.save_turn(
        envelope.state_key,
        "[adapter_allowed_read_capabilities: query_internal_company_info]\n周报",
        ReplyResponse(
            response_id="resp-old",
            reply=PrimaryReply(kind="clarification", text="我需要再确认一下具体策略。"),
            actions=[],
        ).model_dump_json(exclude_none=True),
    )
    preflight = CapturingResolvedWeeklyPreflight()
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=store,
        preflight_service=preflight,
    )
    install_fake_planner(
        runtime,
        make_weekly_plan_spec(
            request=envelope.request,
            user_need="send prior weekly report request after clarification",
            requested_capabilities=["weekly_report"],
            ambiguity_slots=[],
        ),
    )

    response = asyncio.run(runtime.reply(envelope))

    assert preflight.resolve_material_pack_options == {}
    assert response.reply.kind == "answer"
    assert response.actions[0].type == "send_weekly_report"


def test_planner_prompt_includes_pending_clarification_context():
    envelope = make_v2_envelope(
        "中证1000的",
        request_id=next_request_id("pending-clarification"),
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
            "read_capabilities": ["resolve_weekly_report"],
            "outbound_actions": ["send_weekly_report"],
            "mention_types": [],
        },
    )
    store = ConversationStore(max_messages=12)
    store.save_turn(
        envelope.state_key,
        "[adapter_allowed_read_capabilities: query_internal_company_info]\n周报",
        assistant_history_with_pending(
            text="我需要再确认一下具体策略。",
            pending_plan={
                "artifact_kind": "weekly_report",
                "response_mode": "clarification",
                "ambiguity_slots": ["artifact"],
                "capabilities": ["weekly_report"],
            },
        ),
    )
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=store,
        preflight_service=ResolvedWeeklyPreflight(),
    )
    planner_prompts: list[str] = []
    install_planner_agent_builder(
        runtime,
        lambda: FakePlannerAgent(
            make_weekly_plan_spec(
                request=envelope.request,
                user_need="send clarified weekly report",
            ),
            planner_prompts,
        ),
    )

    response = asyncio.run(runtime.reply(envelope))

    assert response.reply.kind == "answer"
    assert response.actions[0].type == "send_weekly_report"
