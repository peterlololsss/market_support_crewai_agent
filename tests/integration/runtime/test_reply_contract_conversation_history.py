from __future__ import annotations

import asyncio
import os

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"


from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_agents import FakePlannerAgent
from tests.helpers.reply_contract_composer import install_fake_clarification_composer
from tests.helpers.reply_contract_plan_fixtures import make_support_plan_spec
from tests.helpers.reply_contract_preflight import EmptyPreflightService
from tests.helpers.reply_contract_requests import make_state_key
from tests.helpers.reply_contract_runtime import (
    ReplyRequestBuilder,
    coordinator_conversation_texts,
    install_planner_agent_builder,
    reply_test_settings,
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_same_conversation_key_reuses_prior_turns():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=store,
        preflight_service=EmptyPreflightService(),
    )
    prompts: list[str] = []
    install_planner_agent_builder(
        runtime,
        lambda: FakePlannerAgent(
            make_support_plan_spec(),
            prompts,
        ),
    )
    install_fake_clarification_composer(
        runtime,
        text="请确认一下具体需求。",
    )

    first = ReplyRequestBuilder("first question").payload()
    second = ReplyRequestBuilder("follow up").payload()
    _ = asyncio.run(runtime.reply(first))
    _ = asyncio.run(runtime.reply(second))

    history = coordinator_conversation_texts(runtime, first.state_key)
    assert "first question" in history
    assert "follow up" in history


def test_different_conversation_key_does_not_share_history():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=store,
        preflight_service=EmptyPreflightService(),
    )
    prompts: list[str] = []
    install_planner_agent_builder(
        runtime,
        lambda: FakePlannerAgent(
            make_support_plan_spec(),
            prompts,
        ),
    )
    install_fake_clarification_composer(
        runtime,
        text="请确认一下具体需求。",
    )

    first = ReplyRequestBuilder("group one history").payload()
    second = ReplyRequestBuilder(
        "group two current",
        conversation_key="wecom:group-2:sender-1",
        group_id="group-2",
    ).payload()
    _ = asyncio.run(runtime.reply(first))
    _ = asyncio.run(runtime.reply(second))

    first_history = coordinator_conversation_texts(runtime, first.state_key)
    second_history = coordinator_conversation_texts(runtime, second.state_key)
    assert "group one history" in first_history
    assert "group one history" not in second_history
    assert "group two current" in second_history


def test_adapter_execution_history_is_in_planner_prompt():
    store = ConversationStore(max_messages=12)
    ledger = ActionLedger()
    state_key = ReplyRequestBuilder("刚才发了吗").payload().state_key
    _ = ledger.record_feedback(
        ActionFeedbackRequestV2.model_validate(
            {
                "contract_version": "action-feedback.v2",
                "feedback_id": "fb:history-1",
                "request_id": "req:history-1",
                "response_id": "resp-" + "1" * 32,
                "identity": {
                    "contract_version": "conversation-identity.v1",
                    "surface": "wecom",
                    "scene": "group",
                    "tenant_ref": "tenant:test",
                    "group_ref": "group:group-1",
                    "principal_ref": "principal:sender-1",
                },
                "executions": [
                    {
                        "action_type": "send_weekly_report",
                        "status": "executed",
                        "action_id": "act-" + "1" * 32,
                        "artifact": {
                            "type": "weekly_report",
                            "resolve_ref": "weekly:resolve-ref",
                            "artifact_ref": "weekly:artifact-ref",
                            "period": "20260529",
                            "report_date": "2026-05-29",
                        },
                        "adapter_result": {"ok": True, "private": "not prompted"},
                    }
                ],
            }
        ),
        state_key,
    )
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=store,
        action_ledger=ledger,
        preflight_service=EmptyPreflightService(),
    )
    planner_prompts: list[str] = []
    install_planner_agent_builder(
        runtime,
        lambda: FakePlannerAgent(
            prompts=planner_prompts,
        ),
    )
    install_fake_clarification_composer(
        runtime,
        text="请确认一下具体需求。",
    )

    _ = asyncio.run(runtime.reply(ReplyRequestBuilder("刚才发了吗").payload()))

    assert '"recent_executed_actions"' in planner_prompts[0]
    assert '"weekly_report"' in planner_prompts[0]
    assert "weekly:resolve-ref" not in planner_prompts[0]
    assert "not prompted" not in planner_prompts[0]


def test_runtime_bounds_history_in_strict_planner_stage_input():
    store = ConversationStore(max_messages=12)
    huge = "HUGE-HISTORY-CONTENT" * 500
    store.save_turn(make_state_key(), "old user", huge)
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=store,
        preflight_service=EmptyPreflightService(),
    )
    prompts: list[str] = []
    install_planner_agent_builder(
        runtime,
        lambda: FakePlannerAgent(
            make_support_plan_spec(),
            prompts,
        ),
    )
    install_fake_clarification_composer(
        runtime,
        text="请确认一下具体需求。",
    )

    _ = asyncio.run(runtime.reply(ReplyRequestBuilder("current question").payload()))

    assert prompts
    assert '"history"' in prompts[0]
    assert huge not in prompts[0]
