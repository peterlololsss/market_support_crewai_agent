from __future__ import annotations

import asyncio
import os

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from tests.helpers.reply_contract_agents import install_fake_planner
from tests.helpers.reply_contract_composer import install_fake_clarification_composer
from tests.helpers.reply_contract_plan_fixtures import make_support_plan_spec
from tests.helpers.reply_contract_preflight import EmptyPreflightService
from tests.helpers.reply_contract_runtime import (
    ReplyRequestBuilder,
    reply_test_settings,
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_runtime_clarifies_concrete_artifact_choice():
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="unclear artifact choice",
            artifact_kind="unclear",
            action_intent="none",
            ambiguity_slots=["artifact"],
            requested_capabilities=[],
        ),
    )
    composer_prompts: list[str] = []
    composer_stages: list[str] = []
    install_fake_clarification_composer(
        runtime,
        text="???????????",
        prompts=composer_prompts,
        stages=composer_stages,
    )

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("????").payload()))

    assert response.reply.kind == "clarification"
    assert response.reply.text == "老师，麻烦补充一下具体需求，我再继续处理。"
    assert not response.actions
    assert composer_stages == []
    assert composer_prompts == []
