from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Literal

import anyio
from fastapi.testclient import TestClient
from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIKickoffOutputV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.server.main import app
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import invoke_captured_completion, make_llm_adapter
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.helpers.reply_contract_requests import make_v2_envelope

client = TestClient(app, headers={"X-API-Key": "integration-key"})
_TEST_REQUEST_IDS = count(1)


def next_request_id(label: str) -> str:
    return f"req:{label}-{next(_TEST_REQUEST_IDS)}"


@dataclass(frozen=True, slots=True)
class PlannerAgentFactory:
    build_agent: Callable[[], CrewAIAgentAdapterV1]

    def build_planner_agent(self) -> CrewAIAgentAdapterV1:
        return self.build_agent()


@dataclass(frozen=True, slots=True)
class ComposerAgentFactory:
    build_agent: Callable[
        [Literal["knowledge_composer", "smalltalk_composer"]], CrewAIAgentAdapterV1
    ]

    def build_composer_agent(
        self,
        stage: Literal["knowledge_composer", "smalltalk_composer"],
    ) -> CrewAIAgentAdapterV1:
        return self.build_agent(stage)


def failing_crewai_agent(role: str, message: str) -> CrewAIAgentAdapterV1:
    def completion(
        *,
        params: dict[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> JsonValue:
        del params, available_functions, from_task, from_agent, response_model
        return "{}"

    llm = make_llm_adapter(
        provider="openai",
        model=f"fake-{role}",
        completion=completion,
    )

    async def kickoff(
        prompt: str,
        response_format: type[BaseModel],
    ) -> CrewAIKickoffOutputV1:
        del prompt, response_format
        raise AssertionError(message)

    return CrewAIAgentAdapterV1(role=role, llm=llm, kickoff_async=kickoff)


def sleeping_planner_agent(delay_seconds: float) -> CrewAIAgentAdapterV1:
    def completion(
        *,
        params: dict[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> BaseModel:
        del params, available_functions, from_task, from_agent
        if response_model is None:
            raise AssertionError("response_model_missing")
        return make_weekly_plan_spec()

    llm = make_llm_adapter(
        provider="openai", model="fake-planner", completion=completion
    )

    async def kickoff(
        prompt: str,
        response_format: type[BaseModel],
    ) -> CrewAIKickoffOutputV1:
        await anyio.sleep(delay_seconds)
        return invoke_captured_completion(llm, prompt, response_format)

    return CrewAIAgentAdapterV1(role="planner", llm=llm, kickoff_async=kickoff)


def install_planner_agent_builder(
    runtime: CrewAIReplyRuntime,
    builder: Callable[[], CrewAIAgentAdapterV1],
) -> None:
    def build_agent() -> CrewAIAgentAdapterV1:
        return builder()

    runtime.planner_agent_factory = PlannerAgentFactory(build_agent)


def install_composer_agent_builder(
    runtime: CrewAIReplyRuntime,
    builder: Callable[
        [Literal["knowledge_composer", "smalltalk_composer"]], CrewAIAgentAdapterV1
    ],
) -> None:
    def build_agent(
        stage: Literal["knowledge_composer", "smalltalk_composer"],
    ) -> CrewAIAgentAdapterV1:
        return builder(stage)

    runtime.composer_agent_factory = ComposerAgentFactory(build_agent)


def reply_test_settings() -> Settings:
    return Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False)


def invalid_weekly_plan_raw() -> str:
    return (
        make_weekly_plan_spec()
        .model_dump_json()
        .replace(
            '"selected_capability_id":"weekly_report.send",',
            "",
            1,
        )
    )


@dataclass(frozen=True, slots=True)
class ReplyRequestBuilder:
    message: str
    conversation_key: str = "wecom:group-1:sender-1"
    group_id: str = "group-1"
    sender_id: str = "sender-1"
    request_id: str = field(
        default_factory=lambda: f"req:builder-{next(_TEST_REQUEST_IDS)}"
    )

    def payload(self) -> VerifiedRequestEnvelopeV1:
        return make_v2_envelope(
            self.message,
            request_id=self.request_id,
            identity={
                "contract_version": "conversation-identity.v1",
                "surface": "wecom",
                "scene": "group",
                "tenant_ref": "tenant:test",
                "group_ref": f"group:{self.group_id}",
                "principal_ref": f"principal:{self.sender_id}",
            },
        )


def clarification_reply(text: str) -> PrimaryReply:
    return PrimaryReply(kind="clarification", text=text, mentions=[])


def coordinator_conversation_texts(
    runtime: CrewAIReplyRuntime,
    state_key: ConversationStateKey,
) -> list[str]:
    journal = runtime.coordinator.last_journal()
    if journal is None:
        return []
    return [
        turn.text
        for turn in journal.candidate_root.conversation_turns.get(state_key, ())
    ]
