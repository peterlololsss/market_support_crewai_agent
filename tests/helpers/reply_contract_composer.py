from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.schemas.reply import PrimaryReply
from tests.helpers.crewai_adapter import make_completion_agent_adapter
from tests.helpers.reply_contract_runtime import install_composer_agent_builder


def install_fake_clarification_composer(
    runtime: CrewAIReplyRuntime,
    *,
    text: str,
    prompts: list[str] | None = None,
    stages: list[str] | None = None,
) -> None:
    def build_agent(
        stage: Literal[
            "knowledge_composer", "smalltalk_composer"
        ] = "knowledge_composer",
    ) -> CrewAIAgentAdapterV1:
        if stages is not None:
            stages.append(stage)

        def on_prompt(prompt: str) -> None:
            if prompts is not None:
                prompts.append(prompt)

        def completion(response_format: type[BaseModel]) -> BaseModel:
            assert response_format is ComposerReplyOutput
            return ComposerReplyOutput(
                response_mode="clarify",
                missing_inputs=["ambiguity"],
                reply=PrimaryReply(kind="clarification", text=text, mentions=[]),
                actions=[],
            )

        return make_completion_agent_adapter(
            completion,
            role="composer",
            model="fake-composer",
            on_prompt=on_prompt,
        )

    install_composer_agent_builder(runtime, build_agent)
