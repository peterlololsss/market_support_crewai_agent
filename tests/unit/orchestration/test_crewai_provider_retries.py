from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn

import pytest
from pydantic import JsonValue

from market_support_crewai_agent.runtime.integrations.crewai import sdk_governance
from market_support_crewai_agent.runtime.integrations.crewai.agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    LlmHealthProbeOutputV1,
)
from market_support_crewai_agent.runtime.prompts.neutral_program import (
    assemble_neutral_prompt_program,
)
from market_support_crewai_agent.runtime.prompts.profiles import PromptProfile
from market_support_crewai_agent.settings_model import Settings

AgentBuilder = Callable[[CrewAIAgentFactory], CrewAIAgentAdapterV1]


def _health_profile() -> PromptProfile:
    return assemble_neutral_prompt_program(
        stage="llm_health_probe",
        response_model=LlmHealthProbeOutputV1,
        temperature=0.0,
        max_tokens=64,
    ).profile


def _planner(factory: CrewAIAgentFactory) -> CrewAIAgentAdapterV1:
    return factory.build_planner_agent()


def _knowledge_composer(factory: CrewAIAgentFactory) -> CrewAIAgentAdapterV1:
    return factory.build_composer_agent()


def _smalltalk_composer(factory: CrewAIAgentFactory) -> CrewAIAgentAdapterV1:
    return factory.build_composer_agent("smalltalk_composer")


def _alignment_verifier(factory: CrewAIAgentFactory) -> CrewAIAgentAdapterV1:
    return factory.build_alignment_verifier_agent()


def _health_composer(factory: CrewAIAgentFactory) -> CrewAIAgentAdapterV1:
    return factory.build_health_probe_agent(
        target_slot="composer",
        prompt_profile=_health_profile(),
    )


def _health_planner(factory: CrewAIAgentFactory) -> CrewAIAgentAdapterV1:
    return factory.build_health_probe_agent(
        target_slot="planner",
        prompt_profile=_health_profile(),
    )


_BUILDERS: tuple[AgentBuilder, ...] = (
    _planner,
    _knowledge_composer,
    _smalltalk_composer,
    _alignment_verifier,
    _health_composer,
    _health_planner,
)


@pytest.mark.parametrize(
    ("provider", "model"),
    (("openai", "deepseek-v4-pro"), ("deepseek", "deepseek-chat")),
)
@pytest.mark.parametrize("build_agent", _BUILDERS)
def test_factory_disables_openai_sdk_retries_for_every_real_crewai_target(
    provider: str,
    model: str,
    build_agent: AgentBuilder,
) -> None:
    settings = Settings(
        llm_provider=provider,
        llm_model=model,
        llm_api_key="test-key",
        planner_llm_provider=provider,
        planner_llm_model=model,
        planner_llm_api_key="test-key",
    )

    agent = build_agent(CrewAIAgentFactory(settings))

    assert agent.llm.max_retries == 0


@pytest.mark.parametrize(
    ("provider", "model"),
    (("gemini", "gemini-3-flash-preview"), ("google", "gemini-3-flash-preview")),
)
@pytest.mark.parametrize("build_agent", _BUILDERS)
def test_factory_disables_gemini_sdk_retries_for_every_real_crewai_target(
    provider: str,
    model: str,
    build_agent: AgentBuilder,
) -> None:
    settings = Settings(
        llm_provider=provider,
        llm_model=model,
        llm_api_key="test-key",
        planner_llm_provider=provider,
        planner_llm_model=model,
        planner_llm_api_key="test-key",
    )

    agent = build_agent(CrewAIAgentFactory(settings))

    assert agent.llm.gemini_retry_attempts == 1


def test_factory_rejects_unverified_provider_retry_configuration_before_agent_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_constructions: list[bool] = []

    class UnsupportedRetryLlm:
        provider: str = "openai"

    def unsupported_llm(**_kwargs: JsonValue) -> UnsupportedRetryLlm:
        return UnsupportedRetryLlm()

    def agent_constructor(**_kwargs: JsonValue) -> NoReturn:
        agent_constructions.append(True)
        raise AssertionError("agent_construction_must_not_run")

    def load_types() -> tuple[
        Callable[..., NoReturn],
        Callable[..., UnsupportedRetryLlm],
    ]:
        return agent_constructor, unsupported_llm

    monkeypatch.setattr(
        sdk_governance,
        "load_crewai_types_without_dotenv",
        load_types,
    )

    with pytest.raises(
        ValueError,
        match="crewai_provider_retry_configuration_unsupported",
    ):
        _ = CrewAIAgentFactory(Settings()).build_planner_agent()
    assert agent_constructions == []
