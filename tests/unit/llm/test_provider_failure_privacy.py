from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, final
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from market_support_crewai_agent.health import llm_health
from market_support_crewai_agent.runtime.context.stage_inputs import (
    build_alignment_verifier_prompt_input_v1,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIKickoffOutputV1,
    CrewAITransportInvariantError,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    InvocationJournalError,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    PromptGovernanceError,
)
from market_support_crewai_agent.runtime.prompts.provider_response_text import (
    DirectProviderTransportError,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    ProviderTargetError,
)
from market_support_crewai_agent.runtime.rendering import composer
from market_support_crewai_agent.runtime.rendering.composer import CrewAIV2Composer
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    build_composer_prompt_input_v1,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.validation.alignment import (
    InternalAlignmentVerifierSourceV1,
    verify_reply_alignment,
)
from market_support_crewai_agent.runtime.validation.alignment_runtime_contracts import (
    AlignmentAgentFactoryV1,
    ComposerAgentFactoryV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerifier,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_llm_adapter
from tests.unit.llm._composer_stage_contract_fixtures import (
    composer_scenario,
    with_static_facts,
)
from tests.unit.llm._stage_input_fixtures import stage_sources


def _failing_provider_agent(
    settings: Settings,
    raw_error: str,
) -> CrewAIAgentAdapterV1:
    async def adapter_kickoff(
        prompt: str,
        response_format: type[BaseModel],
    ) -> CrewAIKickoffOutputV1:
        del prompt, response_format
        raise RuntimeError(raw_error)

    return CrewAIAgentAdapterV1(
        role="failing-provider",
        llm=make_llm_adapter(
            provider=settings.llm_provider,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        ),
        kickoff_async=adapter_kickoff,
    )


@final
class _FailingComposerAgentFactory:
    def __init__(self, agent: CrewAIAgentAdapterV1) -> None:
        self._agent = agent

    def build_composer_agent(
        self,
        stage: Literal["knowledge_composer", "smalltalk_composer"],
    ) -> CrewAIAgentAdapterV1:
        del stage
        return self._agent


@dataclass(frozen=True, slots=True)
class _ComposerRuntime:
    settings: Settings
    composer_agent_factory: ComposerAgentFactoryV1


@final
class _FailingAlignmentAgentFactory:
    def __init__(self, agent: CrewAIAgentAdapterV1) -> None:
        self._agent = agent

    def build_alignment_verifier_agent(self) -> CrewAIAgentAdapterV1:
        return self._agent


@dataclass(frozen=True, slots=True)
class _AlignmentRuntime:
    settings: Settings
    alignment_agent_factory: AlignmentAgentFactoryV1
    alignment_verifier: ReplyAlignmentVerifier | None = None


@pytest.mark.anyio
async def test_composer_provider_failure_exposes_only_closed_code(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(llm_api_key="test-key")
    raw_error = (
        f"composer-secret-canary {settings.llm_base_url} model={settings.llm_model}"
    )
    monitor = llm_health.LlmHealthMonitor(
        settings,
        process_health_key=b"h" * 32,
    )
    monkeypatch.setattr(llm_health, "_monitor", monitor)
    agent = _failing_provider_agent(settings, raw_error)
    scenario = with_static_facts(
        composer_scenario("knowledge_answer"),
        ("company_shareholders",),
    )
    input_value = build_composer_prompt_input_v1(scenario.invocation)
    runtime = _ComposerRuntime(
        settings=settings,
        composer_agent_factory=_FailingComposerAgentFactory(agent),
    )
    caplog.set_level("DEBUG")

    with pytest.raises(
        AgentRuntimeError,
        match="^composer_provider_failure$",
    ) as raised:
        _ = await CrewAIV2Composer(runtime).compose(input_value)

    state = monitor.states[monitor.targets[0].key]
    exposed = repr(state) + repr(raised.value) + caplog.text
    assert state.last_error == "provider_transport_unavailable"
    assert "composer-secret-canary" not in exposed
    assert settings.llm_base_url not in exposed
    assert settings.llm_model not in exposed
    assert raised.value.__cause__ is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure",
    (
        CrewAITransportInvariantError("crewai_dispatch_invalid"),
        DirectProviderTransportError("direct_dispatch_invalid"),
        InvocationJournalError("journal_capacity_invalid"),
        PromptGovernanceError("prompt_authority_invalid"),
        ProviderTargetError("provider_target_invalid"),
    ),
)
async def test_composer_expected_internal_failures_use_closed_code(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    # Given: a typed runtime/configuration failure from the composer boundary.
    settings = Settings(llm_api_key="test-key")
    agent = _failing_provider_agent(settings, "unused")
    scenario = with_static_facts(
        composer_scenario("knowledge_answer"),
        ("company_shareholders",),
    )
    input_value = build_composer_prompt_input_v1(scenario.invocation)
    runtime = _ComposerRuntime(
        settings=settings,
        composer_agent_factory=_FailingComposerAgentFactory(agent),
    )
    monkeypatch.setattr(
        composer,
        "run_composer_kickoff_with_retry",
        AsyncMock(side_effect=failure),
    )

    # When/Then: the expected failure is normalized without exposing details.
    with pytest.raises(AgentRuntimeError, match="^composer_internal_failure$"):
        _ = await CrewAIV2Composer(runtime).compose(input_value)


@pytest.mark.anyio
async def test_alignment_provider_failure_exposes_only_closed_code(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(llm_api_key="test-key")
    raw_error = (
        f"alignment-secret-canary {settings.llm_base_url} model={settings.llm_model}"
    )
    monitor = llm_health.LlmHealthMonitor(
        settings,
        process_health_key=b"h" * 32,
    )
    monkeypatch.setattr(llm_health, "_monitor", monitor)
    agent = _failing_provider_agent(settings, raw_error)
    input_value = build_alignment_verifier_prompt_input_v1(stage_sources()[3])
    runtime = _AlignmentRuntime(
        settings=settings,
        alignment_agent_factory=_FailingAlignmentAgentFactory(agent),
    )
    source = InternalAlignmentVerifierSourceV1(
        model_family="generic",
        prompt_programs=[],
        llm_executions=[],
    )
    caplog.set_level("DEBUG")

    with pytest.raises(
        AgentRuntimeError,
        match="^alignment_verifier_provider_failure$",
    ) as raised:
        _ = await verify_reply_alignment(runtime, input_value, source)

    state = monitor.states[monitor.targets[0].key]
    exposed = repr(state) + repr(raised.value) + caplog.text
    assert state.last_error == "provider_transport_unavailable"
    assert "alignment-secret-canary" not in exposed
    assert settings.llm_base_url not in exposed
    assert settings.llm_model not in exposed
    assert raised.value.__cause__ is None
