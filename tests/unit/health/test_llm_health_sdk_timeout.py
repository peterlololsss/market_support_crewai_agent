from __future__ import annotations

# pyright: reportPrivateUsage=false
from types import SimpleNamespace

import anyio
import pytest
from google import genai
from google.genai import types
from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.health.llm_health import LlmHealthMonitor
from market_support_crewai_agent.health.models import LlmHealthTarget
from market_support_crewai_agent.health.probe import health_probe_program
from market_support_crewai_agent.runtime.integrations.crewai import sdk_governance
from market_support_crewai_agent.runtime.integrations.crewai.agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.sdk_payload import (
    mapping_value,
    optional_int_value,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    PromptGovernanceError,
)
from market_support_crewai_agent.settings_model import Settings


@pytest.mark.parametrize("provider", ("gemini", "google"))
def test_health_gemini_sdk_uses_health_timeout_override(provider: str) -> None:
    # Given: health and normal reply timeouts that cannot be confused.
    factory = CrewAIAgentFactory(
        Settings(
            llm_provider=provider,
            llm_model="gemini-3-flash-preview",
            llm_api_key="test-key",
            llm_timeout_seconds=17.0,
            llm_health_probe_timeout_seconds=0.5,
        )
    )

    # When: the real CrewAI Gemini health provider is constructed.
    agent = factory.build_health_probe_agent(
        target_slot="composer",
        prompt_profile=health_probe_program().profile,
    )

    # Then: its typed Gemini client config carries the health limit in milliseconds.
    assert agent.llm.gemini_retry_attempts == 1
    health_options = mapping_value(dict(agent.llm.client_params), "http_options")
    assert optional_int_value(health_options, "timeout") == 500
    assert agent.llm.gemini_sync_client_factory is not None


def test_health_gemini_http_request_uses_health_timeout_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a real health agent whose Gemini client construction is observed.
    settings = Settings(
        llm_provider="gemini",
        llm_model="gemini-3-flash-preview",
        llm_api_key="test-key",
        llm_base_url="https://provider.invalid",
        llm_timeout_seconds=17.0,
        llm_health_probe_timeout_seconds=0.5,
    )
    monitor = LlmHealthMonitor(settings, process_health_key=b"h" * 32)
    target = monitor.targets[0]
    agent = monitor._build_probe_agent(target)
    client_timeouts: list[int | None] = []

    class FakeGeminiModels:
        def generate_content(
            self,
            *,
            model: str,
            contents: str,
            config: BaseModel,
        ) -> SimpleNamespace:
            del model, contents, config
            return SimpleNamespace(
                text=('{"contract_version":"llm-health-probe-output.v1","ok":true}'),
                usage_metadata=None,
            )

    class FakeGeminiClient:
        models: FakeGeminiModels

        def __init__(
            self,
            *,
            api_key: str | None,
            http_options: types.HttpOptions,
            **_unused: JsonValue,
        ) -> None:
            del api_key
            client_timeouts.append(http_options.timeout)
            self.models = FakeGeminiModels()

    def build_probe_agent(_target: LlmHealthTarget) -> CrewAIAgentAdapterV1:
        return agent

    monkeypatch.setattr(genai, "Client", FakeGeminiClient)
    monkeypatch.setattr(monitor, "_build_probe_agent", build_probe_agent)

    # When: the production health path performs one real Gemini wrapper dispatch.
    anyio.run(monitor._probe_target, target)

    # Then: the client receives 0.5 seconds as 500 ms, not the 17-second reply limit.
    assert client_timeouts == [500]
    assert monitor.states[target.key].status == "healthy"


def test_normal_gemini_sdk_preserves_general_reply_timeout() -> None:
    # Given: a normal reply factory with distinct general and health limits.
    factory = CrewAIAgentFactory(
        Settings(
            llm_provider="gemini",
            llm_model="gemini-3-flash-preview",
            llm_api_key="test-key",
            llm_timeout_seconds=17.0,
            llm_health_probe_timeout_seconds=0.5,
        )
    )

    # When: the normal composer provider is constructed.
    agent = factory.build_composer_agent()

    # Then: its typed Gemini client config keeps the general reply limit.
    assert agent.llm.gemini_retry_attempts == 1
    reply_options = mapping_value(dict(agent.llm.client_params), "http_options")
    assert optional_int_value(reply_options, "timeout") == 17_000
    assert agent.llm.gemini_sync_client_factory is not None


def test_factory_rejects_stale_gemini_sdk_timeout_before_agent_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a provider that reports a stale SDK timeout.
    agent_constructions = 0

    class MisleadingLlm:
        timeout: float | None = None
        max_retries: int | None = None

        def model_dump(self, *, mode: str) -> dict[str, JsonValue]:
            assert mode == "json"
            return {
                "provider": "gemini",
                "model": "gemini-3-flash-preview",
                "api_key": "test-key",
                "client_params": {
                    "http_options": {
                        "retry_options": {"attempts": 1},
                        "timeout": 17_000,
                    }
                },
            }

    def misleading_llm(**_kwargs: JsonValue) -> MisleadingLlm:
        return MisleadingLlm()

    def agent_constructor(**_kwargs: JsonValue) -> None:
        nonlocal agent_constructions
        agent_constructions += 1

    def load_types():
        return agent_constructor, misleading_llm

    monkeypatch.setattr(
        sdk_governance,
        "load_crewai_types_without_dotenv",
        load_types,
    )
    factory = CrewAIAgentFactory(
        Settings(
            llm_provider="gemini",
            llm_model="gemini-3-flash-preview",
            llm_api_key="test-key",
            llm_timeout_seconds=17.0,
            llm_health_probe_timeout_seconds=0.5,
        )
    )

    # When/Then: governance fails closed before constructing an executable Agent.
    with pytest.raises(
        PromptGovernanceError,
        match="crewai_provider_sdk_timeout_unsupported",
    ):
        _ = factory.build_health_probe_agent(
            target_slot="composer",
            prompt_profile=health_probe_program().profile,
        )
    assert agent_constructions == 0
