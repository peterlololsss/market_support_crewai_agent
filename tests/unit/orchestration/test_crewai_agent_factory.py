from __future__ import annotations

# pyright: reportPrivateUsage=false
import pytest
from pydantic import JsonValue

from market_support_crewai_agent.runtime.integrations.crewai import (
    agent_factory,
    sdk_adapters,
    sdk_governance,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAILlmAdapterV1,
)
from market_support_crewai_agent.runtime.prompts.profiles import (
    SceneKeyV1,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    AgentExecutionSpecV1,
    PromptProgramV2,
    ProviderIdV1,
    StageKindV1,
    resolve_active_prompt_program_v2,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_agent_adapter, make_llm_adapter


def test_factory_disables_openai_provider_sdk_retries() -> None:
    # Given: the real default OpenAI-compatible planner construction.
    factory = agent_factory.CrewAIAgentFactory(
        Settings(
            llm_api_key="test-key",
            planner_llm_api_key="test-key",
        )
    )

    # When: the production factory creates its CrewAI provider and SDK clients.
    agent = factory.build_planner_agent()

    # Then: the adapted provider remains retry-disabled and capture-capable.
    assert agent.llm.max_retries == 0
    assert agent.llm.configure_openai_client is not None
    assert agent.llm.completion_capture is not None
    assert agent.llm.completion_capture.current() is not None


def test_factory_builds_agent_from_packaged_execution_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a packaged planner spec with a role mutation visible at the adapter boundary.
    program, execution_spec = resolve_active_prompt_program_v2(
        stage="planner_intent",
        scene_key="wecom_group.v1",
    )
    mutated_spec = execution_spec.model_copy(update={"role": "Packaged role"})

    def resolve_program(
        *,
        stage: StageKindV1,
        scene_key: SceneKeyV1,
    ) -> tuple[PromptProgramV2, AgentExecutionSpecV1]:
        del stage, scene_key
        return program, mutated_spec

    monkeypatch.setattr(
        agent_factory,
        "resolve_active_prompt_program_v2",
        resolve_program,
    )

    # When: the real factory path builds the adapter.
    agent = agent_factory.CrewAIAgentFactory(
        Settings(llm_api_key="test-key")
    ).build_planner_agent()

    # Then: the packaged role is preserved at the runtime adapter boundary.
    assert agent.role == "Packaged role"


def test_build_crewai_agent_preserves_governed_construction_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: observable SDK boundaries around the real factory orchestration.
    events: list[str] = []
    llm_inputs: dict[str, JsonValue] = {}
    agent_inputs: dict[str, JsonValue] = {}
    _, execution_spec = resolve_active_prompt_program_v2(
        stage="planner_intent",
        scene_key="wecom_group.v1",
    )
    profile = prompt_profile_by_stage("planner_intent", "generic")
    llm_adapter = make_llm_adapter(provider="openai", model="model")
    agent_adapter = make_agent_adapter(role=execution_spec.role, llm=llm_adapter)

    class FakeLlm:
        timeout: float | None = None
        max_retries: int | None = None

    fake_llm = FakeLlm()

    def llm_constructor(**kwargs: JsonValue) -> FakeLlm:
        events.append("construct_llm")
        llm_inputs.update(kwargs)
        return fake_llm

    def agent_constructor(**kwargs: JsonValue) -> JsonValue:
        events.append("construct_agent")
        agent_inputs.update(
            {key: value for key, value in kwargs.items() if key != "llm"}
        )
        assert kwargs["llm"] is fake_llm
        return None

    def load_types():
        events.append("load_types")
        return agent_constructor, llm_constructor

    def adapt_llm(value: FakeLlm):
        events.append("adapt_llm")
        assert value is fake_llm
        assert value.timeout == 17.0
        assert value.max_retries == 0
        return llm_adapter

    def require_limits(
        _llm: CrewAILlmAdapterV1,
        _provider_id: ProviderIdV1,
        _timeout: float,
    ) -> None:
        events.append("require_limits")

    def adapt_agent(value: JsonValue):
        events.append("adapt_agent")
        assert value is None
        return agent_adapter

    monkeypatch.setattr(sdk_governance, "load_crewai_types_without_dotenv", load_types)
    monkeypatch.setattr(sdk_adapters, "adapt_sdk_llm", adapt_llm)
    monkeypatch.setattr(sdk_governance, "require_provider_limits", require_limits)
    monkeypatch.setattr(sdk_adapters, "adapt_sdk_agent", adapt_agent)

    # When: the orchestration owner constructs one governed SDK agent.
    result = agent_factory.CrewAIAgentFactory(Settings())._build_crewai_agent(
        execution_spec=execution_spec,
        inject_date=True,
        prompt_profile=profile,
        llm_model="model",
        llm_provider="openai",
        llm_base_url="https://provider.invalid/v1",
        llm_api_key="secret",
        llm_timeout_seconds=17.0,
    )

    # Then: sequencing and fixed SDK constructor controls remain exact.
    assert result is agent_adapter
    assert events == [
        "load_types",
        "construct_llm",
        "adapt_llm",
        "require_limits",
        "construct_agent",
        "adapt_agent",
    ]
    assert llm_inputs == {
        "model": "model",
        "provider": "openai",
        "base_url": "https://provider.invalid/v1",
        "api_key": "secret",
        "temperature": 0.1,
        "max_tokens": 6000,
    }
    assert agent_inputs == {
        "role": execution_spec.role,
        "goal": execution_spec.goal,
        "backstory": execution_spec.backstory,
        "system_template": ('{{ .System }}\n{"agent_spec_version":1}'),
        "prompt_template": "{{ .Prompt }}\n" + execution_spec.task_template,
        "response_template": (
            execution_spec.expected_output_template
            + "\n{{ .Response }}\n"
            + execution_spec.expected_output_template
        ),
        "allow_delegation": False,
        "verbose": False,
        "max_iter": 1,
        "max_execution_time": 120,
        "max_retry_limit": 0,
        "planning": False,
        "inject_date": True,
        "date_format": "%Y-%m-%d",
    }
