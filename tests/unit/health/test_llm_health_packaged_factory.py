from __future__ import annotations

# pyright: reportPrivateUsage=false
from types import SimpleNamespace
from typing import final

import anyio
import pytest
from crewai.llms.base_llm import BaseLLM as CrewAISdkLLM
from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.health import llm_health, probe
from market_support_crewai_agent.runtime.integrations.crewai import sdk_governance
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIProviderResultV1,
    HealthInvocationLogContextV1,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    LlmHealthProbeOutputV1,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    resolve_active_prompt_program_v2,
)
from market_support_crewai_agent.settings_model import Settings


def test_health_probe_dispatch_uses_packaged_agent_profile_and_zero_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the real health factory path with fake CrewAI types and dispatch boundary.
    settings = Settings(
        llm_api_key="health-key",
        llm_health_probe_timeout_seconds=0.5,
    )
    monitor = llm_health.LlmHealthMonitor(settings, process_health_key=b"h" * 32)
    target = monitor.targets[0]
    captured: dict[
        str,
        CrewAIAgentAdapterV1 | PromptProgram | HealthInvocationLogContextV1 | float,
    ] = {}
    constructed: dict[str, JsonValue] = {}
    _, llm_cls = sdk_governance.load_crewai_types_without_dotenv()

    @final
    class FakeHealthSdkAgent:
        def __init__(
            self,
            *,
            role: str,
            goal: str,
            backstory: str,
            llm: CrewAISdkLLM,
            allow_delegation: bool,
            max_retry_limit: int,
            planning: bool,
            inject_date: bool,
            **_unused: JsonValue,
        ) -> None:
            constructed.update(
                role=role,
                goal=goal,
                backstory=backstory,
                allow_delegation=allow_delegation,
                max_retry_limit=max_retry_limit,
                planning=planning,
                inject_date=inject_date,
            )
            self.role = role
            self.llm = llm
            self.allow_delegation = allow_delegation
            self.max_retry_limit = max_retry_limit
            self.planning = planning
            self.inject_date = inject_date

        async def kickoff_async(
            self,
            messages: str,
            response_format: type[BaseModel] | None = None,
        ) -> SimpleNamespace:
            del messages, response_format
            output = LlmHealthProbeOutputV1(
                contract_version="llm-health-probe-output.v1",
                ok=True,
            )
            return SimpleNamespace(raw=output.model_dump_json(), pydantic=output)

    monkeypatch.setattr(
        sdk_governance,
        "load_crewai_types_without_dotenv",
        lambda: (FakeHealthSdkAgent, llm_cls),
    )

    async def fake_run_crewai_kickoff(
        agent: CrewAIAgentAdapterV1,
        program: PromptProgram,
        *,
        timeout_seconds: float,
        health_log_context: HealthInvocationLogContextV1,
    ) -> tuple[CrewAIProviderResultV1, dict[str, JsonValue]]:
        captured.update(
            agent=agent,
            program=program,
            timeout_seconds=timeout_seconds,
            health_log_context=health_log_context,
        )
        output = LlmHealthProbeOutputV1(
            contract_version="llm-health-probe-output.v1",
            ok=True,
        )
        return (
            CrewAIProviderResultV1(
                raw=output.model_dump_json(),
                pydantic=output,
                agent_role=agent.role,
                usage_metrics=None,
            ),
            {},
        )

    monkeypatch.setattr(probe, "run_crewai_kickoff", fake_run_crewai_kickoff)

    # When: production health code performs one no-network probe dispatch.
    anyio.run(monitor._probe_target, target)

    # Then: the packaged health identity/profile and bounded settings reach dispatch.
    _, expected_spec = resolve_active_prompt_program_v2(
        stage="llm_health_probe",
        scene_key="scene_neutral.v1",
    )
    dispatched_agent = captured["agent"]
    dispatched_program = captured["program"]
    assert isinstance(dispatched_agent, CrewAIAgentAdapterV1)
    assert dispatched_agent.role == expected_spec.role
    assert constructed["goal"] == expected_spec.goal
    assert constructed["backstory"] == expected_spec.backstory
    assert dispatched_agent.inject_date is False
    assert dispatched_agent.max_retry_limit == 0
    assert dispatched_agent.llm.temperature == 0.0
    assert dispatched_agent.llm.max_tokens == 64
    assert dispatched_agent.llm.timeout == 0.5
    assert isinstance(dispatched_program, PromptProgram)
    assert dispatched_program.profile.id == "llm_health_probe.generic@1"
    assert dispatched_program.profile.response_model is LlmHealthProbeOutputV1
    assert len(dispatched_program.prompt_text.encode("utf-8")) == 160
    assert captured["timeout_seconds"] == 0.5
    assert monitor.states[target.key].status == "healthy"
