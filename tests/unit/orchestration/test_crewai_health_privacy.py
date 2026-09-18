from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import BaseModel

from market_support_crewai_agent.health import llm_health
from market_support_crewai_agent.runtime.context.stage_inputs import (
    build_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAICompletionValueV1,
)
from market_support_crewai_agent.runtime.planning.planner_llm import (
    run_planner_kickoff_with_retry,
)
from market_support_crewai_agent.runtime.prompts.context import (
    render_prompt_context_layers,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_completion_agent_adapter
from tests.unit.llm._stage_input_fixtures import stage_sources
from tests.unit.orchestration._crewai_io_support import planner_program, run_async


def test_planner_provider_failure_records_only_closed_health_code(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = Settings(llm_api_key="test-key")
    raw_error = (
        f"provider-secret-canary {settings.llm_base_url} model={settings.llm_model}"
    )
    monitor = llm_health.LlmHealthMonitor(settings, process_health_key=b"h" * 32)
    monkeypatch.setattr(llm_health, "_monitor", monitor)

    def fail_with_secret(_response_format: type[BaseModel]) -> CrewAICompletionValueV1:
        raise RuntimeError(raw_error)

    agent = make_completion_agent_adapter(
        fail_with_secret,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
    )
    planner_source, _, _, _ = stage_sources("请发一下周报")
    planner_input = build_planner_prompt_input_v1(planner_source)
    strict_runtime = render_prompt_context_layers(planner_input)["runtime"].strip()
    program = replace(
        planner_program(), prompt_text=f"<runtime>\n{strict_runtime}\n</runtime>"
    )
    caplog.set_level("DEBUG")

    with pytest.raises(
        ProviderInvocationError,
        match="^provider_transport_unavailable$",
    ) as raised:
        _ = run_async(
            lambda: run_planner_kickoff_with_retry(
                agent,
                program,
                planner_input=planner_input,
                timeout_seconds=1,
                retry_attempts=0,
                base_delay_seconds=0,
            )
        )

    planner_target = next(
        target for target in monitor.targets if target.target_slot == "planner"
    )
    state = monitor.states[planner_target.key]
    exposed = repr(state) + repr(raised.value) + caplog.text
    assert state.last_error == "provider_transport_unavailable"
    assert "provider-secret-canary" not in exposed
    assert settings.llm_base_url not in exposed
    assert settings.llm_model not in exposed
    assert raised.value.__cause__ is None
