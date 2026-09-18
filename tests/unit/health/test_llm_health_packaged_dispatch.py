from __future__ import annotations

# pyright: reportPrivateUsage=false
import logging

import anyio
import pytest
from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.health import llm_health
from market_support_crewai_agent.health.models import LlmHealthTarget
from market_support_crewai_agent.health.probe import health_probe_program
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAICompletionValueV1,
    CrewAIKickoffOutputV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.sdk_adapters import (
    provider_raw_text,
)
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    LlmHealthProbeOutputV1,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import (
    invoke_completion,
    make_agent_adapter,
    make_llm_adapter,
)


def test_health_probe_program_matches_packaged_static_budget_exactly() -> None:
    # Given/When: the registered operational health program is assembled.
    program = health_probe_program()

    # Then: dispatched static instructions are exactly the sealed baseline.
    dispatched_bytes = len(program.prompt_text.encode("utf-8"))
    assert program.program_id == "llm_health_probe.scene_neutral.v1@1"
    assert dispatched_bytes == program.static_bytes
    assert program.static_bytes == program.baseline_bytes == 160
    assert program.profile.id == "llm_health_probe.generic@1"
    assert program.profile.response_model is LlmHealthProbeOutputV1


def test_health_probe_actual_io_log_uses_only_opaque_operational_projection(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a no-network CrewAI-compatible health agent with raw target canaries.
    settings = Settings(
        llm_api_key="health-key",
        llm_model="private-health-model-canary",
        llm_base_url="https://private-health-endpoint-canary.invalid/v1",
    )
    monitor = llm_health.LlmHealthMonitor(
        settings,
        process_health_key=b"h" * 32,
    )
    target = monitor.targets[0]

    output = LlmHealthProbeOutputV1(
        contract_version="llm-health-probe-output.v1",
        ok=True,
    )
    health_agent = _captured_health_agent(
        output,
        model="private-health-model-canary",
    )

    def build_probe_agent(_target: LlmHealthTarget) -> CrewAIAgentAdapterV1:
        return health_agent

    monkeypatch.setattr(monitor, "_build_probe_agent", build_probe_agent)
    caplog.set_level(
        logging.INFO,
        logger="market_support_crewai_agent.runtime.integrations.crewai.observability",
    )

    # When: the production health probe crosses the real governed I/O wrapper.
    anyio.run(monitor._probe_target, target)

    # Then: the emitted row has only opaque operational identity and metrics.
    health_log = "\n".join(
        record.getMessage()
        for record in caplog.records
        if "llm_health_probe" in record.getMessage()
    )
    assert f"target_slot={target.target_slot}" in health_log
    assert f"htk1={target.htk1}" in health_log
    assert "status=success" in health_log
    for forbidden in (
        "hph1=",
        "prh1=",
        "out1=",
        "provider=",
        "model=",
        "url=",
        "private-health-model-canary",
        "private-health-endpoint-canary",
    ):
        assert forbidden not in health_log


def test_health_probe_output_failure_log_uses_closed_error_code(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a no-network health agent returning schema-invalid provider text.
    monitor = llm_health.LlmHealthMonitor(
        Settings(llm_api_key="health-key"),
        process_health_key=b"h" * 32,
    )
    target = monitor.targets[0]

    invalid_health_agent = _captured_health_agent(
        '{"ok":false}',
        model="private-health-model-canary",
    )

    def build_probe_agent(_target: LlmHealthTarget) -> CrewAIAgentAdapterV1:
        return invalid_health_agent

    monkeypatch.setattr(
        monitor,
        "_build_probe_agent",
        build_probe_agent,
    )
    caplog.set_level(
        logging.INFO,
        logger="market_support_crewai_agent.runtime.integrations.crewai.observability",
    )

    # When: the invalid result crosses the governed health boundary.
    anyio.run(monitor._probe_target, target)

    # Then: logging and retained state use the same closed contract code.
    health_log = "\n".join(record.getMessage() for record in caplog.records)
    assert "status=output_contract_error" in health_log
    assert "error_code=provider_output_contract" in health_log
    assert monitor.states[target.key].last_error == "provider_output_contract"
    assert "hph1=" not in health_log
    assert "prh1=" not in health_log
    assert "out1=" not in health_log


def _captured_health_agent(
    completion_result: CrewAICompletionValueV1,
    *,
    model: str,
) -> CrewAIAgentAdapterV1:
    def complete(
        *,
        params: dict[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> CrewAICompletionValueV1:
        del params, available_functions, from_task, from_agent
        if response_model is None:
            raise AssertionError("response_model_missing")
        return completion_result

    llm = make_llm_adapter(
        provider="openai",
        model=model,
        api_key="health-key",
        base_url=None,
        timeout=1.0,
        temperature=0.0,
        max_tokens=64,
        completion=complete,
    )

    def dispatch(
        prompt: str, response_format: type[BaseModel]
    ) -> CrewAIKickoffOutputV1:
        captured = invoke_completion(
            llm,
            params={"model": model, "messages": [{"role": "user", "content": prompt}]},
            response_model=response_format,
        )
        if isinstance(captured, BaseModel):
            return CrewAIKickoffOutputV1(
                pydantic=captured,
                raw=captured.model_dump_json(),
            )
        return CrewAIKickoffOutputV1(pydantic=None, raw=provider_raw_text(captured))

    return make_agent_adapter(role="health", llm=llm, on_prompt=dispatch)
