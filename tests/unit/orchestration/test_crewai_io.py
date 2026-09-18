from __future__ import annotations

import os
from dataclasses import replace

import pytest
from pydantic import BaseModel

from market_support_crewai_agent.runtime.context.stage_inputs import (
    build_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAICompletionValueV1,
    CrewAIKickoffOutputV1,
    CrewAITransportInvariantError,
)
from market_support_crewai_agent.runtime.integrations.crewai.io import (
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.planning.planner_llm import (
    run_planner_kickoff_with_retry,
)
from market_support_crewai_agent.runtime.prompts.context import (
    render_prompt_context_layers,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
)
from tests.helpers.crewai_adapter import (
    make_agent_adapter,
    make_completion_agent_adapter,
    make_llm_adapter,
)
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.unit.llm._stage_input_fixtures import stage_sources
from tests.unit.orchestration._crewai_io_support import planner_program, run_async


class ProviderTestError(RuntimeError):
    pass


def test_run_crewai_kickoff_keeps_response_format_for_default_provider() -> None:
    response_formats: list[type[BaseModel]] = []

    def completion(response_format: type[BaseModel]) -> CrewAICompletionValueV1:
        response_formats.append(response_format)
        return "{}"

    agent = make_completion_agent_adapter(completion, model="model")
    _ = run_async(
        lambda: run_crewai_kickoff(agent, planner_program(), timeout_seconds=1)
    )

    assert response_formats == [PlanSpec]


def test_planner_retry_threads_turn_journal_into_real_kickoff() -> None:
    agent = make_completion_agent_adapter(
        lambda _response_format: make_weekly_plan_spec(),
        model="fake-planner",
    )
    planner_source, _, _, _ = stage_sources("请发一下周报")
    planner_input = build_planner_prompt_input_v1(planner_source)
    strict_runtime = render_prompt_context_layers(planner_input)["runtime"].strip()
    program = replace(
        planner_program(), prompt_text=f"<runtime>\n{strict_runtime}\n</runtime>"
    )
    journal = TurnLlmInvocationJournalV1()

    _, executions = run_async(
        lambda: run_planner_kickoff_with_retry(
            agent,
            program,
            planner_input=planner_input,
            timeout_seconds=1,
            retry_attempts=0,
            base_delay_seconds=0,
            journal=journal,
        )
    )

    assert len(executions) == 1
    assert len(journal.rows) == 1
    assert journal.rows[0].status == "success"
    assert journal.rows[0].stage_kind == "planner_intent"
    assert journal.rows[0].prh1 is not None
    assert journal.rows[0].output_digest is not None


def test_run_crewai_kickoff_closes_reservation_when_provider_fails() -> None:
    def fail_provider(_response_format: type[BaseModel]) -> CrewAICompletionValueV1:
        raise ProviderTestError("provider failed")

    agent = make_completion_agent_adapter(
        fail_provider,
        model="fake-planner",
        base_url="https://provider.invalid/v1",
    )
    journal = TurnLlmInvocationJournalV1()

    with pytest.raises(
        ProviderInvocationError,
        match="^provider_transport_unavailable$",
    ) as raised:
        _ = run_async(
            lambda: run_crewai_kickoff(
                agent,
                planner_program(),
                timeout_seconds=1,
                journal=journal,
            )
        )

    assert len(journal.rows) == 1
    assert journal.rows[0].status == "transport_error"
    assert journal.rows[0].error_code == "provider_transport_unavailable"
    assert journal.rows[0].provider_id == "openai_compatible"
    assert raised.value.__cause__ is None


def test_run_crewai_kickoff_classifies_non_string_provider_output() -> None:
    agent = make_completion_agent_adapter(
        lambda _response_format: 7,
        model="fake-planner",
    )
    journal = TurnLlmInvocationJournalV1()

    _ = run_async(
        lambda: run_crewai_kickoff(
            agent,
            planner_program(),
            timeout_seconds=1,
            journal=journal,
        )
    )

    assert journal.rows[0].status == "output_contract_error"
    assert journal.rows[0].error_code == "provider_output_type"


def test_run_crewai_kickoff_stdio_violation_closes_journal() -> None:
    def noisy_provider(_response_format: type[BaseModel]) -> CrewAICompletionValueV1:
        _ = os.write(2, b"crewai-stdio-sentinel")
        return make_weekly_plan_spec()

    agent = make_completion_agent_adapter(noisy_provider, model="fake-planner")
    journal = TurnLlmInvocationJournalV1()

    with pytest.raises(
        ProviderInvocationError,
        match="^direct_provider_stdio_violation$",
    ):
        _ = run_async(
            lambda: run_crewai_kickoff(
                agent,
                planner_program(),
                timeout_seconds=1,
                journal=journal,
            )
        )

    assert journal.rows[0].status == "transport_error"
    assert journal.rows[0].error_code == "direct_provider_stdio_violation"


def test_run_crewai_kickoff_rejects_direct_scene_before_agent_dispatch() -> None:
    dispatched_prompts: list[str] = []

    def forbidden_dispatch(
        prompt: str,
        response_format: type[BaseModel],
    ) -> CrewAIKickoffOutputV1:
        del response_format
        dispatched_prompts.append(prompt)
        return CrewAIKickoffOutputV1(raw="{}", pydantic=None)

    agent = make_agent_adapter(on_prompt=forbidden_dispatch)
    direct_program = replace(
        planner_program(),
        program_id="planner_intent.wecom_direct.v1@1",
        scene_key="wecom_direct.v1",
        scene_contract_id="scene.wecom_direct.planner_intent.v1",
        scene_contract_version="2026-07-15.1",
    )

    with pytest.raises(
        CrewAITransportInvariantError,
        match="^crewai_direct_scene_forbidden$",
    ):
        _ = run_async(
            lambda: run_crewai_kickoff(agent, direct_program, timeout_seconds=1)
        )

    assert dispatched_prompts == []


def test_run_crewai_kickoff_fails_closed_when_capture_is_unavailable() -> None:
    dispatched_prompts: list[str] = []

    def forbidden_dispatch(
        prompt: str,
        response_format: type[BaseModel],
    ) -> CrewAIKickoffOutputV1:
        del response_format
        dispatched_prompts.append(prompt)
        return CrewAIKickoffOutputV1(raw="{}", pydantic=None)

    agent = make_agent_adapter(
        llm=make_llm_adapter(provider="openai", model="fake-planner"),
        on_prompt=forbidden_dispatch,
    )
    journal = TurnLlmInvocationJournalV1()

    with pytest.raises(
        ProviderInvocationError,
        match="^provider_transport_unavailable$",
    ):
        _ = run_async(
            lambda: run_crewai_kickoff(
                agent,
                planner_program(),
                timeout_seconds=1,
                journal=journal,
            )
        )

    assert dispatched_prompts == []
    assert len(journal.rows) == 1
    assert journal.rows[0].status == "transport_error"
    assert journal.rows[0].error_code == "provider_transport_unavailable"
    assert journal.rows[0].prh1 is None
