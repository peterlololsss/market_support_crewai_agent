from __future__ import annotations

from collections.abc import Mapping
from inspect import signature

from pydantic import JsonValue

from market_support_crewai_agent.runtime.context.stage_inputs import (
    PlannerPromptInputV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.io import (
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.integrations.crewai.retry import (
    RetryPolicy,
    run_with_retry,
)
from market_support_crewai_agent.runtime.observability.runtime_trace import trace_event
from market_support_crewai_agent.runtime.planning.planner_results import (
    PlannerFrameResult,
    planner_frame_from_result,
)
from market_support_crewai_agent.runtime.planning.planner_retry import (
    planner_result_retry_reason,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.context import (
    render_prompt_context_layers,
)
from market_support_crewai_agent.runtime.prompts.direct_provider_client import (
    run_direct_prompt_program,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
    provider_failure_code_from_exception,
)
from market_support_crewai_agent.settings_model import Settings


async def run_planner_kickoff_with_retry(
    planner_agent: CrewAIAgentAdapterV1 | None,
    planner_program: PromptProgram,
    *,
    planner_input: PlannerPromptInputV1,
    timeout_seconds: float | None,
    retry_attempts: int,
    base_delay_seconds: float,
    journal: TurnLlmInvocationJournalV1 | None = None,
    settings: Settings | None = None,
) -> tuple[PlannerFrameResult, list[dict[str, JsonValue]]]:
    strict_runtime = render_prompt_context_layers(planner_input)["runtime"].strip()
    if planner_program.prompt_text.count(strict_runtime) != 1:
        raise ContextViewInvariantError("planner_kickoff_input_not_bound")
    executions: list[dict[str, JsonValue]] = []

    async def call() -> PlannerFrameResult:
        if planner_program.scene_key == "wecom_direct.v1":
            if settings is None:
                raise ContextViewInvariantError("direct_planner_settings_required")
            _validate_direct_prompt_runner(
                prompt_program=planner_program,
                stage_input=planner_input,
                settings=settings,
                journal=journal,
            )
            result, execution = await run_direct_prompt_program(
                prompt_program=planner_program,
                stage_input=planner_input,
                settings=settings,
                journal=journal,
            )
        else:
            _validate_crewai_kickoff_runner(
                planner_agent,
                planner_program,
                timeout_seconds=timeout_seconds,
                journal=journal,
            )
            result, execution = await run_crewai_kickoff(
                planner_agent,
                planner_program,
                timeout_seconds=timeout_seconds,
                journal=journal,
            )
        executions.append(_json_execution_payload(execution))
        return planner_frame_from_result(result)

    try:
        result = await run_with_retry(
            call,
            policy=RetryPolicy(
                retry_attempts=retry_attempts,
                base_delay_seconds=base_delay_seconds,
            ),
            should_retry_result=planner_result_retry_reason,
            should_retry_exception=provider_failure_code_from_exception,
            on_retry=_trace_planner_retry,
        )
    except Exception as exc:
        record_llm_failure(
            planner_agent,
            planner_program.profile.stage,
            provider_failure_code_from_exception(exc),
        )
        raise
    final_reason = planner_result_retry_reason(result)
    if final_reason:
        record_llm_failure(
            planner_agent,
            planner_program.profile.stage,
            "provider_output_missing",
        )
    return result, executions


def _json_execution_payload(
    execution: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    return dict(execution)


def _validate_crewai_kickoff_runner(
    planner_agent: CrewAIAgentAdapterV1 | None,
    planner_program: PromptProgram,
    *,
    timeout_seconds: float | None,
    journal: TurnLlmInvocationJournalV1 | None,
) -> None:
    try:
        _ = signature(run_crewai_kickoff).bind(
            planner_agent,
            planner_program,
            timeout_seconds=timeout_seconds,
            journal=journal,
        )
    except TypeError as exc:
        raise ContextViewInvariantError("crewai_kickoff_signature_invalid") from exc


def _validate_direct_prompt_runner(
    *,
    prompt_program: PromptProgram,
    stage_input: PlannerPromptInputV1,
    settings: Settings,
    journal: TurnLlmInvocationJournalV1 | None,
) -> None:
    try:
        _ = signature(run_direct_prompt_program).bind(
            prompt_program=prompt_program,
            stage_input=stage_input,
            settings=settings,
            journal=journal,
        )
    except TypeError as exc:
        raise ContextViewInvariantError("direct_prompt_signature_invalid") from exc


def _trace_planner_retry(attempt: int, delay_seconds: float, reason: str) -> None:
    trace_event(
        "planner.transient_retry",
        attempt=attempt,
        delay_ms=round(delay_seconds * 1000, 3),
        reason=reason,
    )


def record_llm_failure(
    agent: CrewAIAgentAdapterV1 | None,
    stage: str,
    code: ProviderFailureCodeV1,
) -> None:
    _record_llm_failure_for_agent(agent, stage, code)


def _record_llm_failure_for_agent(
    agent: CrewAIAgentAdapterV1 | None,
    stage: str,
    reason: ProviderFailureCodeV1,
) -> None:
    from market_support_crewai_agent.health.llm_health_hooks import (
        record_llm_failure_for_agent,
    )

    record_llm_failure_for_agent(agent, stage, reason)
