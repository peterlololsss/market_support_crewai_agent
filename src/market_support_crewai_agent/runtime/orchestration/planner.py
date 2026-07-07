from __future__ import annotations

from pydantic import JsonValue
import hashlib
import logging
from dataclasses import replace

from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
    validate_execution_plan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.llm.retry import RetryPolicy, run_with_retry
from market_support_crewai_agent.runtime.orchestration.crewai_io import (
    coerce_planner_plan,
    plan_spec_error_summary,
    run_crewai_kickoff,
    safe_short_text,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.state.runtime_trace import trace_event
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings

logger = logging.getLogger(__name__)


def validation_error_summary(validation: PlanValidationResult) -> str:
    return "; ".join(issue.code for issue in validation.issues) or "unknown"


async def run_planner_kickoff_with_retry(
    planner_agent,
    planner_program: PromptProgram,
    *,
    timeout_seconds: float | None,
    retry_attempts: int,
    base_delay_seconds: float,
) -> tuple[object, list[dict[str, JsonValue]]]:
    executions: list[dict[str, JsonValue]] = []

    async def call():
        result, execution = await run_crewai_kickoff(
            planner_agent,
            planner_program,
            timeout_seconds=timeout_seconds,
        )
        executions.append(execution)
        return result

    try:
        result = await run_with_retry(
            call,
            policy=RetryPolicy(
                retry_attempts=retry_attempts,
                base_delay_seconds=base_delay_seconds,
            ),
            should_retry_result=_planner_result_retry_reason,
            should_retry_exception=lambda exc: (
                safe_short_text(exc) or "planner_call_failed"
            ),
            on_retry=_trace_planner_retry,
        )
    except Exception as exc:
        record_llm_failure(
            planner_agent,
            planner_program.profile.stage,
            safe_short_text(exc) or "planner_call_failed",
        )
        raise
    final_reason = _planner_result_retry_reason(result)
    if final_reason:
        record_llm_failure(
            planner_agent,
            planner_program.profile.stage,
            f"{final_reason}_after_retry",
        )
    return result, executions


def _planner_result_retry_reason(result) -> str | None:
    if (
        getattr(result, "pydantic", None) is None
        and not str(getattr(result, "raw", "") or "").strip()
    ):
        return "empty_output"
    return None


def is_empty_planner_result(result) -> bool:
    return _planner_result_retry_reason(result) == "empty_output"


def can_fallback_to_default_planner(settings: Settings) -> bool:
    planner_is_gemini = settings.planner_llm_provider.lower() in {"gemini", "google"}
    planner_differs = (
        settings.planner_llm_provider != settings.llm_provider
        or settings.planner_llm_model != settings.llm_model
        or settings.planner_llm_base_url != settings.llm_base_url
    )
    return planner_is_gemini and planner_differs and bool(settings.llm_api_key)


def _trace_planner_retry(attempt: int, delay_seconds: float, reason: str) -> None:
    trace_event(
        "planner.transient_retry",
        attempt=attempt,
        delay_ms=round(delay_seconds * 1000, 3),
        reason=reason,
    )


def coerce_planner_plan_with_error(
    frame_result,
    request: ReplyRequest,
    policy: PolicyManifest,
    *,
    domain_context: DomainContext,
    history: list[ConversationMessage],
) -> tuple[ExecutionPlan | None, str]:
    try:
        plan = coerce_planner_plan(
            frame_result,
            request,
            policy,
            domain_context=domain_context,
            history=history,
        )
    except ValueError as exc:
        return None, f"PlanSpec compile error: {exc}"
    if plan is not None:
        validation = validate_execution_plan(plan, policy)
        if not validation.valid:
            return (
                None,
                "ExecutionPlan validation error: "
                f"{validation_error_summary(validation)}",
            )
        return plan, ""
    return None, plan_spec_error_summary(frame_result)


def planner_retry_program(program: PromptProgram, error_summary: str) -> PromptProgram:
    feedback = (
        '\n\n<prompt_layer id="ephemeral">\n'
        "Previous PlanSpec validation error:\n"
        f"{error_summary}\n\n"
        "Rewrite the full PlanSpec JSON only. Do not output explanations. "
        "Fix the listed contract errors and keep each plan_units item aligned "
        "with its selected capability.\n"
        "</prompt_layer>"
    )
    prompt_text = program.prompt_text + feedback
    layers = program.layers
    if "ephemeral" not in layers:
        layers = (*layers, "ephemeral")
    return replace(
        program,
        prompt_text=prompt_text,
        prompt_hash="sha256:" + hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        layers=layers,
    )


def record_llm_failure(agent, stage: str, reason: str) -> None:
    try:
        from market_support_crewai_agent.health.llm_health import (
            record_llm_failure_for_agent,
        )

        record_llm_failure_for_agent(agent, stage, reason)
    except Exception:
        logger.debug("LLM health failure hook failed", exc_info=True)
