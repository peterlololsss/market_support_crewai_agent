from __future__ import annotations

import hashlib
from dataclasses import replace

from market_support_crewai_agent.runtime.context.retry_view_models import (
    PlannerRetryOverlayV1,
    PlanValidationIssueViewV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.planning import PlanValidationResult
from market_support_crewai_agent.runtime.planning.planner_results import (
    PlannerFrameResult,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)


def validation_error_summary(validation: PlanValidationResult) -> str:
    return "; ".join(issue.code for issue in validation.issues) or "unknown"


def planner_schema_repair_overlay(error_summary: str) -> PlannerRetryOverlayV1:
    message = _retry_text(error_summary, "plan_spec_schema_invalid")
    return PlannerRetryOverlayV1(
        attempt=1,
        phase="schema_repair",
        phase_attempt=1,
        issues=(
            PlanValidationIssueViewV1(
                code="plan_spec_schema_invalid",
                message=message,
            ),
        ),
        feedback=message,
    )


def planner_schema_repair_allowed(overlay: PlannerRetryOverlayV1 | None) -> bool:
    return overlay is None


def planner_alignment_replan_overlay(
    verdict: ReplyAlignmentVerdict | None,
    phase_attempt: int,
) -> PlannerRetryOverlayV1 | None:
    if verdict is None and phase_attempt == 0:
        return None
    if (
        verdict is None
        or verdict.remediation != "replan"
        or phase_attempt not in (1, 2)
    ):
        raise ContextViewInvariantError("planner_alignment_replan_mapping_invalid")
    message = _retry_text(
        verdict.planner_feedback or verdict.rationale,
        "alignment_requested_replan",
    )
    return PlannerRetryOverlayV1(
        attempt=phase_attempt + 1,
        phase="alignment_replan",
        phase_attempt=phase_attempt,
        issues=(
            PlanValidationIssueViewV1(
                code="plan_spec_manifest_contract_mismatch",
                message=message,
            ),
        ),
        feedback=message,
    )


def planner_result_retry_reason(result: PlannerFrameResult) -> str | None:
    if result.pydantic is None and not result.raw.strip():
        return "empty_output"
    return None


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


def _retry_text(value: str, fallback: str) -> str:
    normalized = " ".join(value.split())
    return (normalized or fallback)[:300]
