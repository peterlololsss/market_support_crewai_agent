from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, JsonValue, ValidationError

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIProviderResultV1,
    ProviderRawTextV1,
)
from market_support_crewai_agent.runtime.planning import (
    ExecutionPlanV2,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.prompts.provider_response_text import (
    DirectPromptProgramResultV1,
)


@dataclass(frozen=True, slots=True)
class PlannerFrameResult:
    raw: str
    pydantic: BaseModel | JsonValue | None


def planner_frame_from_result(
    result: CrewAIProviderResultV1 | DirectPromptProgramResultV1,
) -> PlannerFrameResult:
    return PlannerFrameResult(raw=_raw_text(result.raw), pydantic=result.pydantic)


def coerce_planner_plan_with_error(
    frame_result: PlannerFrameResult,
    policy: PolicyManifestV2,
    *,
    scope_authority: BusinessScopeAuthorityV1,
) -> tuple[ExecutionPlanV2 | None, str]:
    plan_spec = _coerce_plan_spec(frame_result)
    if plan_spec is None:
        return None, _plan_spec_error_summary(frame_result)
    try:
        plan = finalize_execution_plan_v2(
            plan_spec,
            policy,
            scope_authority,
            origin="planner",
        )
    except ValueError as exc:
        return None, f"PlanSpec compile error: {exc}"
    return plan, ""


def _raw_text(value: ProviderRawTextV1) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def _coerce_plan_spec(result: PlannerFrameResult) -> PlanSpec | None:
    if result.pydantic is not None:
        try:
            return PlanSpec.model_validate(result.pydantic)
        except ValueError:
            return None
    try:
        return PlanSpec.model_validate_json(result.raw)
    except ValueError:
        return None


def _plan_spec_error_summary(result: PlannerFrameResult, max_issues: int = 5) -> str:
    if result.pydantic is not None:
        try:
            _ = PlanSpec.model_validate(result.pydantic)
            return ""
        except ValidationError as exc:
            return _validation_error_summary(exc, max_issues=max_issues)
        except ValueError as exc:
            return str(exc) or "PlanSpec validation failed"
    if not result.raw.strip():
        return "empty planner output"
    try:
        _ = PlanSpec.model_validate_json(result.raw)
        return ""
    except ValidationError as exc:
        return _validation_error_summary(exc, max_issues=max_issues)
    except ValueError as exc:
        return str(exc) or "PlanSpec validation failed"


def _validation_error_summary(exc: ValidationError, *, max_issues: int) -> str:
    errors = exc.errors()
    parts: list[str] = []
    for error in errors[:max_issues]:
        loc = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {error.get('msg', 'validation failed')}")
    if len(errors) > max_issues:
        parts.append(f"... {len(errors) - max_issues} more")
    return "; ".join(parts) or "PlanSpec validation failed"
