from __future__ import annotations

from pydantic import BaseModel, JsonValue, ValidationError

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIProviderResultV1,
    ProviderRawTextV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.observability import (
    safe_short_text,
)
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse


def coerce_plan_spec(result: CrewAIProviderResultV1) -> PlanSpec | None:
    if result.pydantic is not None:
        try:
            return PlanSpec.model_validate(result.pydantic)
        except ValueError:
            return None
    try:
        return PlanSpec.model_validate_json(raw_json(result.raw))
    except ValueError:
        return None


def plan_spec_error_summary(result: CrewAIProviderResultV1, max_issues: int = 5) -> str:
    if result.pydantic is not None:
        try:
            _ = PlanSpec.model_validate(result.pydantic)
            return ""
        except ValidationError as exc:
            return _validation_error_summary(exc, max_issues=max_issues)
        except ValueError as exc:
            return safe_short_text(exc) or "PlanSpec validation failed"
    raw = raw_json(result.raw)
    if not raw.strip():
        return "empty planner output"
    try:
        _ = PlanSpec.model_validate_json(raw)
        return ""
    except ValidationError as exc:
        return _validation_error_summary(exc, max_issues=max_issues)
    except ValueError as exc:
        return safe_short_text(exc) or "PlanSpec validation failed"


def coerce_composer_output(
    result: CrewAIProviderResultV1,
) -> ComposerReplyOutput | None:
    if result.pydantic is not None:
        try:
            return ComposerReplyOutput.model_validate(result.pydantic)
        except ValueError:
            return None
    try:
        return ComposerReplyOutput.model_validate_json(raw_json(result.raw))
    except ValueError:
        return None


def coerce_agent_response(result: CrewAIProviderResultV1) -> ReplyResponse | None:
    if result.pydantic is not None:
        try:
            parsed = _composer_output_from_pydantic(result.pydantic)
            if parsed is not None:
                return parsed.to_reply_response()
            return ReplyResponse.model_validate(result.pydantic)
        except ValueError:
            return None
    try:
        return ComposerReplyOutput.model_validate_json(
            raw_json(result.raw)
        ).to_reply_response()
    except ValueError:
        pass
    try:
        return ReplyResponse.model_validate_json(raw_json(result.raw))
    except ValueError:
        return None


def coerce_alignment_verdict(
    result: CrewAIProviderResultV1,
) -> ReplyAlignmentVerdict | None:
    if result.pydantic is not None:
        try:
            return ReplyAlignmentVerdict.model_validate(result.pydantic)
        except ValueError:
            return None
    try:
        return ReplyAlignmentVerdict.model_validate_json(raw_json(result.raw))
    except ValueError:
        return None


def _composer_output_from_pydantic(
    value: BaseModel | JsonValue,
) -> ComposerReplyOutput | None:
    if isinstance(value, ComposerReplyOutput):
        return value
    try:
        return ComposerReplyOutput.model_validate(value)
    except ValueError:
        return None


def _validation_error_summary(exc: ValidationError, *, max_issues: int) -> str:
    errors = exc.errors()
    parts: list[str] = []
    for error in errors[:max_issues]:
        loc = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {error.get('msg', 'validation failed')}")
    if len(errors) > max_issues:
        parts.append(f"... {len(errors) - max_issues} more")
    return "; ".join(parts) or "PlanSpec validation failed"


def raw_json(value: ProviderRawTextV1) -> str | bytes:
    if isinstance(value, (str, bytes)):
        return value
    return ""
