from __future__ import annotations

import asyncio
from time import perf_counter

from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    compile_plan_spec,
)
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.runtime.llm.prompting.assembler import PromptProgram
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse


async def run_crewai_kickoff(
    agent,
    prompt_program: PromptProgram,
    *,
    timeout_seconds: float | None,
):
    started_at = perf_counter()
    result = await asyncio.wait_for(
        agent.kickoff_async(
            prompt_program.prompt_text,
            response_format=prompt_program.profile.response_model,
        ),
        timeout=timeout_seconds,
    )
    latency_ms = (perf_counter() - started_at) * 1000
    return result, _compact_crewai_execution(prompt_program, result, latency_ms)


def coerce_planner_plan(
    result,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    policy: PolicyManifest,
    *,
    domain_context: DomainContext,
    history: list | None = None,
) -> ExecutionPlan | None:
    del history
    plan_spec = coerce_plan_spec(result)
    if plan_spec is not None:
        return compile_plan_spec(
            plan_spec,
            request,
            canonical_context,
            policy,
            domain_context=domain_context,
        )
    return None


def coerce_plan_spec(result) -> PlanSpec | None:
    if result.pydantic is not None:
        try:
            return PlanSpec.model_validate(result.pydantic)
        except ValueError:
            return None

    try:
        return PlanSpec.model_validate_json(result.raw)
    except ValueError:
        return None


def coerce_composer_output(result) -> ComposerReplyOutput | None:
    if result.pydantic is not None:
        try:
            return ComposerReplyOutput.model_validate(result.pydantic)
        except ValueError:
            return None

    try:
        return ComposerReplyOutput.model_validate_json(result.raw)
    except ValueError:
        return None


def coerce_agent_response(result) -> ReplyResponse | None:
    if result.pydantic is not None:
        try:
            if isinstance(result.pydantic, ComposerReplyOutput):
                return result.pydantic.to_reply_response()
            try:
                return ComposerReplyOutput.model_validate(
                    result.pydantic
                ).to_reply_response()
            except ValueError:
                pass
            return ReplyResponse.model_validate(result.pydantic)
        except ValueError:
            return None

    try:
        return ComposerReplyOutput.model_validate_json(result.raw).to_reply_response()
    except ValueError:
        pass

    try:
        return ReplyResponse.model_validate_json(result.raw)
    except ValueError:
        return None


def coerce_alignment_verdict(result) -> ReplyAlignmentVerdict | None:
    if result.pydantic is not None:
        try:
            return ReplyAlignmentVerdict.model_validate(result.pydantic)
        except ValueError:
            return None

    try:
        return ReplyAlignmentVerdict.model_validate_json(result.raw)
    except ValueError:
        return None


def safe_short_text(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= 160:
        return text
    return text[:157] + "..."


def _compact_crewai_execution(
    prompt_program: PromptProgram,
    result,
    latency_ms: float,
) -> dict:
    prompt_profile = prompt_program.profile
    return {
        "stage": prompt_profile.stage,
        "prompt_profile_id": prompt_profile.id,
        "prompt_fragment_ids": list(prompt_program.fragment_ids),
        "prompt_layers": list(prompt_program.layers),
        "prompt_hash": prompt_program.prompt_hash,
        "agent_role": str(getattr(result, "agent_role", "") or ""),
        "response_format": getattr(
            prompt_profile.response_model,
            "__name__",
            str(prompt_profile.response_model),
        ),
        "latency_ms": round(latency_ms, 3),
        "usage_metrics": _compact_usage_metrics(
            getattr(result, "usage_metrics", None)
        ),
        "pydantic_type": _pydantic_type_name(getattr(result, "pydantic", None)),
        "raw_length": len(str(getattr(result, "raw", "") or "")),
    }


def _compact_usage_metrics(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(key): _compact_usage_metrics(item)
            for key, item in value.items()
            if not str(key).lower().endswith(("key", "token_value", "secret"))
        }
    if isinstance(value, (list, tuple)):
        return [_compact_usage_metrics(item) for item in value[:20]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return safe_short_text(value)


def _pydantic_type_name(value) -> str:
    if value is None:
        return ""
    return value.__class__.__name__
