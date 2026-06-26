from __future__ import annotations

import asyncio
import copy
import logging
from types import SimpleNamespace
from time import perf_counter

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
from market_support_crewai_agent.runtime.state.runtime_trace import trace_span

logger = logging.getLogger(__name__)


async def run_crewai_kickoff(
    agent,
    prompt_program: PromptProgram,
    *,
    timeout_seconds: float | None,
):
    started_at = perf_counter()
    gemini_structured = _is_gemini_agent(agent)
    with trace_span(
        "llm.crewai_kickoff",
        stage=prompt_program.profile.stage,
        prompt_profile_id=prompt_program.profile.id,
        prompt_hash=prompt_program.prompt_hash,
        prompt_chars=len(prompt_program.prompt_text),
        response_format=getattr(
            prompt_program.profile.response_model,
            "__name__",
            str(prompt_program.profile.response_model),
        ),
    ):
        if gemini_structured:
            result = await asyncio.wait_for(
                asyncio.to_thread(_run_gemini_structured, agent, prompt_program),
                timeout=timeout_seconds,
            )
        else:
            result = await asyncio.wait_for(
                agent.kickoff_async(
                    prompt_program.prompt_text,
                    response_format=prompt_program.profile.response_model,
                ),
                timeout=timeout_seconds,
            )
    latency_ms = (perf_counter() - started_at) * 1000
    _log_llm_execution(
        agent,
        prompt_program,
        result,
        latency_ms,
        mode="gemini_structured" if gemini_structured else "crewai",
    )
    if getattr(result, "pydantic", None) is not None:
        _record_llm_success(agent, prompt_program.profile.stage)
    return result, _compact_crewai_execution(prompt_program, result, latency_ms)


def _is_gemini_agent(agent) -> bool:
    provider = str(getattr(getattr(agent, "llm", None), "provider", "")).lower()
    return provider in {"gemini", "google"}


def _run_gemini_structured(agent, prompt_program: PromptProgram):
    from google.genai import types

    llm = agent.llm
    response_model = prompt_program.profile.response_model
    config = types.GenerateContentConfig(
        temperature=getattr(llm, "temperature", None),
        top_p=getattr(llm, "top_p", None),
        top_k=getattr(llm, "top_k", None),
        max_output_tokens=getattr(llm, "max_output_tokens", None),
        stop_sequences=getattr(llm, "stop_sequences", None) or None,
        response_mime_type="application/json",
        response_json_schema=_gemini_json_schema(response_model),
        thinking_config=getattr(llm, "thinking_config", None),
    )
    response = llm._get_sync_client().models.generate_content(
        model=llm.model,
        contents=prompt_program.prompt_text,
        config=config,
    )
    raw = getattr(response, "text", None) or ""
    try:
        pydantic = response_model.model_validate_json(raw)
    except ValueError:
        pydantic = None
    return SimpleNamespace(
        raw=raw,
        pydantic=pydantic,
        agent_role=str(getattr(agent, "role", "") or ""),
        usage_metrics=_usage_metadata(response),
    )


def _gemini_json_schema(model) -> dict:
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    drop_keys = {"$defs", "$schema", "title", "default", "examples"}
    drop_constraints = {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
    }

    def convert(node):
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            merged = copy.deepcopy(defs[name])
            merged.update({key: value for key, value in node.items() if key != "$ref"})
            return convert(merged)
        if "anyOf" in node:
            choices = node["anyOf"]
            non_null = [item for item in choices if item.get("type") != "null"]
            if len(non_null) == 1 and len(non_null) != len(choices):
                output = convert(non_null[0])
                type_value = output.get("type")
                types = [type_value] if isinstance(type_value, str) else list(type_value or [])
                output["type"] = types + ["null"]
                return output
            return convert(non_null[0] if non_null else choices[0])

        output = {}
        for key, value in node.items():
            if key in drop_keys or key in drop_constraints or key == "additionalProperties":
                continue
            if key == "const":
                output["enum"] = [value]
            elif key == "properties":
                properties = {prop: convert(prop_schema) for prop, prop_schema in value.items()}
                output["properties"] = properties
                output["propertyOrdering"] = list(properties)
            else:
                output[key] = convert(value)
        return output

    return convert(schema)


def _usage_metadata(response):
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    return usage


def _log_llm_execution(
    agent,
    prompt_program: PromptProgram,
    result,
    latency_ms: float,
    *,
    mode: str,
) -> None:
    llm = getattr(agent, "llm", None)
    logger.info(
        "[LLM] stage=%s mode=%s provider=%s model=%s prompt_hash=%s "
        "prompt_chars=%s raw_length=%s pydantic_type=%s latency_ms=%.1f",
        prompt_program.profile.stage,
        mode,
        str(getattr(llm, "provider", "") or ""),
        str(getattr(llm, "model", "") or ""),
        prompt_program.prompt_hash,
        len(prompt_program.prompt_text),
        len(str(getattr(result, "raw", "") or "")),
        _pydantic_type_name(getattr(result, "pydantic", None)),
        latency_ms,
    )
    if (
        prompt_program.profile.stage == "planner_intent"
        and getattr(result, "pydantic", None) is None
    ):
        logger.warning(
            "[Planner] invalid structured output: %s",
            plan_spec_error_summary(result),
        )


def coerce_planner_plan(
    result,
    request: ReplyRequest,
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


def plan_spec_error_summary(result, max_issues: int = 5) -> str:
    if getattr(result, "pydantic", None) is not None:
        try:
            PlanSpec.model_validate(result.pydantic)
            return ""
        except ValueError as exc:
            return _validation_error_summary(exc, max_issues=max_issues)

    raw = str(getattr(result, "raw", "") or "")
    if not raw.strip():
        return "empty planner output"
    try:
        PlanSpec.model_validate_json(raw)
        return ""
    except ValueError as exc:
        return _validation_error_summary(exc, max_issues=max_issues)


def _validation_error_summary(exc: ValueError, *, max_issues: int) -> str:
    errors = getattr(exc, "errors", lambda: [])()
    if errors:
        parts = []
        for error in errors[:max_issues]:
            loc = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
            parts.append(f"{loc}: {error.get('msg', 'validation failed')}")
        if len(errors) > max_issues:
            parts.append(f"... {len(errors) - max_issues} more")
        return "; ".join(parts)
    return safe_short_text(exc) or "PlanSpec validation failed"


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


def _record_llm_success(agent, stage: str) -> None:
    try:
        from market_support_crewai_agent.health.llm_health import (
            record_llm_success_for_agent,
        )

        record_llm_success_for_agent(agent, stage)
    except Exception:
        logger.debug("LLM health success hook failed", exc_info=True)
