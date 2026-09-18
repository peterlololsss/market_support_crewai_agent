from __future__ import annotations

import logging
from typing import Literal, TypedDict

from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIProviderResultV1,
    HealthInvocationLogContextV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.request_capture import (
    optional_request_capture,
)
from market_support_crewai_agent.runtime.integrations.crewai.targeting import (
    input_schema_version,
    output_schema_version,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    ProviderMessageEnvelopeV1,
)

logger = logging.getLogger(__name__)
HEALTH_PROVIDER_LOG_TEMPLATE = (
    "health_provider_invocation stage=llm_health_probe program_id=%s "
    "program_version=%s input_schema_version=%s output_schema_version=%s "
    "target_slot=%s htk1=%s status=%s error_code=%s "
    "input_bytes=%s output_bytes=%s latency_ms=%.1f"
)
LLM_EXECUTION_LOG_TEMPLATE = (
    "[LLM] stage=%s mode=%s hph1=%s prh1=%s out1=%s "
    "prompt_chars=%s raw_length=%s pydantic_type=%s latency_ms=%.1f"
)


class TraceAttributesV1(TypedDict, total=False):
    stage: str
    prompt_profile_id: str
    program_id: str
    target_slot: str
    htk1: str
    hph1: str
    osh1: str
    poh1: str
    prompt_chars: int
    response_format: str


def trace_attributes(
    prompt_program: PromptProgram,
    *,
    osh1: str,
    poh1: str,
    health_log_context: HealthInvocationLogContextV1 | None,
) -> TraceAttributesV1:
    if health_log_context is not None:
        return {
            "stage": prompt_program.profile.stage,
            "program_id": prompt_program.program_id,
            "target_slot": health_log_context.target_slot,
            "htk1": health_log_context.htk1,
        }
    return {
        "stage": prompt_program.profile.stage,
        "prompt_profile_id": prompt_program.profile.id,
        "hph1": prompt_program.hph1,
        "osh1": osh1,
        "poh1": poh1,
        "prompt_chars": len(prompt_program.prompt_text),
        "response_format": prompt_program.profile.response_model.__name__,
    }


def log_health_transport_error(
    prompt_program: PromptProgram,
    context: HealthInvocationLogContextV1 | None,
    captures: list[ProviderMessageEnvelopeV1],
    *,
    error_code: ProviderFailureCodeV1,
    latency_ms: float,
) -> None:
    if context is None:
        return
    capture = optional_request_capture(captures)
    log_health_execution(
        prompt_program,
        context,
        status="transport_error",
        error_code=error_code,
        input_bytes=capture.byte_count if capture is not None else 0,
        output_bytes=None,
        latency_ms=latency_ms,
    )


def log_health_execution(
    prompt_program: PromptProgram,
    context: HealthInvocationLogContextV1,
    *,
    status: Literal["success", "output_contract_error", "transport_error"],
    error_code: str | None,
    input_bytes: int,
    output_bytes: int | None,
    latency_ms: float,
) -> None:
    log = logger.info if status == "success" else logger.warning
    log(
        HEALTH_PROVIDER_LOG_TEMPLATE,
        prompt_program.program_id,
        prompt_program.program_version,
        input_schema_version("llm_health_probe"),
        output_schema_version(prompt_program.profile.response_model),
        context.target_slot,
        context.htk1,
        status,
        error_code or "",
        input_bytes,
        output_bytes if output_bytes is not None else "",
        latency_ms,
    )


def log_llm_execution(
    prompt_program: PromptProgram,
    result: CrewAIProviderResultV1,
    latency_ms: float,
    *,
    mode: str,
    prh1: str,
    out1: str | None,
) -> None:
    logger.info(
        LLM_EXECUTION_LOG_TEMPLATE,
        prompt_program.profile.stage,
        mode,
        prompt_program.hph1,
        prh1,
        out1 or "",
        len(prompt_program.prompt_text),
        len(str(result.raw or "")),
        pydantic_type_name(result.pydantic),
        latency_ms,
    )
    if prompt_program.profile.stage == "planner_intent" and result.pydantic is None:
        logger.warning("[Planner] invalid structured output")


def compact_crewai_execution(
    prompt_program: PromptProgram,
    result: CrewAIProviderResultV1,
    latency_ms: float,
) -> dict[str, JsonValue]:
    prompt_profile = prompt_program.profile
    return {
        "stage": prompt_profile.stage,
        "prompt_profile_id": prompt_profile.id,
        "prompt_fragment_ids": list(prompt_program.fragment_ids),
        "prompt_layers": list(prompt_program.layers),
        "prompt_hash": prompt_program.prompt_hash,
        "agent_role": result.agent_role,
        "response_format": prompt_profile.response_model.__name__,
        "latency_ms": round(latency_ms, 3),
        "usage_metrics": compact_usage_metrics(result.usage_metrics),
        "pydantic_type": pydantic_type_name(result.pydantic),
        "raw_length": len(str(result.raw or "")),
    }


def compact_usage_metrics(value: JsonValue | None) -> JsonValue | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            str(key): compact_usage_metrics(item)
            for key, item in value.items()
            if not str(key).lower().endswith(("key", "token_value", "secret"))
        }
    if isinstance(value, (list, tuple)):
        return [compact_usage_metrics(item) for item in value[:20]]
    return value


def pydantic_type_name(value: BaseModel | JsonValue | None) -> str:
    if value is None:
        return ""
    return value.__class__.__name__


def record_llm_success(agent: CrewAIAgentAdapterV1 | None, stage: str) -> None:
    try:
        from market_support_crewai_agent.health.llm_health_hooks import (
            record_llm_success_for_agent,
        )

        record_llm_success_for_agent(agent, stage)
    except RuntimeError:
        logger.debug("LLM health success hook failed", exc_info=True)


def safe_short_text(value: JsonValue | BaseModel | Exception | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= 160:
        return text
    return text[:157] + "..."
