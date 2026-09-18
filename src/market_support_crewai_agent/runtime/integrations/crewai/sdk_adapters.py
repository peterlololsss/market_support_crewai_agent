from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from crewai.lite_agent_output import LiteAgentOutput
from openai import OpenAI
from pydantic import BaseModel, JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAIKickoffOutputV1,
    CrewAILlmAdapterV1,
    CrewAITransportInvariantError,
)
from market_support_crewai_agent.runtime.integrations.crewai.gemini_sdk_client import (
    gemini_retry_attempts,
    gemini_sync_client_factory,
)
from market_support_crewai_agent.runtime.integrations.crewai.sdk_completion_capture import (
    completion_capture_target,
)
from market_support_crewai_agent.runtime.integrations.crewai.sdk_payload import (
    mapping_value,
    optional_float_value,
    optional_int_value,
    optional_str_value,
    str_value,
    string_sequence_value,
)

if TYPE_CHECKING:
    from crewai.llms.base_llm import BaseLLM as CrewAISdkLLM

_JSON_MAPPING: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


class CrewAISdkAgentAdapterV1(Protocol):
    role: str
    llm: str | CrewAISdkLLM | None
    planning: bool
    allow_delegation: bool
    inject_date: bool
    max_retry_limit: int

    async def kickoff_async(
        self,
        messages: str,
        response_format: type[BaseModel] | None = None,
    ) -> LiteAgentOutput: ...


def adapt_sdk_agent(agent: CrewAISdkAgentAdapterV1) -> CrewAIAgentAdapterV1:
    from crewai.llms.base_llm import BaseLLM

    if not isinstance(agent.llm, BaseLLM):
        raise CrewAITransportInvariantError("crewai_llm_type_unsupported")
    llm = agent.llm
    sdk_kickoff: Callable[..., Awaitable[LiteAgentOutput]] = agent.kickoff_async

    async def kickoff(
        prompt: str, response_format: type[BaseModel]
    ) -> CrewAIKickoffOutputV1:
        sdk_result = await sdk_kickoff(prompt, response_format=response_format)
        return CrewAIKickoffOutputV1(
            raw=provider_raw_text(sdk_result.raw),
            pydantic=provider_pydantic(sdk_result.pydantic),
        )

    return CrewAIAgentAdapterV1(
        role=agent.role,
        llm=adapt_sdk_llm(llm),
        kickoff_async=kickoff,
        planning=agent.planning,
        allow_delegation=agent.allow_delegation,
        inject_date=agent.inject_date,
        max_retry_limit=agent.max_retry_limit,
    )


def adapt_sdk_llm(llm: CrewAISdkLLM) -> CrewAILlmAdapterV1:
    payload = _JSON_MAPPING.validate_python(llm.model_dump(mode="json"))
    return CrewAILlmAdapterV1(
        provider=str_value(payload, "provider", "openai"),
        model=str_value(payload, "model", "unknown-model"),
        api_key=optional_str_value(payload, "api_key"),
        base_url=optional_str_value(payload, "base_url"),
        client_params=mapping_value(payload, "client_params"),
        max_tokens=optional_int_value(payload, "max_tokens"),
        max_output_tokens=optional_int_value(payload, "max_output_tokens"),
        timeout=optional_float_value(payload, "timeout"),
        temperature=optional_float_value(payload, "temperature"),
        top_p=optional_float_value(payload, "top_p"),
        top_k=optional_int_value(payload, "top_k"),
        stop_sequences=string_sequence_value(payload, "stop_sequences"),
        thinking_config=payload.get("thinking_config"),
        max_retries=optional_int_value(payload, "max_retries"),
        gemini_retry_attempts=gemini_retry_attempts(payload),
        configure_openai_client=openai_client_configurator(llm),
        completion_capture=completion_capture_target(llm),
        gemini_sync_client_factory=gemini_sync_client_factory(payload),
    )


def openai_client_configurator(llm: CrewAISdkLLM) -> Callable[[OpenAI], None]:
    def configure(client: OpenAI) -> None:
        llm._client = client

    return configure


def provider_raw_text(value: JsonValue) -> str | bytes | int | float | bool | None:
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    return str(value)


def provider_pydantic(
    value: JsonValue | BaseModel | None,
) -> BaseModel | JsonValue | None:
    if isinstance(value, BaseModel):
        return value
    return value
