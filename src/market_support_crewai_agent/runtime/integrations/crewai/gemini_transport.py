from __future__ import annotations

from importlib.metadata import version

from google.genai import types
from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.hashing import canonical_json_bytes
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAILlmAdapterV1,
    CrewAIProviderResultV1,
    CrewAITransportInvariantError,
    GeminiSyncClientV1,
    gemini_json_schema,
)
from market_support_crewai_agent.runtime.integrations.crewai.request_capture import (
    canonical_value,
    provider_json_value,
    record_request_capture,
)
from market_support_crewai_agent.runtime.integrations.crewai.targeting import (
    agent_llm,
    agent_role,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderTransportEnvelopeV1,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    GeminiContentV1,
    GeminiGenerateContentEnvelopeV1,
    GeminiGenerationParametersV1,
    GeminiTextPartV1,
    GeminiThinkingConfigV1,
    ProviderMessageEnvelopeV1,
)

_JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def run_gemini_structured(
    agent: CrewAIAgentAdapterV1 | None,
    prompt_program: PromptProgram,
    envelope: ProviderTransportEnvelopeV1,
    request_captures: list[ProviderMessageEnvelopeV1],
) -> CrewAIProviderResultV1:
    llm = agent_llm(agent)
    if llm is None:
        raise CrewAITransportInvariantError("crewai_agent_llm_required")
    response_model = prompt_program.profile.response_model
    config = types.GenerateContentConfig(
        temperature=llm.temperature,
        top_p=llm.top_p,
        top_k=llm.top_k,
        max_output_tokens=llm.max_output_tokens,
        stop_sequences=list(llm.stop_sequences) if llm.stop_sequences else None,
        response_mime_type="application/json",
        response_json_schema=gemini_json_schema(response_model),
        thinking_config=gemini_sdk_thinking_config(llm.thinking_config),
    )
    assert_gemini_config_closed(config)
    record_request_capture(
        request_captures,
        GeminiGenerateContentEnvelopeV1(
            model_family=envelope.model_family,
            model_name=llm.model,
            contents=(
                GeminiContentV1(
                    role="user",
                    parts=(GeminiTextPartV1(text=prompt_program.prompt_text),),
                ),
            ),
            provider_schema_json=canonical_json_bytes(
                canonical_value(provider_json_value(config.response_json_schema))
            ),
            provider_output_schema_hash=envelope.poh1(),
            generation_parameters=GeminiGenerationParametersV1(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k if isinstance(config.top_k, int) else None,
                max_output_tokens=config.max_output_tokens,
                stop_sequences=tuple(config.stop_sequences or ()),
                response_mime_type=str(config.response_mime_type),
                safety_settings_json=(
                    canonical_json_bytes(
                        canonical_value(
                            provider_json_value(
                                config.model_dump(mode="json", exclude_none=True).get(
                                    "safety_settings"
                                )
                            )
                        )
                    )
                    if config.safety_settings is not None
                    else None
                ),
                thinking_config=gemini_thinking_config(config.thinking_config),
            ),
            google_sdk_version=version("google-genai"),
        ),
    )
    response = gemini_sync_client(llm).models.generate_content(
        model=llm.model,
        contents=prompt_program.prompt_text,
        config=config,
    )
    raw = gemini_response_text(response)
    try:
        pydantic = response_model.model_validate_json(raw)
    except ValueError:
        pydantic = None
    return CrewAIProviderResultV1(
        raw=raw,
        pydantic=pydantic,
        agent_role=agent_role(agent),
        usage_metrics=usage_metadata(response),
    )


def gemini_response_text(response: JsonValue) -> str:
    if isinstance(response, dict):
        value = response.get("text")
        return value if isinstance(value, str) else ""
    return ""


def assert_gemini_config_closed(config: types.GenerateContentConfig) -> None:
    payload = config.model_dump(mode="json", exclude_none=True)
    allowed = {
        "max_output_tokens",
        "response_json_schema",
        "response_mime_type",
        "safety_settings",
        "stop_sequences",
        "temperature",
        "thinking_config",
        "tools",
        "top_k",
        "top_p",
    }
    if set(payload) - allowed:
        raise CrewAITransportInvariantError("provider_request_field_unmodeled")


def gemini_thinking_config(
    value: JsonValue | types.ThinkingConfig | None,
) -> GeminiThinkingConfigV1 | None:
    if value is None:
        return None
    payload = provider_json_value(value)
    if not isinstance(payload, dict):
        raise CrewAITransportInvariantError("gemini_thinking_config_invalid")
    normalized = {key: item for key, item in payload.items() if item is not None}
    if not normalized:
        return None
    return GeminiThinkingConfigV1.model_validate(normalized)


def usage_metadata(response: JsonValue) -> JsonValue | None:
    if isinstance(response, dict):
        usage_value = response.get("usage_metadata")
        return (
            _JSON_ADAPTER.validate_python(usage_value)
            if usage_value is not None
            else None
        )
    return None


def gemini_sdk_thinking_config(value: JsonValue | None) -> types.ThinkingConfig | None:
    if value is None:
        return None
    payload = provider_json_value(value)
    if not isinstance(payload, dict):
        raise CrewAITransportInvariantError("gemini_thinking_config_invalid")
    return types.ThinkingConfig.model_validate(payload)


def gemini_sync_client(llm: CrewAILlmAdapterV1) -> GeminiSyncClientV1:
    if llm.gemini_sync_client_factory is None:
        raise CrewAITransportInvariantError("gemini_sync_client_unavailable")
    return llm.gemini_sync_client_factory()
