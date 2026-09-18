from __future__ import annotations

from pydantic import BaseModel, JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAILlmAdapterV1,
    CrewAITransportInvariantError,
    gemini_json_schema,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderTargetConfigV1,
    ProviderTransportEnvelopeV1,
    TargetSlotV1,
    resolve_active_prompt_program_v2,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    build_internal_crewai_sdk_target,
    build_provider_target_from_settings,
)
from market_support_crewai_agent.runtime.prompts.schema_canonicalization import (
    canonicalize_json_schema,
)

_STRING_OR_NONE_ADAPTER: TypeAdapter[str | None] = TypeAdapter(str | None)


def is_gemini_agent(agent: CrewAIAgentAdapterV1 | None) -> bool:
    llm = agent_llm(agent)
    provider = "" if llm is None else llm.provider.lower()
    return provider in {"gemini", "google"}


def crewai_transport_envelope(
    agent: CrewAIAgentAdapterV1 | None,
    prompt_program: PromptProgram,
) -> ProviderTransportEnvelopeV1:
    llm = agent_llm(agent)
    target = provider_target_from_agent(llm, prompt_program)
    response_model = prompt_program.profile.response_model
    canonical_schema = canonicalize_json_schema(response_model.model_json_schema())
    match target.transport_variant:
        case "openai_chat_completions":
            provider_schema: JsonValue = {
                "name": response_model.__name__,
                "schema": canonical_schema,
                "strict": True,
            }
            openai_body: JsonValue = {
                "model": target.model_name,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": provider_schema,
                },
            }
            return ProviderTransportEnvelopeV1(
                variant="openai_chat_completions",
                stage_kind=prompt_program.profile.stage,
                target_slot=target.target_slot,
                provider_id=target.provider_id,
                model_family=target.model_family,
                model=target.model_name,
                provider_schema_transform_version="openai-response-format.v1",
                canonical_output_schema=canonical_schema,
                provider_output_schema=provider_schema,
                body=openai_body,
            )
        case "gemini_generate_content":
            provider_schema = gemini_json_schema(response_model)
            gemini_body: JsonValue = {
                "model": target.model_name,
                "config": {"response_json_schema": provider_schema},
            }
            return ProviderTransportEnvelopeV1(
                variant="gemini_generate_content",
                stage_kind=prompt_program.profile.stage,
                target_slot=target.target_slot,
                provider_id=target.provider_id,
                model_family=target.model_family,
                model=target.model_name,
                provider_schema_transform_version="gemini-provider-schema.v1",
                canonical_output_schema=canonical_schema,
                provider_output_schema=provider_schema,
                body=gemini_body,
            )


def reject_invalid_dispatch(
    prompt_program: PromptProgram,
    health_log_context_present: bool,
) -> None:
    if prompt_program.scene_key == "wecom_direct.v1":
        raise CrewAITransportInvariantError("crewai_direct_scene_forbidden")
    active_program, _execution_spec = resolve_active_prompt_program_v2(
        stage=prompt_program.profile.stage,
        scene_key=prompt_program.scene_key,
    )
    if (
        active_program.program_id != prompt_program.program_id
        or active_program.scene_contract_id != prompt_program.scene_contract_id
        or active_program.scene_contract_version
        != prompt_program.scene_contract_version
    ):
        raise CrewAITransportInvariantError("prompt_program_v2_dispatch_mismatch")
    is_health_probe = prompt_program.profile.stage == "llm_health_probe"
    if is_health_probe != health_log_context_present:
        raise CrewAITransportInvariantError("health_log_context_mismatch")


def provider_target_from_agent(
    llm: CrewAILlmAdapterV1 | None,
    prompt_program: PromptProgram,
) -> ProviderTargetConfigV1:
    provider = _str_attr(llm, "provider", "openai")
    model = _str_attr(llm, "model", "unknown-model")
    base_url = agent_base_url(llm)
    max_tokens = _max_tokens(llm)
    target_slot = target_slot_for_stage(prompt_program.profile.stage)
    api_key_configured = bool(_attr(llm, "api_key"))
    timeout_seconds = min(_float_attr(llm, "timeout", 30.0), 30.0)
    temperature = _float_attr(llm, "temperature", 0.0)
    if not base_url:
        return build_internal_crewai_sdk_target(
            provider=provider,
            target_slot=target_slot,
            model=model,
            api_key_configured=api_key_configured,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_config=_attr(llm, "thinking_config"),
        )
    return build_provider_target_from_settings(
        provider=provider,
        target_slot=target_slot,
        model=model,
        api_key_configured=api_key_configured,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


def target_slot_for_stage(stage: str) -> TargetSlotV1:
    match stage:
        case "planner_intent":
            return "planner"
        case "approved_knowledge_selector" | "document_product_selector":
            return "selector"
        case "llm_health_probe":
            return "health"
        case "knowledge_composer" | "smalltalk_composer" | "alignment_verifier":
            return "composer"
        case _:
            raise CrewAITransportInvariantError("unknown_prompt_stage")


def agent_llm(agent: CrewAIAgentAdapterV1 | None) -> CrewAILlmAdapterV1 | None:
    if agent is None:
        return None
    return agent.llm


def agent_base_url(llm: CrewAILlmAdapterV1 | None) -> str:
    base_url = _str_attr(llm, "base_url", "")
    if base_url:
        return base_url
    client_params = _attr(llm, "client_params")
    if not isinstance(client_params, dict):
        return ""
    http_options = client_params.get("http_options", {})
    if not isinstance(http_options, dict):
        return ""
    value = http_options.get("base_url", "")
    return value if isinstance(value, str) else ""


def agent_role(agent: CrewAIAgentAdapterV1 | None) -> str:
    if agent is None:
        return ""
    return agent.role


def input_schema_version(stage: str) -> str:
    return {
        "planner_intent": "planner-prompt-input.v1",
        "knowledge_composer": "knowledge-composer-input.v1",
        "smalltalk_composer": "smalltalk-composer-input.v1",
        "alignment_verifier": "alignment-verifier-input.v1",
        "approved_knowledge_selector": "approved-knowledge-selector-input.v1",
        "document_product_selector": "document-product-selector-input.v1",
        "llm_health_probe": "llm-health-probe-input.v1",
    }[stage]


def output_schema_version(response_model: type[BaseModel]) -> str:
    field = response_model.model_fields.get("contract_version")
    if field is not None:
        return response_model.__name__
    return response_model.__name__


def _max_tokens(llm: CrewAILlmAdapterV1 | None) -> int:
    max_tokens = _attr(llm, "max_tokens")
    if isinstance(max_tokens, int):
        return max_tokens
    max_output_tokens = _attr(llm, "max_output_tokens")
    if isinstance(max_output_tokens, int):
        return max_output_tokens
    return 1200


def _attr(llm: CrewAILlmAdapterV1 | None, name: str) -> JsonValue | None:
    if llm is None:
        return None
    match name:
        case "provider":
            return llm.provider
        case "model":
            return llm.model
        case "api_key":
            return llm.api_key
        case "base_url":
            return llm.base_url
        case "client_params":
            return {key: _json_value(value) for key, value in llm.client_params.items()}
        case "max_tokens":
            return llm.max_tokens
        case "max_output_tokens":
            return llm.max_output_tokens
        case "timeout":
            return llm.timeout
        case "temperature":
            return llm.temperature
        case "thinking_config":
            return llm.thinking_config
        case _:
            raise CrewAITransportInvariantError("crewai_llm_attribute_unmodeled")


def _json_value(value: JsonValue | None) -> JsonValue | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return {str(key): _json_value(item) for key, item in value.items()}


def _str_attr(llm: CrewAILlmAdapterV1 | None, name: str, default: str) -> str:
    value = _attr(llm, name)
    return value if isinstance(value, str) and value else default


def _float_attr(llm: CrewAILlmAdapterV1 | None, name: str, default: float) -> float:
    value = _attr(llm, name)
    if isinstance(value, (int, float)):
        return float(value)
    return default
