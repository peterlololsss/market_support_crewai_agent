from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from typing import Literal, TypedDict

from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.runtime.hashing import (
    CanonicalValue,
    canonical_json_bytes,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAITransportInvariantError,
)
from market_support_crewai_agent.runtime.integrations.crewai.targeting import agent_llm
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderChatMessageV1,
    ProviderJsonSchemaFormatV1,
    ProviderTransportEnvelopeV1,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    ChatGenerationParametersV1,
    CrewAIChatEnvelopeV1,
    ProviderMessageEnvelopeV1,
    ProviderResponseFormatV1,
)


class CrewAICompletionParamsV1(TypedDict):
    model: str
    messages: Sequence[JsonValue]


class CrewAICompletionKwargsV1(TypedDict, total=False):
    params: Mapping[str, JsonValue]
    available_functions: JsonValue | None
    from_task: JsonValue | None
    from_agent: JsonValue | None
    response_model: type[BaseModel] | None


RestoreCrewAIRequestCaptureV1 = Callable[[], None]


def install_crewai_request_capture(
    agent: CrewAIAgentAdapterV1 | None,
    _prompt_program: PromptProgram,
    envelope: ProviderTransportEnvelopeV1,
    request_captures: list[ProviderMessageEnvelopeV1],
) -> RestoreCrewAIRequestCaptureV1:
    llm = agent_llm(agent)
    if llm is None or llm.completion_capture is None:
        raise CrewAITransportInvariantError("provider_request_capture_unavailable")
    completion_capture = llm.completion_capture
    original = completion_capture.current()
    if original is None:
        raise CrewAITransportInvariantError("provider_request_capture_unavailable")

    def captured_handle_completion(
        *,
        params: Mapping[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> JsonValue | BaseModel:
        record_request_capture(
            request_captures,
            crewai_request_capture(params, response_model, envelope),
        )
        return original(
            params=params,
            available_functions=available_functions,
            from_task=from_task,
            from_agent=from_agent,
            response_model=response_model,
        )

    completion_capture.replace(captured_handle_completion)

    def restore() -> None:
        completion_capture.replace(original)

    return restore


def crewai_request_capture(
    params: Mapping[str, JsonValue],
    response_model: type[BaseModel] | None,
    envelope: ProviderTransportEnvelopeV1,
) -> CrewAIChatEnvelopeV1:
    payload = dict(params)
    try:
        model = str(payload.pop("model"))
        raw_messages = payload.pop("messages")
    except KeyError as exc:
        raise CrewAITransportInvariantError(
            "provider_request_required_field_missing"
        ) from exc
    raw_tools = payload.pop("tools", ())
    if payload.pop("response_format", None) is not None:
        raise CrewAITransportInvariantError("provider_request_field_unmodeled")
    generation_parameters = ChatGenerationParametersV1(
        temperature=_pop_float(payload, "temperature"),
        max_tokens=_pop_int(payload, "max_tokens"),
        top_p=_pop_float(payload, "top_p"),
        stop=_pop_str_tuple(payload, "stop"),
        seed=_pop_int(payload, "seed"),
        tool_choice=_pop_tool_choice(payload),
    )
    if payload:
        raise CrewAITransportInvariantError("provider_request_field_unmodeled")
    if model != envelope.model:
        raise CrewAITransportInvariantError("provider_request_model_mismatch")
    if not isinstance(raw_messages, (list, tuple)):
        raise CrewAITransportInvariantError("provider_request_messages_invalid")
    if not isinstance(raw_tools, (list, tuple)):
        raise CrewAITransportInvariantError("provider_request_tools_invalid")
    active_response_model = response_model
    return CrewAIChatEnvelopeV1(
        provider_id="openai_compatible",
        model_family=envelope.model_family,
        model_name=model,
        messages=tuple(
            ProviderChatMessageV1.model_validate(provider_json_value(message))
            for message in raw_messages
        ),
        tool_declarations_json=tuple(
            canonical_json_bytes(canonical_value(provider_json_value(tool)))
            for tool in raw_tools
        ),
        response_format=(
            ProviderResponseFormatV1(
                json_schema=ProviderJsonSchemaFormatV1(
                    name=active_response_model.__name__,
                    json_schema=active_response_model.model_json_schema(),
                ),
                provider_output_schema_hash=envelope.poh1(),
            )
            if active_response_model is not None
            else None
        ),
        generation_parameters=generation_parameters,
        crewai_version=version("crewai"),
    )


def provider_json_value(
    value: JsonValue | BaseModel | Sequence[JsonValue],
) -> JsonValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [provider_json_value(item) for item in value]
    if isinstance(value, list):
        return [provider_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): provider_json_value(item) for key, item in value.items()}
    return [provider_json_value(item) for item in value]


def canonical_value(value: JsonValue) -> CanonicalValue:
    if isinstance(value, float):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [canonical_value(item) for item in value]
    return {str(key): canonical_value(item) for key, item in value.items()}


def record_request_capture(
    captures: list[ProviderMessageEnvelopeV1],
    capture: ProviderMessageEnvelopeV1,
) -> None:
    if captures:
        raise CrewAITransportInvariantError("unmodeled_provider_retry")
    captures.append(capture)


def require_request_capture(
    captures: list[ProviderMessageEnvelopeV1],
) -> ProviderMessageEnvelopeV1:
    if len(captures) != 1:
        raise CrewAITransportInvariantError("provider_request_capture_missing")
    return captures[0]


def optional_request_capture(
    captures: list[ProviderMessageEnvelopeV1],
) -> ProviderMessageEnvelopeV1 | None:
    return captures[0] if len(captures) == 1 else None


def request_transport_id(capture: ProviderMessageEnvelopeV1) -> str:
    return capture.transport_variant


def _pop_float(payload: dict[str, JsonValue], key: str) -> float | None:
    value = payload.pop(key, None)
    return float(value) if isinstance(value, (int, float)) else None


def _pop_int(payload: dict[str, JsonValue], key: str) -> int | None:
    value = payload.pop(key, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _pop_str_tuple(payload: dict[str, JsonValue], key: str) -> tuple[str, ...]:
    value = payload.pop(key, None)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _pop_tool_choice(
    payload: dict[str, JsonValue],
) -> Literal["none", "auto", "required"] | None:
    value = payload.pop("tool_choice", None)
    match value:
        case "none" | "auto" | "required":
            return value
        case _:
            return None
