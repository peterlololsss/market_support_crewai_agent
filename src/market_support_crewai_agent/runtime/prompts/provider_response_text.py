from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
)

JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


class DirectProviderTransportError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DirectPromptProgramResultV1:
    raw: str
    pydantic: BaseModel | JsonValue | None
    agent_role: str
    usage_metrics: JsonValue | None


def build_direct_prompt_program_result(
    prompt_program: PromptProgram,
    text: str,
    *,
    latency_ms: float,
) -> tuple[DirectPromptProgramResultV1, dict[str, JsonValue]]:
    response_model = prompt_program.profile.response_model
    parsed = response_model.model_validate_json(text)
    result = DirectPromptProgramResultV1(
        raw=text,
        pydantic=parsed,
        agent_role=prompt_program.profile.stage,
        usage_metrics=None,
    )
    execution: dict[str, JsonValue] = {
        "stage": prompt_program.profile.stage,
        "prompt_profile_id": prompt_program.profile.id,
        "prompt_fragment_ids": list(prompt_program.fragment_ids),
        "prompt_layers": list(prompt_program.layers),
        "prompt_hash": prompt_program.prompt_hash,
        "response_format": response_model.__name__,
        "latency_ms": latency_ms,
        "pydantic_type": parsed.__class__.__name__,
        "raw_length": len(text),
    }
    return result, execution


def openai_response_text(response_bytes: bytes) -> str:
    payload = JSON_VALUE_ADAPTER.validate_json(response_bytes)
    try:
        if not isinstance(payload, dict):
            raise ProviderInvocationError("provider_output_type")
        choices = payload["choices"]
        if not isinstance(choices, list):
            raise ProviderInvocationError("provider_output_type")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ProviderInvocationError("provider_output_type")
        message = first_choice["message"]
        if not isinstance(message, dict):
            raise ProviderInvocationError("provider_output_type")
        text = message["content"]
    except (KeyError, IndexError):
        raise ProviderInvocationError("provider_output_missing") from None
    if not isinstance(text, str):
        raise ProviderInvocationError("provider_output_type")
    return text


def gemini_response_text(response_bytes: bytes) -> str:
    payload = JSON_VALUE_ADAPTER.validate_json(response_bytes)
    try:
        if not isinstance(payload, dict):
            raise ProviderInvocationError("provider_output_type")
        candidates = payload["candidates"]
        if not isinstance(candidates, list):
            raise ProviderInvocationError("provider_output_type")
        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            raise ProviderInvocationError("provider_output_type")
        content = first_candidate["content"]
        if not isinstance(content, dict):
            raise ProviderInvocationError("provider_output_type")
        parts = content["parts"]
        if not isinstance(parts, list):
            raise ProviderInvocationError("provider_output_type")
    except (KeyError, IndexError):
        raise ProviderInvocationError("provider_output_missing") from None
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or "text" not in part:
            continue
        text = part["text"]
        if not isinstance(text, str):
            raise ProviderInvocationError("provider_output_type")
        texts.append(text)
    if not texts:
        raise ProviderInvocationError("provider_output_missing")
    return "".join(texts)
