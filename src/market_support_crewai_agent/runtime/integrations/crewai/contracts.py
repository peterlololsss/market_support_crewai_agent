from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from market_support_crewai_agent.runtime.prompts.schema_canonicalization import (
    canonicalize_json_schema,
)
from market_support_crewai_agent.schemas.base import StrictModel


class CrewAITransportInvariantError(ValueError):
    pass


class HealthInvocationLogContextV1(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    target_slot: Literal["composer", "planner"]
    htk1: str = Field(pattern=r"^htk1:[A-Za-z0-9_-]{22}$")


type ProviderRawTextV1 = str | bytes | int | float | bool | None
type CrewAICompletionValueV1 = JsonValue | BaseModel
type CrewAICompletionInvokeV1 = Callable[..., CrewAICompletionValueV1]
type RestoreCrewAIRequestCaptureV1 = Callable[[], None]
type ReplaceCrewAICompletionV1 = Callable[[CrewAICompletionInvokeV1 | None], None]
type CrewAIGeminiSyncClientFactoryV1 = Callable[[], "GeminiSyncClientV1"]
type CrewAIKickoffCallableV1 = Callable[
    [str, type[BaseModel]], Awaitable["CrewAIKickoffOutputV1"]
]


@dataclass(frozen=True, slots=True)
class CrewAIProviderResultV1:
    raw: ProviderRawTextV1
    pydantic: BaseModel | JsonValue | None
    agent_role: str
    usage_metrics: JsonValue | None


@dataclass(frozen=True, slots=True)
class CrewAIKickoffOutputV1:
    raw: ProviderRawTextV1
    pydantic: BaseModel | JsonValue | None


class GeminiGenerateContentAdapterV1(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    model: str
    contents: str
    config: BaseModel


@dataclass(frozen=True, slots=True)
class GeminiModelsClientV1:
    generate_content_fn: Callable[[GeminiGenerateContentAdapterV1], JsonValue]

    def generate_content(
        self,
        *,
        model: str,
        contents: str,
        config: BaseModel,
    ) -> JsonValue:
        return self.generate_content_fn(
            GeminiGenerateContentAdapterV1(
                model=model,
                contents=contents,
                config=config,
            )
        )


@dataclass(frozen=True, slots=True)
class GeminiSyncClientV1:
    models: GeminiModelsClientV1


@dataclass(frozen=True, slots=True)
class CrewAICompletionCaptureTargetV1:
    current: Callable[[], CrewAICompletionInvokeV1 | None]
    replace: ReplaceCrewAICompletionV1


@dataclass(frozen=True, slots=True)
class CrewAILlmAdapterV1:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    client_params: Mapping[str, JsonValue]
    max_tokens: int | None
    max_output_tokens: int | None
    timeout: float | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    stop_sequences: Sequence[str] | None
    thinking_config: JsonValue | None
    max_retries: int | None = None
    gemini_retry_attempts: int | None = None
    configure_openai_client: Callable[[OpenAI], None] | None = None
    completion_capture: CrewAICompletionCaptureTargetV1 | None = None
    gemini_sync_client_factory: CrewAIGeminiSyncClientFactoryV1 | None = None


@dataclass(frozen=True, slots=True)
class CrewAIAgentAdapterV1:
    role: str
    llm: CrewAILlmAdapterV1
    kickoff_async: CrewAIKickoffCallableV1
    planning: bool = False
    allow_delegation: bool = False
    inject_date: bool = False
    max_retry_limit: int = 0

    async def kickoff(
        self,
        prompt: str,
        *,
        response_format: type[BaseModel],
    ) -> CrewAIKickoffOutputV1:
        return await self.kickoff_async(prompt, response_format)


def gemini_json_schema(model: type[BaseModel]) -> JsonValue:
    schema = canonicalize_json_schema(model.model_json_schema())
    if not isinstance(schema, dict):
        raise CrewAITransportInvariantError("provider_output_schema_invalid")
    defs_raw = schema.get("$defs", {})
    defs: dict[str, JsonValue] = defs_raw if isinstance(defs_raw, dict) else {}
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

    def convert(node: JsonValue) -> JsonValue:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node
        ref_value = node.get("$ref")
        if isinstance(ref_value, str):
            name = ref_value.rsplit("/", 1)[-1]
            ref_schema = defs.get(name)
            if not isinstance(ref_schema, dict):
                raise CrewAITransportInvariantError("provider_ref_schema_invalid")
            merged: dict[str, JsonValue] = copy.deepcopy(ref_schema)
            merged.update({key: value for key, value in node.items() if key != "$ref"})
            return convert(merged)
        choices_raw = node.get("anyOf")
        if isinstance(choices_raw, list):
            choices = tuple(item for item in choices_raw if isinstance(item, dict))
            non_null = [item for item in choices if item.get("type") != "null"]
            if len(non_null) == 1 and len(non_null) != len(choices):
                converted = convert(non_null[0])
                if not isinstance(converted, dict):
                    return converted
                type_value = converted.get("type")
                types_list: list[JsonValue] = (
                    [type_value] if isinstance(type_value, str) else []
                )
                converted["type"] = [*types_list, "null"]
                return converted
            return convert(non_null[0] if non_null else choices[0])
        output: dict[str, JsonValue] = {}
        for key, value in node.items():
            if (
                key in drop_keys
                or key in drop_constraints
                or key == "additionalProperties"
            ):
                continue
            if key == "const":
                output["enum"] = [value]
            elif key == "properties":
                if not isinstance(value, dict):
                    raise CrewAITransportInvariantError("provider_properties_invalid")
                properties = {
                    prop: convert(prop_schema) for prop, prop_schema in value.items()
                }
                output["properties"] = properties
                output["propertyOrdering"] = list(properties)
            else:
                output[key] = convert(value)
        return output

    return convert(schema)
