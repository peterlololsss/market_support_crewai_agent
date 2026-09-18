from __future__ import annotations

import math
from typing import Annotated, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.hashing import (
    canonical_json_bytes,
    hash_canonical_model,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderChatMessageV1,
    ProviderJsonSchemaFormatV1,
)
from market_support_crewai_agent.schemas.base import StrictModel


class _FrozenEnvelopeModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class _ProviderMessageEnvelopeModel(_FrozenEnvelopeModel):
    contract_version: Literal["provider-message-envelope.v1"] = (
        "provider-message-envelope.v1"
    )

    def prh1(self) -> str:
        return hash_canonical_model(
            "provider-visible-prompt.v1",
            self,
            prefix="prh1",
        )

    @property
    def byte_count(self) -> int:
        return len(
            canonical_json_bytes(
                self.model_dump(
                    mode="json",
                    exclude_none=False,
                    exclude_defaults=False,
                )
            )
        )


class ProviderResponseFormatV1(_FrozenEnvelopeModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: ProviderJsonSchemaFormatV1
    provider_output_schema_hash: str = Field(pattern=r"^poh1:[0-9a-f]{64}$")


class ChatGenerationParametersV1(_FrozenEnvelopeModel):
    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: tuple[str, ...] = Field(default=(), max_length=8)
    seed: int | None = None
    tool_choice: Literal["none", "auto", "required"] | None = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("provider_temperature_non_finite")
        return value


class CrewAIChatEnvelopeV1(_ProviderMessageEnvelopeModel):
    kind: Literal["crewai_chat"] = "crewai_chat"
    transport_variant: Literal["crewai_openai_compatible.v1"] = (
        "crewai_openai_compatible.v1"
    )
    provider_id: Literal["openai_compatible"] = "openai_compatible"
    model_family: ModelFamily
    model_name: str = Field(min_length=1, max_length=160)
    messages: tuple[ProviderChatMessageV1, ...] = Field(min_length=1, max_length=64)
    tool_declarations_json: tuple[bytes, ...] = Field(default=(), max_length=16)
    response_format: ProviderResponseFormatV1 | None = None
    generation_parameters: ChatGenerationParametersV1
    crewai_version: str = Field(min_length=1, max_length=80)
    synthesis_template_version: Literal[1] = 1
    transport_adapter_version: Literal[1] = 1


class GeminiTextPartV1(_FrozenEnvelopeModel):
    text: str = Field(max_length=2_000_000)


class GeminiContentV1(_FrozenEnvelopeModel):
    role: Literal["user", "model"]
    parts: tuple[GeminiTextPartV1, ...] = Field(min_length=1, max_length=32)


class GeminiThinkingConfigV1(_FrozenEnvelopeModel):
    include_thoughts: bool | None = None
    thinking_budget: int | None = Field(default=None, ge=0, le=32_768)
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None

    @model_validator(mode="after")
    def validate_budget_or_level(self) -> Self:
        if self.thinking_budget is not None and self.thinking_level is not None:
            raise ValueError("gemini_thinking_budget_level_conflict")
        return self


class GeminiGenerationParametersV1(_FrozenEnvelopeModel):
    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    stop_sequences: tuple[str, ...] = Field(default=(), max_length=8)
    response_mime_type: str = Field(min_length=1, max_length=120)
    safety_settings_json: bytes | None = None
    thinking_config: GeminiThinkingConfigV1 | None = None

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("provider_temperature_non_finite")
        return value


class GeminiGenerateContentEnvelopeV1(_ProviderMessageEnvelopeModel):
    kind: Literal["gemini_generate_content"] = "gemini_generate_content"
    transport_variant: Literal["gemini_generate_content.v1"] = (
        "gemini_generate_content.v1"
    )
    provider_id: Literal["gemini"] = "gemini"
    model_family: ModelFamily
    model_name: str = Field(min_length=1, max_length=160)
    contents: tuple[GeminiContentV1, ...] = Field(min_length=1, max_length=64)
    system_instruction: GeminiContentV1 | None = None
    tools_json: bytes | None = None
    provider_schema_json: bytes
    provider_output_schema_hash: str = Field(pattern=r"^poh1:[0-9a-f]{64}$")
    generation_parameters: GeminiGenerationParametersV1
    google_sdk_version: str = Field(min_length=1, max_length=80)
    synthesis_template_version: Literal[1] = 1
    transport_adapter_version: Literal[1] = 1


ProviderMessageEnvelopeV1 = Annotated[
    CrewAIChatEnvelopeV1 | GeminiGenerateContentEnvelopeV1,
    Field(discriminator="kind"),
]
