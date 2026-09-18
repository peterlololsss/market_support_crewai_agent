from __future__ import annotations

# noqa: SIZE_OK - strict prompt governance DTOs and their packaged-resource resolver.
import hashlib
import json
from importlib.resources import files
from typing import ClassVar, Literal, Self, assert_never

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from market_support_crewai_agent.runtime.hashing import (
    CanonicalValue,
    canonical_json_bytes,
    hash_canonical_model,
    sha256_frame,
)
from market_support_crewai_agent.runtime.prompts.profiles import (
    ModelFamily,
    PromptStage,
    SceneKeyV1,
)
from market_support_crewai_agent.runtime.prompts.registry_models import PromptLayer
from market_support_crewai_agent.runtime.prompts.schema_canonicalization import (
    CANONICAL_JSON_SCHEMA_TRANSFORM_VERSION,
    canonicalize_json_schema,
)
from market_support_crewai_agent.schemas.base import StrictModel

StageKindV1 = Literal[
    "planner_intent",
    "knowledge_composer",
    "smalltalk_composer",
    "alignment_verifier",
    "approved_knowledge_selector",
    "document_product_selector",
    "llm_health_probe",
]
TransportVariantV1 = Literal["openai_chat_completions", "gemini_generate_content"]
ProviderIdV1 = Literal["openai_compatible", "gemini"]
TargetSlotV1 = Literal["planner", "composer", "selector", "health"]
ProviderOutputStatusV1 = Literal[
    "available_text",
    "missing_text",
    "non_string_text",
    "invalid_utf8",
    "oversize_text",
    "output_contract_error",
]


class PromptGovernanceError(ValueError):
    pass


class _FrozenProgramModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class PromptBuildMetadataV1(_FrozenProgramModel):
    contract_version: Literal["prompt-build-metadata.v1"] = "prompt-build-metadata.v1"
    source_resource: str = Field(min_length=1, max_length=240)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PromptProgramSourceV2(_FrozenProgramModel):
    fragment_id: str = Field(min_length=1, max_length=160)
    template_name: str = Field(min_length=1, max_length=240)
    layer: PromptLayer
    priority: int = Field(ge=0)
    purpose: Literal[
        "base",
        "instruction_precedence",
        "domain_schema",
        "scene_presentation",
        "output_schema",
    ] = "domain_schema"
    frh1: str = Field(pattern=r"^frh1:[0-9a-f]{64}$")
    model_families: tuple[str, ...] = Field(min_length=1)


class AgentExecutionSpecV1(_FrozenProgramModel):
    spec_id: str = Field(min_length=1, max_length=160)
    program_id: str = Field(min_length=1, max_length=180)
    stage: PromptStage
    scene_key: SceneKeyV1
    status: Literal["active"]
    role: str = Field(min_length=1, max_length=240)
    goal: str = Field(min_length=1, max_length=500)
    backstory: str = Field(min_length=1, max_length=500)
    task_template: str = Field(min_length=1, max_length=500)
    expected_output_template: str = Field(min_length=1, max_length=500)
    agent_spec_version: int = Field(ge=1)


class PromptProgramV2(_FrozenProgramModel):
    contract_version: Literal["prompt-program.v2"] = "prompt-program.v2"
    program_id: str = Field(min_length=1, max_length=180)
    program_version: str = Field(min_length=1, max_length=40)
    stage: StageKindV1
    scene_key: SceneKeyV1
    status: Literal["active"]
    sources: tuple[PromptProgramSourceV2, ...] = Field(min_length=1)
    scene_contract_id: str | None
    scene_contract_version: str | None
    build_metadata: PromptBuildMetadataV1

    @model_validator(mode="after")
    def validate_program(self) -> Self:
        precedence_count = sum(
            1 for source in self.sources if source.purpose == "instruction_precedence"
        )
        if precedence_count < 1:
            raise PromptGovernanceError("prompt_program_precedence_missing")
        universal_precedence_count = sum(
            1
            for source in self.sources
            if source.fragment_id == "instruction.registered_over_untrusted_data.v1"
            and source.purpose == "instruction_precedence"
        )
        if universal_precedence_count != 1:
            raise PromptGovernanceError("prompt_program_universal_precedence_count")
        scene_count = sum(
            1 for source in self.sources if source.purpose == "scene_presentation"
        )
        match self.scene_key:
            case "scene_neutral.v1":
                if scene_count != 0 or self.scene_contract_id is not None:
                    raise PromptGovernanceError("neutral_program_scene_forbidden")
            case "wecom_group.v1" | "wecom_direct.v1":
                if scene_count != 1 or self.scene_contract_id is None:
                    raise PromptGovernanceError("user_program_scene_required")
            case unreachable:
                assert_never(unreachable)
        return self


class ProviderTargetIdentityV1(_FrozenProgramModel):
    contract_version: Literal["provider-target-config.v1"] = "provider-target-config.v1"
    provider_id: ProviderIdV1
    target_slot: TargetSlotV1
    model_family: ModelFamily
    model_name: str = Field(min_length=1, max_length=160)
    normalized_endpoint: str = Field(min_length=1, max_length=512)
    transport_variant: TransportVariantV1

    @model_validator(mode="after")
    def validate_transport_variant(self) -> Self:
        expected = (
            "openai_chat_completions"
            if self.provider_id == "openai_compatible"
            else "gemini_generate_content"
        )
        if self.transport_variant != expected:
            raise PromptGovernanceError("provider_target_transport_mismatch")
        return self


class ProviderTargetConfigV1(ProviderTargetIdentityV1):
    api_key_configured: bool
    timeout_seconds: float = Field(ge=0.1, le=30.0)
    temperature: float = Field(ge=0.0)
    max_tokens: int = Field(gt=0)
    retry_attempts: Literal[0] = 0
    retry_base_seconds: Literal[0.0] = 0.0
    thinking_config: JsonValue | None = None

    def identity(self) -> ProviderTargetIdentityV1:
        return ProviderTargetIdentityV1(
            provider_id=self.provider_id,
            target_slot=self.target_slot,
            model_family=self.model_family,
            model_name=self.model_name,
            normalized_endpoint=self.normalized_endpoint,
            transport_variant=self.transport_variant,
        )


class ProviderJsonSchemaFormatV1(_FrozenProgramModel):
    name: str = Field(min_length=1, max_length=160)
    strict: Literal[True] = True
    json_schema: JsonValue

    @field_validator("json_schema", mode="before")
    @classmethod
    def canonicalize_schema(cls, value: JsonValue) -> JsonValue:
        return canonicalize_json_schema(value)


class DirectProviderSystemPayloadV1(_FrozenProgramModel):
    contract_version: Literal["direct-provider-system-payload.v1"] = (
        "direct-provider-system-payload.v1"
    )
    agent_execution_spec: AgentExecutionSpecV1
    output_schema: ProviderJsonSchemaFormatV1
    instruction_precedence: Literal[
        "registered_instructions_over_untrusted_stage_data.v1"
    ] = "registered_instructions_over_untrusted_stage_data.v1"


class DirectProviderUserPayloadV1(_FrozenProgramModel):
    contract_version: Literal["direct-provider-user-payload.v1"] = (
        "direct-provider-user-payload.v1"
    )
    prompt_program_id: str = Field(min_length=1, max_length=180)
    prompt_program_version: str = Field(min_length=1, max_length=40)
    prompt_program_text: str = Field(min_length=1, max_length=2_000_000)
    stage_input: JsonValue


class ProviderChatMessageV1(_FrozenProgramModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(max_length=2_000_000)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_tool_call_id(self) -> Self:
        if self.role == "tool" and self.tool_call_id is None:
            raise PromptGovernanceError("provider_tool_message_id_required")
        if self.role != "tool" and self.tool_call_id is not None:
            raise PromptGovernanceError("provider_tool_message_id_forbidden")
        return self


class DirectProviderSynthesisV1(_FrozenProgramModel):
    contract_version: Literal["direct-provider-synthesis.v1"] = (
        "direct-provider-synthesis.v1"
    )
    system_payload: DirectProviderSystemPayloadV1
    user_payload: DirectProviderUserPayloadV1
    messages: tuple[ProviderChatMessageV1, ProviderChatMessageV1]

    @model_validator(mode="after")
    def validate_messages(self) -> Self:
        expected = (
            ProviderChatMessageV1(
                role="system",
                content=canonical_json_bytes(
                    self.system_payload.model_dump(
                        mode="json",
                        exclude_none=False,
                        exclude_defaults=False,
                    )
                ).decode("utf-8"),
            ),
            ProviderChatMessageV1(
                role="user",
                content=canonical_json_bytes(
                    self.user_payload.model_dump(
                        mode="json",
                        exclude_none=False,
                        exclude_defaults=False,
                    )
                ).decode("utf-8"),
            ),
        )
        if self.messages != expected:
            raise PromptGovernanceError("direct_provider_messages_mismatch")
        return self

    def osh1(self) -> str:
        return sha256_frame(
            "output-schema.v1",
            self.system_payload.output_schema.json_schema,
            prefix="osh1",
        )

    def mch1(self) -> str:
        return sha256_frame(
            "model-visible-context.v1",
            self.user_payload.stage_input,
            prefix="mch1",
        )


class ProviderTransportEnvelopeV1(_FrozenProgramModel):
    contract_version: Literal["provider-transport-envelope.v1"] = (
        "provider-transport-envelope.v1"
    )
    variant: TransportVariantV1
    stage_kind: StageKindV1
    target_slot: TargetSlotV1
    provider_id: ProviderIdV1
    model_family: ModelFamily
    model: str
    provider_schema_transform_version: Literal[
        "openai-response-format.v1", "gemini-provider-schema.v1"
    ]
    canonical_schema_transform_version: Literal[
        "canonical-json-schema-strip-docs.v1"
    ] = CANONICAL_JSON_SCHEMA_TRANSFORM_VERSION
    canonical_output_schema: JsonValue
    provider_output_schema: JsonValue
    body: JsonValue

    @field_validator("canonical_output_schema", "provider_output_schema", mode="before")
    @classmethod
    def canonicalize_schema_fields(cls, value: JsonValue) -> JsonValue:
        return canonicalize_json_schema(value)

    def osh1(self) -> str:
        return sha256_frame(
            "output-schema.v1",
            self.canonical_output_schema,
            prefix="osh1",
        )

    def poh1(self) -> str:
        return sha256_frame(
            "provider-output-schema.v1",
            {
                "model_family": self.model_family,
                "provider_id": self.provider_id,
                "provider_output_schema": self.provider_output_schema,
                "schema_transform_version": self.provider_schema_transform_version,
            },
            prefix="poh1",
        )

    def prh1(self) -> str:
        return hash_canonical_model("provider-visible-prompt.v1", self, prefix="prh1")


class ProviderOutputCaptureV1(_FrozenProgramModel):
    contract_version: Literal["provider-output-capture.v1"] = (
        "provider-output-capture.v1"
    )
    status: ProviderOutputStatusV1
    text: str | None = None
    byte_count: int = Field(ge=0)
    error_code: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def validate_capture(self) -> Self:
        match self.status:
            case "available_text":
                if self.text is None or self.error_code is not None:
                    raise PromptGovernanceError("provider_output_available_invalid")
            case (
                "missing_text"
                | "non_string_text"
                | "invalid_utf8"
                | "oversize_text"
                | "output_contract_error"
            ):
                if self.text is not None or self.error_code is None:
                    raise PromptGovernanceError("provider_output_error_invalid")
            case unreachable:
                assert_never(unreachable)
        return self

    def out1(self) -> str:
        if self.status != "available_text" or self.text is None:
            raise PromptGovernanceError("provider_output_text_hash_unavailable")
        digest = hashlib.sha256(
            b"provider-output-text.v1\0" + self.text.encode("utf-8")
        ).hexdigest()
        return f"out1:{digest}"


class DirectProviderFatalStateV1(_FrozenProgramModel):
    contract_version: Literal["direct-provider-fatal-state.v1"] = (
        "direct-provider-fatal-state.v1"
    )
    exit_code: Literal[70]
    reason_code: Literal["provider_io_restore_unrecoverable"]


def synthesize_direct_provider_messages(
    *,
    program: PromptProgramV2,
    execution_spec: AgentExecutionSpecV1,
    prompt_text: str,
    stage_input: BaseModel,
    output_schema: ProviderJsonSchemaFormatV1,
) -> DirectProviderSynthesisV1:
    if (
        execution_spec.program_id != program.program_id
        or execution_spec.stage != program.stage
        or execution_spec.scene_key != program.scene_key
    ):
        raise PromptGovernanceError("direct_provider_execution_spec_mismatch")
    stage_input_json = stage_input.model_dump(
        mode="json",
        exclude_none=False,
        exclude_defaults=False,
    )
    contract_version = stage_input_json.get("contract_version")
    expected_stage = {
        "planner-prompt-input.v1": "planner_intent",
        "knowledge-composer-input.v1": "knowledge_composer",
        "smalltalk-composer-input.v1": "smalltalk_composer",
        "alignment-verifier-input.v1": "alignment_verifier",
        "approved-knowledge-selector-input.v1": "approved_knowledge_selector",
        "document-product-selector-input.v1": "document_product_selector",
        "llm-health-probe-input.v1": "llm_health_probe",
    }.get(contract_version)
    if expected_stage != program.stage:
        raise PromptGovernanceError("direct_provider_stage_input_mismatch")
    system_payload = DirectProviderSystemPayloadV1(
        agent_execution_spec=execution_spec,
        output_schema=output_schema,
    )
    user_payload = DirectProviderUserPayloadV1(
        prompt_program_id=program.program_id,
        prompt_program_version=program.program_version,
        prompt_program_text=prompt_text,
        stage_input=stage_input_json,
    )
    messages = (
        ProviderChatMessageV1(
            role="system",
            content=canonical_json_bytes(
                system_payload.model_dump(
                    mode="json",
                    exclude_none=False,
                    exclude_defaults=False,
                )
            ).decode("utf-8"),
        ),
        ProviderChatMessageV1(
            role="user",
            content=canonical_json_bytes(
                user_payload.model_dump(
                    mode="json",
                    exclude_none=False,
                    exclude_defaults=False,
                )
            ).decode("utf-8"),
        ),
    )
    return DirectProviderSynthesisV1(
        system_payload=system_payload,
        user_payload=user_payload,
        messages=messages,
    )


def direct_synthesis_stage_kind(
    synthesis: DirectProviderSynthesisV1,
) -> StageKindV1:
    stage_input = synthesis.user_payload.stage_input
    if not isinstance(stage_input, dict):
        raise PromptGovernanceError("direct_provider_stage_input_mismatch")
    contract_version = stage_input.get("contract_version")
    stage = {
        "planner-prompt-input.v1": "planner_intent",
        "knowledge-composer-input.v1": "knowledge_composer",
        "smalltalk-composer-input.v1": "smalltalk_composer",
        "alignment-verifier-input.v1": "alignment_verifier",
        "approved-knowledge-selector-input.v1": "approved_knowledge_selector",
        "document-product-selector-input.v1": "document_product_selector",
        "llm-health-probe-input.v1": "llm_health_probe",
    }.get(contract_version)
    if stage is None:
        raise PromptGovernanceError("direct_provider_stage_input_mismatch")
    return stage


def load_prompt_program_registry_v2() -> tuple[PromptProgramV2, ...]:
    resource = files("market_support_crewai_agent.runtime.prompts.resources").joinpath(
        "prompt_program_registry.v2.json"
    )
    metadata = PromptBuildMetadataV1(
        source_resource="prompt_program_registry.v2.json",
        source_sha256=hashlib.sha256(resource.read_bytes()).hexdigest(),
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return tuple(
        PromptProgramV2.model_validate(row | {"build_metadata": metadata})
        for row in payload["programs"]
    )


def load_agent_execution_specs_v1() -> tuple[AgentExecutionSpecV1, ...]:
    resource = files("market_support_crewai_agent.runtime.prompts.resources").joinpath(
        "prompt_agent_execution_specs.v1.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return tuple(AgentExecutionSpecV1.model_validate(row) for row in payload["specs"])


def resolve_agent_execution_spec_v1(program_id: str) -> AgentExecutionSpecV1:
    specs = tuple(
        spec
        for spec in load_agent_execution_specs_v1()
        if spec.program_id == program_id
    )
    if len(specs) != 1:
        raise PromptGovernanceError("prompt_program_v2_execution_spec_missing")
    return specs[0]


def resolve_active_prompt_program_v2(
    *,
    stage: StageKindV1,
    scene_key: SceneKeyV1,
) -> tuple[PromptProgramV2, AgentExecutionSpecV1]:
    programs = tuple(
        program
        for program in load_prompt_program_registry_v2()
        if program.stage == stage and program.scene_key == scene_key
    )
    if len(programs) != 1:
        raise PromptGovernanceError("prompt_program_v2_authority_missing")
    program = programs[0]
    specs = tuple(
        spec
        for spec in load_agent_execution_specs_v1()
        if spec.program_id == program.program_id
        and spec.stage == stage
        and spec.scene_key == scene_key
    )
    if len(specs) != 1:
        raise PromptGovernanceError("prompt_program_v2_execution_spec_missing")
    return program, specs[0]


def active_prompt_source_ids_v2(
    program: PromptProgramV2,
    model_family: str,
) -> tuple[str, ...]:
    source_ids = tuple(
        source.fragment_id
        for source in sorted(program.sources, key=lambda source: source.priority)
        if model_family in source.model_families
    )
    if not source_ids:
        raise PromptGovernanceError("prompt_program_v2_model_family_missing")
    return source_ids


def require_active_prompt_program_v2(
    *,
    program_id: str,
    stage: StageKindV1,
    scene_key: SceneKeyV1,
    scene_contract_id: str | None,
    scene_contract_version: str | None,
) -> PromptProgramV2:
    program, _spec = resolve_active_prompt_program_v2(
        stage=stage,
        scene_key=scene_key,
    )
    if (
        program.program_id != program_id
        or program.scene_contract_id != scene_contract_id
        or program.scene_contract_version != scene_contract_version
    ):
        raise PromptGovernanceError("prompt_program_v2_authority_mismatch")
    return program
