from __future__ import annotations

from time import perf_counter
from typing import Final, Literal

from pydantic import BaseModel, JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.hashing import hph1
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
    TurnLlmInvocationRowV1,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    DirectProviderSynthesisV1,
    PromptProgramV2,
    ProviderTargetConfigV1,
    ProviderTransportEnvelopeV1,
    direct_synthesis_stage_kind,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
    is_provider_output_failure,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    build_provider_target_from_settings,
)
from market_support_crewai_agent.settings_model import Settings

JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def direct_target_from_settings(
    settings: Settings,
    *,
    stage: str,
    temperature: float,
    max_tokens: int,
) -> tuple[ProviderTargetConfigV1, str | None]:
    if stage == "planner_intent":
        provider = settings.planner_llm_provider
        model = settings.planner_llm_model
        base_url = settings.planner_llm_base_url
        api_key = settings.planner_llm_api_key or (
            settings.llm_api_key
            if (
                provider == settings.llm_provider
                and model == settings.llm_model
                and base_url == settings.llm_base_url
            )
            else None
        )
        target_slot = "planner"
    else:
        provider = settings.llm_provider
        model = settings.llm_model
        base_url = settings.llm_base_url
        api_key = settings.llm_api_key
        target_slot = "composer"
    return (
        build_provider_target_from_settings(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_configured=api_key is not None,
            target_slot=target_slot,
            timeout_seconds=min(settings.llm_timeout_seconds, 30.0),
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        api_key,
    )


def close_direct_journal_success(
    journal: TurnLlmInvocationJournalV1,
    reserved: TurnLlmInvocationRowV1,
    *,
    program: PromptProgramV2,
    synthesis: DirectProviderSynthesisV1,
    target: ProviderTargetConfigV1,
    envelope: ProviderTransportEnvelopeV1,
    response_model: type[BaseModel],
    out1: str,
    latency_ms: int,
    output_bytes: int,
) -> None:
    journal.replace_reserved(
        reserved.invocation_ordinal,
        _closed_direct_row(
            reserved,
            program=program,
            synthesis=synthesis,
            target=target,
            envelope=envelope,
            response_model=response_model,
            status="success",
            error_code=None,
            out1=out1,
            latency_ms=latency_ms,
            output_bytes=output_bytes,
        ),
    )


def close_direct_journal_error(
    journal: TurnLlmInvocationJournalV1,
    reserved: TurnLlmInvocationRowV1,
    *,
    program: PromptProgramV2,
    synthesis: DirectProviderSynthesisV1,
    target: ProviderTargetConfigV1,
    envelope: ProviderTransportEnvelopeV1,
    response_model: type[BaseModel],
    surfaced_code: ProviderFailureCodeV1,
    latency_ms: int,
) -> None:
    output_error = is_provider_output_failure(surfaced_code)
    journal.replace_reserved(
        reserved.invocation_ordinal,
        _closed_direct_row(
            reserved,
            program=program,
            synthesis=synthesis,
            target=target,
            envelope=envelope,
            response_model=response_model,
            status="output_contract_error" if output_error else "transport_error",
            error_code=surfaced_code,
            out1=None,
            latency_ms=latency_ms,
            output_bytes=None,
        ),
    )


def elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _closed_direct_row(
    reserved: TurnLlmInvocationRowV1,
    *,
    program: PromptProgramV2,
    synthesis: DirectProviderSynthesisV1,
    target: ProviderTargetConfigV1,
    envelope: ProviderTransportEnvelopeV1,
    response_model: type[BaseModel],
    status: Literal["success", "output_contract_error", "transport_error"],
    error_code: str | None,
    out1: str | None,
    latency_ms: int,
    output_bytes: int | None,
) -> TurnLlmInvocationRowV1:
    stage_kind = direct_synthesis_stage_kind(synthesis)
    return TurnLlmInvocationRowV1(
        invocation_ordinal=reserved.invocation_ordinal,
        logical_attempt=reserved.logical_attempt,
        stage_kind=reserved.stage_kind,
        program_id=reserved.program_id,
        program_version=program.program_version,
        target_slot=reserved.target_slot,
        scene_key=program.scene_key,
        scene_contract_ref=(
            f"{program.scene_contract_id}@{program.scene_contract_version}"
            if program.scene_contract_id is not None
            and program.scene_contract_version is not None
            else None
        ),
        purpose=reserved.purpose,
        status=status,
        osh1=synthesis.osh1(),
        poh1=envelope.poh1(),
        hph1=hph1(
            synthesis.user_payload.prompt_program_text,
            synthesis.system_payload.agent_execution_spec,
        ),
        prh1=envelope.prh1(),
        input_digest=synthesis.mch1(),
        output_digest=out1,
        error_code=error_code,
        model_family=target.model_family,
        provider_id=envelope.provider_id,
        transport_id=envelope.variant,
        input_schema_version=f"{stage_kind.replace('_', '-')}-input.v1",
        output_schema_version=_output_schema_version(response_model),
        latency_ms=latency_ms,
        input_bytes=len(envelope.model_dump_json().encode("utf-8")),
        output_bytes=output_bytes,
    )


def _output_schema_version(response_model: type[BaseModel]) -> str:
    field = response_model.model_fields.get("contract_version")
    if field is not None:
        default_value: JsonValue = JSON_VALUE_ADAPTER.validate_python(
            field.get_default(call_default_factory=False)
        )
        if isinstance(default_value, str):
            return default_value
    return response_model.__name__
