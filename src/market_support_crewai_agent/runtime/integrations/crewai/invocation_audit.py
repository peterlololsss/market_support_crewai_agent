from __future__ import annotations

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAITransportInvariantError,
)
from market_support_crewai_agent.runtime.integrations.crewai.request_capture import (
    optional_request_capture,
    request_transport_id,
)
from market_support_crewai_agent.runtime.integrations.crewai.targeting import (
    input_schema_version,
    output_schema_version,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
    TurnLlmInvocationRowV1,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderOutputCaptureV1,
    ProviderTransportEnvelopeV1,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    ProviderMessageEnvelopeV1,
)


def close_journal_row(
    journal: TurnLlmInvocationJournalV1,
    reserved: TurnLlmInvocationRowV1,
    *,
    envelope: ProviderTransportEnvelopeV1,
    request_capture: ProviderMessageEnvelopeV1,
    output_capture: ProviderOutputCaptureV1,
    prompt_program: PromptProgram,
    latency_ms: int,
) -> None:
    journal.replace_reserved(
        reserved.invocation_ordinal,
        TurnLlmInvocationRowV1(
            invocation_ordinal=reserved.invocation_ordinal,
            logical_attempt=reserved.logical_attempt,
            stage_kind=reserved.stage_kind,
            program_id=reserved.program_id,
            program_version=prompt_program.program_version,
            target_slot=reserved.target_slot,
            scene_key=prompt_program.scene_key,
            scene_contract_ref=scene_contract_ref(prompt_program),
            purpose=reserved.purpose,
            status="success"
            if output_capture.status == "available_text"
            else "output_contract_error",
            osh1=envelope.osh1(),
            poh1=envelope.poh1(),
            hph1=prompt_program.hph1,
            prh1=request_capture.prh1(),
            input_digest=required_mch1(prompt_program),
            output_digest=output_capture.out1()
            if output_capture.status == "available_text"
            else None,
            error_code=output_capture.error_code,
            model_family=prompt_program.profile.model_family,
            provider_id=envelope.provider_id,
            transport_id=request_transport_id(request_capture),
            input_schema_version=input_schema_version(reserved.stage_kind),
            output_schema_version=output_schema_version(
                prompt_program.profile.response_model
            ),
            latency_ms=latency_ms,
            input_bytes=request_capture.byte_count,
            output_bytes=output_capture.byte_count
            if output_capture.status == "available_text"
            else None,
        ),
    )


def close_journal_transport_error(
    journal: TurnLlmInvocationJournalV1,
    reserved: TurnLlmInvocationRowV1,
    *,
    prompt_program: PromptProgram,
    envelope: ProviderTransportEnvelopeV1,
    request_capture: ProviderMessageEnvelopeV1 | None,
    error_code: ProviderFailureCodeV1,
    latency_ms: int,
) -> None:
    journal.replace_reserved(
        reserved.invocation_ordinal,
        TurnLlmInvocationRowV1(
            invocation_ordinal=reserved.invocation_ordinal,
            logical_attempt=reserved.logical_attempt,
            stage_kind=reserved.stage_kind,
            program_id=reserved.program_id,
            program_version=prompt_program.program_version,
            target_slot=reserved.target_slot,
            scene_key=prompt_program.scene_key,
            scene_contract_ref=scene_contract_ref(prompt_program),
            purpose=reserved.purpose,
            status="transport_error",
            osh1=envelope.osh1(),
            poh1=envelope.poh1(),
            hph1=prompt_program.hph1,
            prh1=request_capture.prh1() if request_capture is not None else None,
            input_digest=required_mch1(prompt_program),
            error_code=error_code,
            model_family=prompt_program.profile.model_family,
            provider_id=envelope.provider_id,
            transport_id=request_transport_id(request_capture)
            if request_capture is not None
            else envelope.variant,
            input_schema_version=input_schema_version(reserved.stage_kind),
            output_schema_version=output_schema_version(
                prompt_program.profile.response_model
            ),
            latency_ms=latency_ms,
            input_bytes=(
                request_capture.byte_count
                if request_capture is not None
                else len(prompt_program.prompt_text.encode("utf-8"))
            ),
        ),
    )


def close_transport_error(
    journal: TurnLlmInvocationJournalV1 | None,
    reserved: TurnLlmInvocationRowV1 | None,
    *,
    prompt_program: PromptProgram,
    envelope: ProviderTransportEnvelopeV1,
    captures: list[ProviderMessageEnvelopeV1],
    error_code: ProviderFailureCodeV1,
    latency_ms: int,
) -> None:
    capture = optional_request_capture(captures)
    if reserved is None or journal is None:
        return
    close_journal_transport_error(
        journal,
        reserved,
        prompt_program=prompt_program,
        envelope=envelope,
        request_capture=capture,
        error_code=error_code,
        latency_ms=latency_ms,
    )


def scene_contract_ref(prompt_program: PromptProgram) -> str | None:
    if (
        prompt_program.scene_contract_id is None
        or prompt_program.scene_contract_version is None
    ):
        return None
    return f"{prompt_program.scene_contract_id}@{prompt_program.scene_contract_version}"


def required_mch1(prompt_program: PromptProgram) -> str:
    if prompt_program.mch1 is None:
        raise CrewAITransportInvariantError("model_visible_context_hash_missing")
    return prompt_program.mch1
