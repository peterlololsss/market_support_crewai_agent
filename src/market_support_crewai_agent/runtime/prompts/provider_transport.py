from __future__ import annotations

from typing import Final

from pydantic import BaseModel

from market_support_crewai_agent.runtime.hashing import (
    CanonicalValue,
    canonical_json_bytes,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    DirectProviderSynthesisV1,
    ProviderOutputCaptureV1,
    ProviderTargetConfigV1,
    ProviderTransportEnvelopeV1,
    direct_synthesis_stage_kind,
)


AGGREGATE_PROVIDER_INPUT_LIMIT_BYTES: Final = 2_000_000
_RESERVED_TRANSPORT_BYTES: Final = 4096


class ProviderTransportError(ValueError):
    pass


def build_provider_transport_envelope(
    synthesis: DirectProviderSynthesisV1,
    target: ProviderTargetConfigV1,
) -> ProviderTransportEnvelopeV1:
    _assert_within_aggregate_budget(synthesis)
    stage_kind = direct_synthesis_stage_kind(synthesis)
    match target.transport_variant:
        case "openai_chat_completions":
            body = _openai_body(synthesis, target)
            return ProviderTransportEnvelopeV1(
                variant="openai_chat_completions",
                stage_kind=stage_kind,
                target_slot=target.target_slot,
                provider_id=target.provider_id,
                model_family=target.model_family,
                model=target.model_name,
                provider_schema_transform_version="openai-response-format.v1",
                canonical_output_schema=(
                    synthesis.system_payload.output_schema.json_schema
                ),
                provider_output_schema=body["response_format"]["json_schema"],
                body=body,
            )
        case "gemini_generate_content":
            body = _gemini_body(synthesis, target)
            return ProviderTransportEnvelopeV1(
                variant="gemini_generate_content",
                stage_kind=stage_kind,
                target_slot=target.target_slot,
                provider_id=target.provider_id,
                model_family=target.model_family,
                model=target.model_name,
                provider_schema_transform_version="gemini-provider-schema.v1",
                canonical_output_schema=(
                    synthesis.system_payload.output_schema.json_schema
                ),
                provider_output_schema=body["config"]["response_json_schema"],
                body=body,
            )
        case unreachable:
            raise AssertionError(f"unreachable transport variant: {unreachable}")


def capture_provider_text_output(
    value: str | bytes | int | float | bool | None,
    *,
    response_model: type[BaseModel],
    max_bytes: int = AGGREGATE_PROVIDER_INPUT_LIMIT_BYTES,
) -> ProviderOutputCaptureV1:
    match value:
        case None:
            return ProviderOutputCaptureV1(
                status="missing_text",
                byte_count=0,
                error_code="provider_output_missing",
            )
        case bytes() as raw_bytes:
            try:
                text = raw_bytes.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                return ProviderOutputCaptureV1(
                    status="invalid_utf8",
                    byte_count=len(raw_bytes),
                    error_code="provider_output_encoding",
                )
            return _capture_string(
                text, response_model=response_model, max_bytes=max_bytes
            )
        case str() as text:
            return _capture_string(
                text, response_model=response_model, max_bytes=max_bytes
            )
        case int() | float() | bool():
            return ProviderOutputCaptureV1(
                status="non_string_text",
                byte_count=0,
                error_code="provider_output_type",
            )
        case unreachable:
            raise AssertionError(f"unreachable provider output variant: {unreachable}")


def _openai_body(
    synthesis: DirectProviderSynthesisV1,
    target: ProviderTargetConfigV1,
) -> CanonicalValue:
    output_schema = synthesis.system_payload.output_schema
    body: CanonicalValue = {
        "model": target.model_name,
        "messages": [message.model_dump(mode="json") for message in synthesis.messages],
        "temperature": target.temperature,
        "max_tokens": target.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.name,
                "schema": output_schema.json_schema,
                "strict": output_schema.strict,
            },
        },
    }
    return body


def _gemini_body(
    synthesis: DirectProviderSynthesisV1,
    target: ProviderTargetConfigV1,
) -> CanonicalValue:
    output_schema = synthesis.system_payload.output_schema
    config: CanonicalValue = {
        "temperature": target.temperature,
        "max_output_tokens": target.max_tokens,
        "response_mime_type": "application/json",
        "response_json_schema": output_schema.json_schema,
    }
    if target.thinking_config is not None:
        config = config | {"thinking_config": target.thinking_config}
    return {
        "model": target.model_name,
        "system_instruction": synthesis.messages[0].content,
        "contents": [
            {"role": "user", "parts": [{"text": synthesis.messages[1].content}]},
        ],
        "config": config,
    }


def _assert_within_aggregate_budget(synthesis: DirectProviderSynthesisV1) -> None:
    payload_bytes = len(
        canonical_json_bytes(synthesis.model_dump(mode="json", exclude_none=False))
    )
    if payload_bytes + _RESERVED_TRANSPORT_BYTES > AGGREGATE_PROVIDER_INPUT_LIMIT_BYTES:
        raise ProviderTransportError("provider_aggregate_input_budget_exceeded")


def _capture_string(
    text: str,
    *,
    response_model: type[BaseModel],
    max_bytes: int,
) -> ProviderOutputCaptureV1:
    if _has_surrogate(text):
        return ProviderOutputCaptureV1(
            status="invalid_utf8",
            byte_count=0,
            error_code="provider_output_encoding",
        )
    byte_count = len(text.encode("utf-8"))
    if byte_count > max_bytes:
        return ProviderOutputCaptureV1(
            status="oversize_text",
            byte_count=byte_count,
            error_code="provider_output_too_large",
        )
    try:
        response_model.model_validate_json(text)
    except ValueError:
        return ProviderOutputCaptureV1(
            status="output_contract_error",
            byte_count=byte_count,
            error_code="provider_output_contract",
        )
    return ProviderOutputCaptureV1(
        status="available_text",
        text=text,
        byte_count=byte_count,
    )


def _has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(char) <= 0xDFFF for char in text)
