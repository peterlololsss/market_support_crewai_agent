from __future__ import annotations

from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    LlmHealthProbeOutputV1,
)
from market_support_crewai_agent.runtime.prompts.provider_transport import (
    capture_provider_text_output,
)


def test_available_text_capture_validates_contract_and_hashes_out1() -> None:
    # Given: a strict provider output string matching the requested schema.
    raw = '{"contract_version":"llm-health-probe-output.v1","ok":true}'

    # When: output is captured at the provider boundary.
    capture = capture_provider_text_output(raw, response_model=LlmHealthProbeOutputV1)

    # Then: available text is bounded and fingerprinted.
    assert capture.status == "available_text"
    assert capture.byte_count == len(raw.encode("utf-8"))
    assert capture.out1().startswith("out1:")


def test_output_capture_maps_missing_type_encoding_oversize_and_contract_errors() -> (
    None
):
    # Given/When/Then: every closed output status maps to a sanitized code.
    assert (
        capture_provider_text_output(
            None, response_model=LlmHealthProbeOutputV1
        ).error_code
        == "provider_output_missing"
    )
    assert (
        capture_provider_text_output(
            7, response_model=LlmHealthProbeOutputV1
        ).error_code
        == "provider_output_type"
    )
    assert (
        capture_provider_text_output(
            b"\xff", response_model=LlmHealthProbeOutputV1
        ).error_code
        == "provider_output_encoding"
    )
    assert (
        capture_provider_text_output(
            '{"contract_version":"llm-health-probe-output.v1","ok":true}',
            response_model=LlmHealthProbeOutputV1,
            max_bytes=1,
        ).error_code
        == "provider_output_too_large"
    )
    assert (
        capture_provider_text_output(
            "{}", response_model=LlmHealthProbeOutputV1
        ).error_code
        == "provider_output_contract"
    )
