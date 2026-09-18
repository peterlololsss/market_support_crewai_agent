from __future__ import annotations

from importlib.metadata import version

import pytest
from google.genai.types import GenerateContentConfig
from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.hashing import (
    canonical_json_bytes,
    sha256_frame,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    GeminiGenerateContentAdapterV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.io import (
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    GeminiContentV1,
    GeminiGenerateContentEnvelopeV1,
    GeminiGenerationParametersV1,
    GeminiTextPartV1,
)
from tests.helpers.crewai_adapter import make_agent_adapter, make_llm_adapter
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.unit.orchestration._crewai_io_support import (
    canonical_value,
    planner_program,
    run_async,
)

_JSON_MAPPING: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_INT_OR_NONE: TypeAdapter[int | None] = TypeAdapter(int | None)


def _gemini_agent(
    captured_requests: list[GeminiGenerateContentAdapterV1],
) -> CrewAIAgentAdapterV1:
    def generate(request: GeminiGenerateContentAdapterV1) -> JsonValue:
        captured_requests.append(request)
        return {
            "text": make_weekly_plan_spec().model_dump_json(),
            "usage_metadata": None,
        }

    return make_agent_adapter(
        llm=make_llm_adapter(
            provider="gemini",
            model="gemini-3-flash-preview",
            temperature=0.1,
            max_output_tokens=6000,
            stop_sequences=(),
            gemini_generate=generate,
        )
    )


def test_run_crewai_kickoff_uses_gemini_structured_schema(
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured_requests: list[GeminiGenerateContentAdapterV1] = []
    agent = _gemini_agent(captured_requests)
    caplog.set_level(
        "INFO",
        logger="market_support_crewai_agent.runtime.integrations.crewai.observability",
    )
    journal = TurnLlmInvocationJournalV1()

    result, _ = run_async(
        lambda: run_crewai_kickoff(
            agent,
            planner_program(),
            timeout_seconds=10,
            journal=journal,
        )
    )

    config = GenerateContentConfig.model_validate(captured_requests[0].config)
    schema = config.response_json_schema
    assert schema is not None
    assert schema["properties"]["plan_units"]["type"] == "array"
    assert config.response_mime_type == "application/json"
    assert result.pydantic is not None
    assert journal.rows[0].prh1 is not None
    assert journal.rows[0].output_digest is not None
    assert "stage=planner_intent mode=gemini_structured" in caplog.text
    assert "gemini-3-flash-preview" not in caplog.text


def test_run_crewai_kickoff_hashes_exact_gemini_sdk_arguments() -> None:
    captured_requests: list[GeminiGenerateContentAdapterV1] = []
    agent = _gemini_agent(captured_requests)
    journal = TurnLlmInvocationJournalV1()

    _ = run_async(
        lambda: run_crewai_kickoff(
            agent,
            planner_program(),
            timeout_seconds=10,
            journal=journal,
        )
    )

    actual = captured_requests[0]
    config = GenerateContentConfig.model_validate(actual.config)
    schema = canonical_value(_JSON_MAPPING.validate_python(config.response_json_schema))
    expected_poh1 = sha256_frame(
        "provider-output-schema.v1",
        {
            "model_family": "generic",
            "provider_id": "gemini",
            "provider_output_schema": schema,
            "schema_transform_version": "gemini-provider-schema.v1",
        },
        prefix="poh1",
    )
    expected = GeminiGenerateContentEnvelopeV1(
        model_family="generic",
        model_name=actual.model,
        contents=(
            GeminiContentV1(
                role="user",
                parts=(GeminiTextPartV1(text=actual.contents),),
            ),
        ),
        provider_schema_json=canonical_json_bytes(schema),
        provider_output_schema_hash=expected_poh1,
        generation_parameters=GeminiGenerationParametersV1(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=_INT_OR_NONE.validate_python(config.top_k),
            max_output_tokens=config.max_output_tokens,
            stop_sequences=tuple(config.stop_sequences or ()),
            response_mime_type=_require_response_mime_type(config),
        ),
        google_sdk_version=version("google-genai"),
    )
    assert journal.rows[0].prh1 == expected.prh1()


def _require_response_mime_type(config: GenerateContentConfig) -> str:
    response_mime_type = config.response_mime_type
    if response_mime_type is None:
        raise AssertionError("response_mime_type_missing")
    return response_mime_type
