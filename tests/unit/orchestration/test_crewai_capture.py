from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from market_support_crewai_agent.runtime.integrations.crewai import (
    request_capture,
    targeting,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAICompletionInvokeV1,
    CrewAICompletionValueV1,
    CrewAITransportInvariantError,
    RestoreCrewAIRequestCaptureV1,
)
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderChatMessageV1,
    ProviderJsonSchemaFormatV1,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    ChatGenerationParametersV1,
    CrewAIChatEnvelopeV1,
    ProviderMessageEnvelopeV1,
    ProviderResponseFormatV1,
)
from tests.helpers.crewai_adapter import (
    invoke_completion,
    make_agent_adapter,
    make_llm_adapter,
)
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.unit.orchestration._crewai_io_support import planner_program

_JSON_MAPPING = TypeAdapter(dict[str, JsonValue])
_PROVIDER_ENVELOPE: TypeAdapter[ProviderMessageEnvelopeV1] = TypeAdapter(
    ProviderMessageEnvelopeV1
)


def _capture_target(
    completion: CrewAICompletionInvokeV1,
) -> tuple[
    CrewAIAgentAdapterV1,
    list[ProviderMessageEnvelopeV1],
    RestoreCrewAIRequestCaptureV1,
]:
    llm = make_llm_adapter(
        provider="openai",
        model="model",
        completion=completion,
    )
    agent = make_agent_adapter(llm=llm)
    captures: list[ProviderMessageEnvelopeV1] = []
    program = planner_program()
    restore = request_capture.install_crewai_request_capture(
        agent,
        program,
        targeting.crewai_transport_envelope(agent, program),
        captures,
    )
    return agent, captures, restore


def test_crewai_capture_rejects_unknown_generation_field_before_sdk_dispatch() -> None:
    dispatched_params: list[Mapping[str, JsonValue]] = []

    def completion(
        *,
        params: Mapping[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> CrewAICompletionValueV1:
        del available_functions, from_task, from_agent, response_model
        dispatched_params.append(params)
        return make_weekly_plan_spec()

    agent, captures, restore = _capture_target(completion)
    try:
        with pytest.raises(
            CrewAITransportInvariantError,
            match="provider_request_field_unmodeled",
        ):
            _ = invoke_completion(
                agent.llm,
                params={
                    "model": "model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "future_prompt_field": "must-not-pass",
                },
                response_model=PlanSpec,
            )
    finally:
        restore()

    assert agent.llm.completion_capture is not None
    assert agent.llm.completion_capture.current() is completion
    assert dispatched_params == []
    assert captures == []


def test_provider_message_envelope_union_rejects_unknown_fields_and_variants() -> None:
    envelope = CrewAIChatEnvelopeV1(
        model_family="generic",
        model_name="model",
        messages=(ProviderChatMessageV1(role="user", content="hello"),),
        response_format=ProviderResponseFormatV1(
            json_schema=ProviderJsonSchemaFormatV1(
                name="PlanSpec",
                json_schema={"type": "object"},
            ),
            provider_output_schema_hash="poh1:" + "0" * 64,
        ),
        generation_parameters=ChatGenerationParametersV1(),
        crewai_version="test",
    )
    payload = _JSON_MAPPING.validate_python(envelope.model_dump(mode="json"))
    generation = _JSON_MAPPING.validate_python(payload["generation_parameters"])

    with pytest.raises(ValidationError):
        _ = _PROVIDER_ENVELOPE.validate_python(
            payload | {"future_prompt_field": "forbidden"}
        )
    with pytest.raises(ValidationError):
        _ = _PROVIDER_ENVELOPE.validate_python(
            payload
            | {"generation_parameters": generation | {"future_generation_field": 1}}
        )
    with pytest.raises(ValidationError):
        _ = _PROVIDER_ENVELOPE.validate_python(payload | {"kind": "future_transport"})
