from __future__ import annotations

import os
from collections.abc import Mapping
from importlib.metadata import version
from typing import Literal

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.hashing import (
    canonical_json_bytes,
    sha256_frame,
)
from market_support_crewai_agent.runtime.integrations.crewai import agent_factory
from market_support_crewai_agent.runtime.integrations.crewai.agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAICompletionValueV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.io import (
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    AgentExecutionSpecV1,
    PromptProgramV2,
    ProviderChatMessageV1,
    ProviderJsonSchemaFormatV1,
    resolve_active_prompt_program_v2,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    ChatGenerationParametersV1,
    CrewAIChatEnvelopeV1,
    ProviderResponseFormatV1,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.unit.orchestration._crewai_io_support import (
    canonical_value,
    planner_program,
    run_async,
)

_STRING = TypeAdapter(str)
_MESSAGES = TypeAdapter(list[ProviderChatMessageV1])
_FLOAT_OR_NONE: TypeAdapter[float | None] = TypeAdapter(float | None)
_INT_OR_NONE: TypeAdapter[int | None] = TypeAdapter(int | None)
_STOP: TypeAdapter[list[str] | None] = TypeAdapter(list[str] | None)
_TOOL_CHOICE: TypeAdapter[Literal["none", "auto", "required"] | None] = TypeAdapter(
    Literal["none", "auto", "required"] | None
)
_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def test_run_crewai_kickoff_hashes_exact_crewai_sdk_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CREWAI_MAX_RETRY_LIMIT", raising=False)
    agent = CrewAIAgentFactory(Settings(llm_api_key="test-key")).build_planner_agent()
    captured_params: dict[str, JsonValue] = {}
    response_models: list[type[BaseModel]] = []

    def capture_completion(
        *,
        params: Mapping[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> CrewAICompletionValueV1:
        del available_functions, from_task, from_agent
        captured_params.update(params)
        if response_model is None:
            raise AssertionError("response_model_missing")
        response_models.append(response_model)
        return make_weekly_plan_spec()

    assert agent.llm.completion_capture is not None
    agent.llm.completion_capture.replace(capture_completion)
    journal = TurnLlmInvocationJournalV1()

    _ = run_async(
        lambda: run_crewai_kickoff(
            agent,
            planner_program(),
            timeout_seconds=1,
            journal=journal,
        )
    )

    response_model = response_models[0]
    messages = _MESSAGES.validate_python(captured_params.pop("messages"))
    model = _STRING.validate_python(captured_params.pop("model"))
    canonical_schema = ProviderJsonSchemaFormatV1(
        name=response_model.__name__,
        json_schema=response_model.model_json_schema(),
    )
    provider_schema = {
        "name": canonical_schema.name,
        "schema": canonical_schema.json_schema,
        "strict": True,
    }
    provider_schema_value = canonical_value(
        _JSON_VALUE.validate_python(provider_schema)
    )
    expected_poh1 = sha256_frame(
        "provider-output-schema.v1",
        {
            "model_family": "ds_v4pro",
            "provider_id": "openai_compatible",
            "provider_output_schema": provider_schema_value,
            "schema_transform_version": "openai-response-format.v1",
        },
        prefix="poh1",
    )
    expected = CrewAIChatEnvelopeV1(
        model_family="ds_v4pro",
        model_name=model,
        messages=tuple(messages),
        response_format=ProviderResponseFormatV1(
            json_schema=canonical_schema,
            provider_output_schema_hash=expected_poh1,
        ),
        generation_parameters=ChatGenerationParametersV1(
            temperature=_FLOAT_OR_NONE.validate_python(
                captured_params.pop("temperature", None)
            ),
            max_tokens=_INT_OR_NONE.validate_python(
                captured_params.pop("max_tokens", None)
            ),
            top_p=_FLOAT_OR_NONE.validate_python(captured_params.pop("top_p", None)),
            stop=tuple(_STOP.validate_python(captured_params.pop("stop", None)) or ()),
            seed=_INT_OR_NONE.validate_python(captured_params.pop("seed", None)),
            tool_choice=_TOOL_CHOICE.validate_python(
                captured_params.pop("tool_choice", None)
            ),
        ),
        crewai_version=version("crewai"),
    )
    assert captured_params == {}
    assert journal.rows[0].prh1 == expected.prh1()
    assert os.getenv("CREWAI_MAX_RETRY_LIMIT") is None


@pytest.mark.parametrize(
    ("field_name", "mutated_value"),
    (
        ("task_template", "Mutated task template"),
        ("expected_output_template", "Mutated expected output"),
        ("agent_spec_version", 2),
    ),
)
def test_crewai_sdk_request_consumes_complete_packaged_execution_spec(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    mutated_value: str | int,
) -> None:
    program, execution_spec = resolve_active_prompt_program_v2(
        stage="planner_intent",
        scene_key="wecom_group.v1",
    )

    def dispatch(spec: AgentExecutionSpecV1) -> tuple[bytes, str | None]:
        def resolve_program(
            **_kwargs: str,
        ) -> tuple[PromptProgramV2, AgentExecutionSpecV1]:
            return program, spec

        monkeypatch.setattr(
            agent_factory,
            "resolve_active_prompt_program_v2",
            resolve_program,
        )
        agent = CrewAIAgentFactory(
            Settings(llm_api_key="test-key")
        ).build_planner_agent()
        captured_params: dict[str, JsonValue] = {}

        def capture_completion(
            *,
            params: Mapping[str, JsonValue],
            available_functions: JsonValue | None = None,
            from_task: JsonValue | None = None,
            from_agent: JsonValue | None = None,
            response_model: type[BaseModel] | None = None,
        ) -> CrewAICompletionValueV1:
            del available_functions, from_task, from_agent, response_model
            captured_params.update(params)
            return make_weekly_plan_spec()

        assert agent.llm.completion_capture is not None
        agent.llm.completion_capture.replace(capture_completion)
        journal = TurnLlmInvocationJournalV1()
        _ = run_async(
            lambda: run_crewai_kickoff(
                agent,
                planner_program(),
                timeout_seconds=1,
                journal=journal,
            )
        )
        messages_value = canonical_value(
            _JSON_VALUE.validate_python(captured_params["messages"])
        )
        return canonical_json_bytes(messages_value), journal.rows[0].prh1

    baseline_messages, baseline_prh1 = dispatch(execution_spec)
    mutated_messages, mutated_prh1 = dispatch(
        execution_spec.model_copy(update={field_name: mutated_value})
    )

    assert mutated_messages != baseline_messages
    assert mutated_prh1 != baseline_prh1
