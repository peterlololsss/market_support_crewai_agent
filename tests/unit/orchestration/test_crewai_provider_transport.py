from __future__ import annotations

import httpx
import pytest
from openai import OpenAI
from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.integrations.crewai import (
    request_capture,
    targeting,
)
from market_support_crewai_agent.runtime.integrations.crewai.agent_factory import (
    CrewAIAgentFactory,
)
from market_support_crewai_agent.runtime.integrations.crewai.io import (
    run_crewai_kickoff,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderChatMessageV1,
)
from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderInvocationError,
)
from market_support_crewai_agent.runtime.prompts.provider_message_envelopes import (
    CrewAIChatEnvelopeV1,
    ProviderMessageEnvelopeV1,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.unit.orchestration._crewai_io_support import planner_program, run_async

_JSON_MAPPING = TypeAdapter(dict[str, JsonValue])
_MESSAGES = TypeAdapter(list[ProviderChatMessageV1])


def test_factory_capture_records_only_real_crewai_completion_invocation() -> None:
    agent = CrewAIAgentFactory(
        Settings(llm_api_key="test-key", planner_llm_api_key="test-key")
    ).build_planner_agent()
    program = planner_program()
    envelope = targeting.crewai_transport_envelope(agent, program)
    request_bodies: list[dict[str, JsonValue]] = []

    def completion_response(request: httpx.Request) -> httpx.Response:
        request_bodies.append(_JSON_MAPPING.validate_json(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-capture",
                "object": "chat.completion",
                "created": 1,
                "model": agent.llm.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": make_weekly_plan_spec().model_dump_json(),
                            "refusal": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(completion_response)) as client:
        if agent.llm.configure_openai_client is None:
            raise AssertionError("openai_configurator_missing")
        agent.llm.configure_openai_client(
            OpenAI(
                api_key="test-key",
                base_url="https://provider.invalid/v1",
                max_retries=agent.llm.max_retries or 0,
                http_client=client,
            )
        )
        captures: list[ProviderMessageEnvelopeV1] = []
        if agent.llm.completion_capture is None:
            raise AssertionError("completion_capture_missing")
        original_completion = agent.llm.completion_capture.current()
        if original_completion is None:
            raise AssertionError("real_completion_capture_missing")
        restore = request_capture.install_crewai_request_capture(
            agent,
            program,
            envelope,
            captures,
        )
        try:
            assert captures == []
            result = run_async(
                lambda: agent.kickoff(
                    program.prompt_text,
                    response_format=program.profile.response_model,
                )
            )
        finally:
            restore()

    assert result.pydantic == make_weekly_plan_spec()
    assert len(request_bodies) == 1
    assert len(captures) == 1
    captured = captures[0]
    assert isinstance(captured, CrewAIChatEnvelopeV1)
    assert captured.model_name == request_bodies[0]["model"]
    assert captured.messages == tuple(
        _MESSAGES.validate_python(request_bodies[0]["messages"])
    )
    assert agent.llm.completion_capture.current() is original_completion


def test_one_journal_row_equals_one_openai_provider_http_dispatch() -> None:
    agent = CrewAIAgentFactory(
        Settings(llm_api_key="test-key", planner_llm_api_key="test-key")
    ).build_planner_agent()
    dispatches: list[httpx.Request] = []

    def retryable_response(request: httpx.Request) -> httpx.Response:
        dispatches.append(request)
        return httpx.Response(
            500,
            headers={"retry-after-ms": "0"},
            json={"error": {"message": "retryable", "type": "server_error"}},
            request=request,
        )

    journal = TurnLlmInvocationJournalV1()
    with httpx.Client(transport=httpx.MockTransport(retryable_response)) as client:
        if agent.llm.configure_openai_client is None:
            raise AssertionError("openai_configurator_missing")
        agent.llm.configure_openai_client(
            OpenAI(
                api_key="test-key",
                base_url="https://provider.invalid/v1",
                max_retries=agent.llm.max_retries or 0,
                http_client=client,
            )
        )

        with pytest.raises(ProviderInvocationError):
            _ = run_async(
                lambda: run_crewai_kickoff(
                    agent,
                    planner_program(),
                    timeout_seconds=5,
                    journal=journal,
                )
            )

    assert len(journal.rows) == 1
    assert len(dispatches) == 1
