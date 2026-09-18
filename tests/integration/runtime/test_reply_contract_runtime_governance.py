from __future__ import annotations

import asyncio
import os

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

import pytest
from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.runtime.identity import VerifiedRequestEnvelopeV1
from market_support_crewai_agent.runtime.rendering.response_ids import (
    ensure_response_ids,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.validation.request_input_guard import (
    InputGuardrailError,
)
from market_support_crewai_agent.schemas.reply import (
    PrimaryReply,
    ReplyResponse,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_completion_agent_adapter
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec
from tests.helpers.reply_contract_preflight import (
    EmptyPreflightService,
    ResolvedWeeklyPreflight,
)
from tests.helpers.reply_contract_requests import make_v2_payload
from tests.helpers.reply_contract_runtime import (
    ReplyRequestBuilder,
    client,
    failing_crewai_agent,
    install_composer_agent_builder,
    install_planner_agent_builder,
    invalid_weekly_plan_raw,
    reply_test_settings,
    sleeping_planner_agent,
)


@pytest.mark.filterwarnings(
    "ignore:function_calling_llm is deprecated.*:DeprecationWarning"
)
@pytest.mark.filterwarnings("ignore:deprecated:DeprecationWarning")
@pytest.mark.filterwarnings(
    "ignore:The 'reasoning' parameter is deprecated.*:DeprecationWarning"
)
def test_crewai_agents_disable_planning_delegation_and_retries():
    with pytest.raises(ValueError):
        _invalid_settings = Settings(crewai_max_retry_limit=4)

    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            llm_timeout_seconds=7,
            crewai_max_retry_limit=0,
            reply_alignment_verifier_enabled=False,
            planner_llm_base_url="http://planner.local/gemini",
            planner_llm_provider="gemini",
            planner_llm_model="gemini-3-flash-preview",
            planner_llm_api_key="planner-key",
        )
    )

    planner = runtime.planner_agent_factory.build_planner_agent()
    composer = runtime.composer_agent_factory.build_composer_agent("knowledge_composer")

    assert planner.planning is False
    assert planner.allow_delegation is False
    assert planner.inject_date is True
    assert planner.llm.model == "gemini-3-flash-preview"
    assert planner.llm.provider == "gemini"
    http_options = planner.llm.client_params["http_options"]
    assert isinstance(http_options, dict)
    assert http_options["base_url"] == ("http://planner.local/gemini")
    assert http_options["timeout"] == 7000
    assert planner.llm.api_key == "planner-key"
    assert planner.llm.max_output_tokens == 6000
    assert planner.max_retry_limit == 0
    assert composer.planning is False
    assert composer.allow_delegation is False
    assert composer.llm.model == "deepseek-v4-pro"
    assert composer.llm.timeout == 7
    assert composer.max_retry_limit == 0


pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "market-support-crewai-agent",
    }


def test_ensure_response_ids_strips_markdown_markers_for_plain_text():
    response = ensure_response_ids(
        ReplyResponse(
            response_id="resp-plain",
            reply=PrimaryReply(kind="answer", text="**量价因子**：80%-90%"),
            actions=[],
        )
    )

    assert response.reply.text == "量价因子：80%-90%"


def test_reply_returns_runtime_response_without_business_rewrite(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = ReplyResponse(
        response_id="resp-test",
        reply=PrimaryReply(kind="answer", text="runtime decided text"),
        actions=[
            SendWeeklyReportAction(
                type="send_weekly_report",
                action_id="act-1",
                resolve_type="weekly_report",
                resolve_ref="weekly:ref",
                period="20260529",
                report_date="2026-05-29",
            )
        ],
    )

    async def fake_build_reply(request: VerifiedRequestEnvelopeV1):
        assert request.request.identity.tenant_ref == "tenant:test"
        return expected

    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply", fake_build_reply
    )

    response = client.post(
        "/reply", json=make_v2_payload("any message"), headers={"X-API-Key": "secret"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "contract_version": "reply",
        "response_id": "resp-test",
        "reply": {
            "kind": "answer",
            "text": "runtime decided text",
            "mentions": [],
        },
        "actions": [
            {
                "action_id": "act-1",
                "type": "send_weekly_report",
                "resolve_type": "weekly_report",
                "resolve_ref": "weekly:ref",
                "period": "20260529",
                "report_date": "2026-05-29",
            }
        ],
    }


def test_reply_route_rejects_message_over_configured_input_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AGENT_INPUT_MAX_MESSAGE_CHARS", "5")

    response = client.post("/reply", json=make_v2_payload("abcdef"))

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_body_too_large"


def test_runtime_input_guardrail_runs_before_llm_configuration():
    runtime = CrewAIReplyRuntime(
        Settings(agent_input_max_message_chars=5),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )

    try:
        _ = asyncio.run(runtime.reply(ReplyRequestBuilder("abcdef").payload()))
    except InputGuardrailError as exc:
        error = exc
    else:
        raise AssertionError("input guardrail should reject oversized message")

    assert error.code == "message_too_long"


def test_runtime_times_out_slow_crewai_planner_before_composer_runs():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            llm_timeout_seconds=0.01,
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )

    install_planner_agent_builder(
        runtime,
        lambda: sleeping_planner_agent(1),
    )
    install_composer_agent_builder(
        runtime,
        lambda _stage: failing_crewai_agent(
            "composer",
            "composer should not run after planner timeout",
        ),
    )

    try:
        _ = asyncio.run(runtime.reply(ReplyRequestBuilder("周报请求").payload()))
    except AgentRuntimeError as exc:
        error = exc
    else:
        raise AssertionError("slow planner should time out")

    assert str(error) == "CrewAI planner timed out"


def test_runtime_retries_invalid_planner_contract_with_feedback():
    runtime = CrewAIReplyRuntime(
        reply_test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )

    prompts: list[str] = []

    def completion(_response_format: type[BaseModel]) -> BaseModel | JsonValue:
        if len(prompts) == 1:
            return invalid_weekly_plan_raw()
        return make_weekly_plan_spec()

    planner = make_completion_agent_adapter(
        completion,
        role="planner",
        model="fake-planner",
        on_prompt=prompts.append,
    )
    install_planner_agent_builder(runtime, lambda: planner)

    response = asyncio.run(runtime.reply(ReplyRequestBuilder("周报请求").payload()))

    assert response.actions[0].type == "send_weekly_report"
    assert len(prompts) == 2
