from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.state.audit import AuditStore
from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.runtime.validation.reply_validator import ReplyContractError
from market_support_crewai_agent.runtime.validation.request_input_guard import InputGuardrailError
from market_support_crewai_agent.runtime.orchestration.reply_agent import (
    AgentRuntimeError,
    CrewAIReplyRuntime,
    build_reply,
)
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.runtime.orchestration.response_ids import ensure_response_ids
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    PrimaryReply,
    ReplyRequest,
    ReplyResponse,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.server.main import app
from market_support_crewai_agent.settings import Settings
from tests.helpers.reply_contract import (
    CapturingEmptyPreflightService,
    CapturingResolvedMaterialPreflight,
    CapturingResolvedWeeklyPreflight,
    EmptyPreflightService,
    FakePlannerAgent,
    MissingWeeklyWithSalesPreflight,
    ResolvedMonthlyPreflight,
    ResolvedWeeklyMonthlyPreflight,
    ResolvedWeeklyPreflight,
    _assistant_history_with_pending,
    install_fake_planner,
    make_support_plan_spec,
    make_monthly_plan_spec,
    make_payload,
    make_weekly_plan_spec,
    resolved_item,
)

client = TestClient(app)


def install_fake_clarification_composer(
    runtime: CrewAIReplyRuntime,
    *,
    text: str,
    prompts: list[str] | None = None,
    stages: list[str] | None = None,
):
    class FakeComposer:
        async def kickoff_async(self, prompt, response_format):
            if prompts is not None:
                prompts.append(prompt)
            assert response_format is ComposerReplyOutput
            return SimpleNamespace(
                pydantic=ComposerReplyOutput(
                    response_mode="clarify",
                    missing_inputs=["ambiguity"],
                    reply=PrimaryReply(kind="clarification", text=text, mentions=[]),
                    actions=[],
                ),
                raw="",
            )

    def build_agent(stage="knowledge_composer"):
        if stages is not None:
            stages.append(stage)
        return FakeComposer()

    runtime._build_agent = build_agent  # type: ignore[method-assign]


@pytest.mark.filterwarnings(
    "ignore:function_calling_llm is deprecated.*:DeprecationWarning"
)
@pytest.mark.filterwarnings("ignore:deprecated:DeprecationWarning")
@pytest.mark.filterwarnings(
    "ignore:The 'reasoning' parameter is deprecated.*:DeprecationWarning"
)
def test_crewai_agents_disable_execution_planning_without_delegation():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            llm_timeout_seconds=7,
            crewai_max_retry_limit=4,
            reply_alignment_verifier_enabled=False,
            planner_llm_base_url="http://planner.local/gemini",
            planner_llm_provider="gemini",
            planner_llm_model="gemini-3-flash-preview",
            planner_llm_api_key="planner-key",
        )
    )

    planner = runtime._build_planner_agent()
    composer = runtime._build_agent()

    assert planner.planning is False
    assert planner.allow_delegation is False
    assert planner.inject_date is True
    assert planner.llm.model == "gemini-3-flash-preview"
    assert planner.llm.provider == "gemini"
    assert planner.llm.client_params["http_options"]["base_url"] == (
        "http://planner.local/gemini"
    )
    assert planner.llm.client_params["http_options"]["timeout"] == 7000
    assert planner.llm.api_key == "planner-key"
    assert planner.llm.max_output_tokens == 6000
    assert planner.max_retry_limit == 4
    assert composer.planning is False
    assert composer.allow_delegation is False
    assert composer.llm.model == "deepseek-v4-pro"
    assert composer.llm.timeout == 7
    assert composer.max_retry_limit == 4


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


def test_reply_returns_runtime_response_without_business_rewrite(monkeypatch):
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

    async def fake_build_reply(request):
        return expected

    monkeypatch.setattr("market_support_crewai_agent.server.main.build_reply", fake_build_reply)

    response = client.post("/reply", json=make_payload("any message"))

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


def test_reply_route_rejects_message_over_configured_input_limit(monkeypatch):
    monkeypatch.setenv("AGENT_INPUT_MAX_MESSAGE_CHARS", "5")

    response = client.post("/reply", json=make_payload("abcdef"))

    assert response.status_code == 413
    assert "message exceeds configured input guardrail limit" in response.json()["detail"]


def test_runtime_input_guardrail_runs_before_llm_configuration():
    runtime = CrewAIReplyRuntime(
        Settings(agent_input_max_message_chars=5),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("abcdef").payload()))
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

    class SlowPlannerAgent:
        async def kickoff_async(self, prompt, response_format):
            await asyncio.sleep(1)
            return SimpleNamespace(pydantic=make_support_plan_spec(), raw="")

    class ComposerShouldNotRun:
        async def kickoff_async(self, prompt, response_format):
            raise AssertionError("composer should not run after planner timeout")

    runtime._build_planner_agent = lambda: SlowPlannerAgent()  # type: ignore[method-assign]
    runtime._build_agent = lambda *_args, **_kwargs: ComposerShouldNotRun()  # type: ignore[method-assign]

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))
    except AgentRuntimeError as exc:
        error = exc
    else:
        raise AssertionError("slow planner should time out")

    assert str(error) == "CrewAI planner timed out"


def test_runtime_retries_invalid_planner_contract_with_feedback():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )

    class RetryPlannerAgent:
        def __init__(self):
            self.prompts: list[str] = []

        async def kickoff_async(self, prompt, response_format):
            del response_format
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return SimpleNamespace(pydantic=None, raw=_invalid_weekly_plan_raw())
            return SimpleNamespace(pydantic=make_weekly_plan_spec(), raw="")

    planner = RetryPlannerAgent()
    runtime._build_planner_agent = lambda: planner  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.actions[0].type == "send_weekly_report"
    assert len(planner.prompts) == 2
    assert "Previous PlanSpec validation error" in planner.prompts[1]
    assert "plan_units.0" in planner.prompts[1]
    assert "selected_capability_id" in planner.prompts[1]


def test_runtime_retries_empty_planner_output_before_contract_feedback():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            reply_alignment_verifier_enabled=False,
            planner_transient_retry_attempts=2,
            planner_transient_retry_base_seconds=0,
        ),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )

    class EmptyThenValidPlanner:
        def __init__(self):
            self.prompts: list[str] = []

        async def kickoff_async(self, prompt, response_format):
            del response_format
            self.prompts.append(prompt)
            if len(self.prompts) < 3:
                return SimpleNamespace(pydantic=None, raw="")
            return SimpleNamespace(pydantic=make_weekly_plan_spec(), raw="")

    planner = EmptyThenValidPlanner()
    runtime._build_planner_agent = lambda: planner  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.actions[0].type == "send_weekly_report"
    assert len(planner.prompts) == 3
    assert len(set(planner.prompts)) == 1


def test_runtime_accepts_missing_mechanical_evidence_contract_ref():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )

    class MissingRefPlannerAgent:
        def __init__(self):
            self.prompts: list[str] = []

        async def kickoff_async(self, prompt, response_format):
            del response_format
            self.prompts.append(prompt)
            return SimpleNamespace(pydantic=None, raw=_missing_ref_weekly_plan_raw())

    planner = MissingRefPlannerAgent()
    runtime._build_planner_agent = lambda: planner  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.actions[0].type == "send_weekly_report"
    assert len(planner.prompts) == 1


def test_runtime_invalid_planner_contract_error_includes_field_feedback():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )

    class InvalidPlannerAgent:
        def __init__(self):
            self.prompts: list[str] = []

        async def kickoff_async(self, prompt, response_format):
            del response_format
            self.prompts.append(prompt)
            return SimpleNamespace(pydantic=None, raw=_invalid_weekly_plan_raw())

    planner = InvalidPlannerAgent()
    runtime._build_planner_agent = lambda: planner  # type: ignore[method-assign]

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))
    except AgentRuntimeError as exc:
        error = exc
    else:
        raise AssertionError("invalid planner output should raise")

    assert len(planner.prompts) == 2
    assert "Previous PlanSpec validation error" in planner.prompts[1]
    assert "plan_units.0" in str(error)
    assert "selected_capability_id" in str(error)


def test_reply_requires_api_key_when_configured(monkeypatch):
    called = False

    async def fake_build_reply(request):
        nonlocal called
        called = True
        return ReplyResponse(
            response_id="resp-test",
            reply=PrimaryReply(kind="answer", text="ok"),
            actions=[],
        )

    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")
    monkeypatch.setattr("market_support_crewai_agent.server.main.build_reply", fake_build_reply)

    response = client.post("/reply", json=make_payload("any message"))

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}
    assert called is False


def test_reply_accepts_bearer_api_key_when_configured(monkeypatch):
    expected = ReplyResponse(
        response_id="resp-test",
        reply=PrimaryReply(kind="answer", text="ok"),
        actions=[],
    )

    async def fake_build_reply(request):
        return expected

    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")
    monkeypatch.setattr("market_support_crewai_agent.server.main.build_reply", fake_build_reply)

    response = client.post(
        "/reply",
        json=make_payload("any message"),
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json()["response_id"] == "resp-test"


def test_reply_returns_502_when_runtime_fails(monkeypatch):
    async def fake_build_reply(request):
        raise AgentRuntimeError("runtime failed")

    monkeypatch.setattr("market_support_crewai_agent.server.main.build_reply", fake_build_reply)

    response = client.post("/reply", json=make_payload("any message"))

    assert response.status_code == 502
    assert response.json() == {"detail": "runtime failed"}


def test_reply_returns_502_when_contract_validation_fails(monkeypatch):
    async def fake_build_reply(request):
        raise ReplyContractError("invalid reply")

    monkeypatch.setattr("market_support_crewai_agent.server.main.build_reply", fake_build_reply)

    response = client.post("/reply", json=make_payload("any message"))

    assert response.status_code == 502
    assert response.json() == {"detail": "invalid reply"}


def test_request_contract_rejects_unknown_available_artifact_type():
    response = client.post(
        "/reply",
        json=make_payload(available_artifacts=[{"type": "calendar"}]),
    )

    assert response.status_code == 422


def test_request_contract_rejects_report_artifact_options():
    response = client.post(
        "/reply",
        json=make_payload(
            available_artifacts=[{"type": "weekly_report", "options": ["x"]}]
        ),
    )

    assert response.status_code == 422


def test_request_contract_rejects_removed_trigger_and_session_fields():
    for removed_field in ("session_id", "bot_mentioned", "trigger_reason"):
        response = client.post(
            "/reply",
            json=make_payload(**{removed_field: "not allowed"}),
        )

        assert response.status_code == 422


def test_reply_response_rejects_unsupported_contract_version():
    from pydantic import ValidationError

    try:
        ReplyResponse.model_validate(
            {
                "contract_version": "reply-versioned",
                "response_id": "resp-unsupported",
                "reply": {"kind": "answer", "text": "ok", "mentions": []},
                "actions": [],
            }
        )
    except ValidationError:
        return

    raise AssertionError("version-suffixed reply contract must not be accepted")


def test_report_action_requires_period_and_report_date():
    from pydantic import ValidationError

    try:
        SendWeeklyReportAction.model_validate(
            {
                "type": "send_weekly_report",
                "action_id": "act-1",
                "resolve_type": "weekly_report",
                "resolve_ref": "weekly:ref",
            }
        )
    except ValidationError:
        return

    raise AssertionError("report send actions must include period and report_date")


def test_report_action_rejects_report_scope_selector():
    from pydantic import ValidationError

    try:
        SendWeeklyReportAction.model_validate(
            {
                "type": "send_weekly_report",
                "action_id": "act-1",
                "resolve_type": "weekly_report",
                "resolve_ref": "weekly:ref",
                "report_scope": "channel_all",
                "period": "20260529",
                "report_date": "2026-05-29",
            }
        )
    except ValidationError:
        return

    raise AssertionError("report send actions must not include report_scope")


def test_build_reply_uses_custom_settings_for_default_runtime_services(monkeypatch):
    settings = Settings(
        llm_api_key="test-key",
        doc_mcp_enabled=True,
        doc_mcp_base_url="http://doc-mcp.local:23000",
        agent_conversation_max_messages=3,
        reply_alignment_verifier_enabled=False,
    )
    seen: dict[str, object] = {}

    async def fake_reply(self, request):
        seen["settings"] = self.settings
        seen["document_settings"] = (
            self.evidence_executor.document_evidence_service.settings
        )
        seen["conversation_max_messages"] = self.conversation_store._max_messages
        return ReplyResponse(
            response_id="resp-ok",
            reply=PrimaryReply(kind="answer", text="ok"),
            actions=[],
        )

    monkeypatch.setattr(CrewAIReplyRuntime, "reply", fake_reply)

    response = asyncio.run(
        build_reply(ReplyRequestShim("介绍一下中证1000").payload(), settings=settings)
    )

    assert response.response_id == "resp-ok"
    assert seen["settings"] is settings
    assert seen["document_settings"] is settings
    assert seen["conversation_max_messages"] == 3


def test_runtime_deterministic_action_does_not_call_composer():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    class ComposerShouldNotRun:
        async def kickoff_async(self, prompt, response_format):
            raise AssertionError("composer should not run for action responses")

    runtime._build_agent = lambda *_args, **_kwargs: ComposerShouldNotRun()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_weekly_report"
    assert response.actions[0].resolve_ref == "weekly:ref"
    assert response.actions[0].period == "20260529"


def test_runtime_clarifies_concrete_artifact_choice():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="unclear artifact choice",
            artifact_kind="unclear",
            action_intent="none",
            ambiguity_slots=["artifact"],
            requested_capabilities=[],
        ),
    )
    composer_prompts: list[str] = []
    composer_stages: list[str] = []
    install_fake_clarification_composer(
        runtime,
        text="???????????",
        prompts=composer_prompts,
        stages=composer_stages,
    )

    response = asyncio.run(runtime.reply(ReplyRequestShim("????").payload()))

    assert response.reply.kind == "clarification"
    assert response.reply.text == "???????????"
    assert response.actions == []
    assert composer_stages == ["knowledge_composer"]
    assert "artifact" in composer_prompts[0]

def test_runtime_allows_mixed_question_plus_unqualified_monthly_send():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedMonthlyPreflight(),
    )
    install_fake_planner(runtime, make_monthly_plan_spec())

    response = asyncio.run(
        runtime.reply(
            ReplyRequest.model_validate(
                make_payload(
                    "在各个策略上的规模是怎么分布呢  然后发我个月报",
                    dist_channel_name="浦发银行",
                    available_artifacts=[{"type": "material_pack", "options": ["中证1000指增", "中证A500指增", "中证全指指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
                )
            )
        )
    )

    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_monthly_report"
    assert response.actions[0].resolve_ref == "monthly:ref"


def test_runtime_retries_report_query_clarification_as_invalid_plan():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    prompts: list[str] = []

    class RetryPlannerAgent:
        async def kickoff_async(self, prompt, response_format):
            del response_format
            prompts.append(prompt)
            if len(prompts) == 1:
                return SimpleNamespace(
                    pydantic=make_weekly_plan_spec(ambiguity_slots=["report_query"]),
                    raw="",
                )
            return SimpleNamespace(pydantic=make_weekly_plan_spec(), raw="")

    runtime._build_planner_agent = lambda: RetryPlannerAgent()  # type: ignore[method-assign]

    response = asyncio.run(
        runtime.reply(
            ReplyRequest.model_validate(
                make_payload(
                    "[adapter_allowed_read_capabilities: query_internal_company_info]\nweekly report",
                    available_artifacts=[{"type": "material_pack", "options": ["option-a", "option-b", "option-c"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
                )
            )
        )
    )

    assert response.reply.kind == "answer"
    assert response.actions[0].type == "send_weekly_report"
    assert len(prompts) == 2
    assert "clarification_missing_supported_slot" in prompts[1]

def test_runtime_uses_planner_resolved_followup_for_weekly_action():
    store = ConversationStore(max_messages=12)
    store.save_turn(
        "wecom:group-1:sender-1",
        "[adapter_allowed_read_capabilities: query_internal_company_info]\n周报",
        ReplyResponse(
            response_id="resp-old",
            reply=PrimaryReply(kind="clarification", text="我需要再确认一下具体策略。"),
            actions=[],
        ).model_dump_json(exclude_none=True),
    )
    preflight = CapturingResolvedWeeklyPreflight()
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=preflight,
    )
    install_fake_planner(
        runtime,
        make_weekly_plan_spec(
            user_need="send prior weekly report request after clarification",
            requested_capabilities=["weekly_report"],
            ambiguity_slots=[],
        ),
    )

    response = asyncio.run(
        runtime.reply(
            ReplyRequest.model_validate(
                make_payload(
                    "中证1000的",
                    available_artifacts=[{"type": "material_pack", "options": ["中证1000指增", "中证A500指增", "中证全指指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
                )
            )
        )
    )

    assert preflight.resolve_material_pack_options == {}
    assert response.reply.kind == "answer"
    assert response.actions[0].type == "send_weekly_report"


def test_planner_prompt_includes_pending_clarification_context():
    store = ConversationStore(max_messages=12)
    store.save_turn(
        "wecom:group-1:sender-1",
        "[adapter_allowed_read_capabilities: query_internal_company_info]\n周报",
        _assistant_history_with_pending(
            text="我需要再确认一下具体策略。",
            pending_plan={
                "artifact_kind": "weekly_report",
                "response_mode": "clarification",
                "ambiguity_slots": ["artifact"],
                "capabilities": ["weekly_report"],
            },
        ),
    )
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=EmptyPreflightService(),
    )
    planner_prompts: list[str] = []
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_weekly_plan_spec(user_need="send clarified weekly report"),
        planner_prompts,
    )

    asyncio.run(
        runtime.reply(
            ReplyRequest.model_validate(
                make_payload(
                    "中证1000的",
                    available_artifacts=[{"type": "material_pack", "options": ["中证1000指增", "中证A500指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
                )
            )
        )
    )

    assert "Pending clarification context JSON" in planner_prompts[0]
    assert '"status": "awaiting_user_answer"' in planner_prompts[0]
    assert '"ambiguity_slots": [' in planner_prompts[0]
    assert "do not ask the same clarification again" in planner_prompts[0]


def test_ambiguous_action_candidates_are_structured_for_composer():
    class AmbiguousMaterialPreflight:
        async def collect(
            self,
            request,
            canonical_context=None,
            resolve_types=None,
            resolve_material_pack_options=None,
        ):
            del request, canonical_context, resolve_types, resolve_material_pack_options
            return AdapterPreflightSnapshot(
                items=[
                    resolved_item(
                        "material_pack",
                        status="ambiguous",
                        candidates=["中证1000指增", "中证A500指增"],
                    )
                ]
            )

    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=AmbiguousMaterialPreflight(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            artifact_kind="material_pack",
            action_intent="send",
            ambiguity_slots=[],
        ),
    )
    composer_prompts: list[str] = []
    install_fake_clarification_composer(
        runtime,
        text="请问你要发哪一个材料？",
        prompts=composer_prompts,
    )

    response = asyncio.run(
        runtime.reply(
            ReplyRequest.model_validate(
                make_payload(
                    "发一下材料",
                    available_artifacts=[{"type": "material_pack", "options": ["中证1000指增", "中证A500指增"]}, {"type": "weekly_report"}, {"type": "monthly_report"}],
                )
            )
        )
    )

    assert response.reply.kind == "clarification"
    assert "ambiguous_action_resolve" in composer_prompts[0]
    assert "中证1000指增" in composer_prompts[0]
    assert "我需要再确认一下你指的是哪一个可发送内容" not in composer_prompts[0]


def test_runtime_records_audit_before_raising_reply_validation_error(monkeypatch):
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    def bad_render_directive(directive, plan, business_facts, evidence_facts):
        del directive, plan, business_facts, evidence_facts
        return ReplyResponse(
            response_id="resp-bad",
            reply=PrimaryReply(kind="answer", text="周报已发送，请查收。"),
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

    monkeypatch.setattr(
        "market_support_crewai_agent.runtime.orchestration.reply_agent.render_directive",
        bad_render_directive,
    )

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))
    except ReplyContractError as exc:
        error = exc
    else:
        raise AssertionError("invalid rendered reply must raise")

    assert "rendered reply failed validation" in str(error)
    trace = audit_store.latest()
    assert trace is not None
    assert trace.reply_validation is not None
    assert trace.reply_validation["valid"] is False


def test_runtime_hands_off_missing_action_when_sales_resolves():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=MissingWeeklyWithSalesPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.reply.kind == "human_handoff"
    assert response.reply.mentions[0].type == "sales"
    assert response.actions == []


def test_runtime_uses_composer_only_for_knowledge_answer():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            ambiguity_slots=[],
        ),
    )
    prompts: list[str] = []

    class FakeComposer:
        async def kickoff_async(self, prompt, response_format):
            prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-ok",
                    reply=PrimaryReply(kind="answer", text="文档证据回答"),
                    actions=[],
                ),
                raw="",
            )

    class FakeEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
            from market_support_crewai_agent.runtime.evidence import EvidenceFact

            facts = [
                EvidenceFact(
                    fact_type="document_context",
                    value="文档证据",
                    source_type="document_mcp",
                    source_id="doc-1",
                ),
                EvidenceFact(
                    fact_type="report_scope_products",
                    value=True,
                    source_type="adapter_report_scope",
                    source_id="weekly_report",
                    resolve_type="weekly_report",
                    metadata={"products": [{"product_name": "Weekly Product"}]},
                    artifact_type="weekly_report",
                ),
            ]
            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=facts,
                business_facts=derive_business_facts(facts, request),
            )

    runtime.evidence_executor = FakeEvidenceExecutor()
    runtime._build_agent = lambda *_args, **_kwargs: FakeComposer()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("介绍一下衍复").payload()))

    assert response.reply.text == "文档证据回答"
    assert prompts
    assert "ExecutionPlan JSON" in prompts[0]
    assert "model-visible-context" in prompts[0]
    assert '"title": "Allowed evidence JSON"' in prompts[0]
    assert '"answerability_assessment": {' in prompts[0]
    assert "Weekly Product" not in prompts[0]


def test_alignment_verifier_blocks_wrong_side_effect_action():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
        ),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    class WrongActionVerifier:
        async def verify(self, **kwargs):
            assert kwargs["response"].actions[0].type == "send_weekly_report"
            return ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="wrong_artifact",
                remediation="return_unable",
                rationale="question asks why a monthly report omits a field",
            )

    runtime.alignment_verifier = WrongActionVerifier()

    response = asyncio.run(runtime.reply(ReplyRequestShim("月报里为什么没有年化收益率").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert response.actions == []


def test_alignment_verifier_allows_valid_action_response():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    class PassingVerifier:
        async def verify(self, **kwargs):
            assert kwargs["response"].actions[0].type == "send_weekly_report"
            return ReplyAlignmentVerdict(
                aligned=True,
                safe_to_return=True,
                confidence=0.9,
            )

    runtime.alignment_verifier = PassingVerifier()

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.reply.text == ""
    assert response.actions[0].type == "send_weekly_report"


def test_alignment_verifier_return_clarification_uses_composer():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())
    composer_stages: list[str] = []
    install_fake_clarification_composer(
        runtime,
        text="当前不确定你要周报还是月报，请确认一下。",
        stages=composer_stages,
    )

    class ClarifyingVerifier:
        async def verify(self, **kwargs):
            assert kwargs["response"].actions[0].type == "send_weekly_report"
            return ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="ambiguous_request",
                remediation="return_clarification",
                rationale="report artifact is ambiguous",
            )

    runtime.alignment_verifier = ClarifyingVerifier()

    response = asyncio.run(runtime.reply(ReplyRequestShim("报告发我一下").payload()))

    assert response.reply.kind == "clarification"
    assert response.reply.text == "当前不确定你要周报还是月报，请确认一下。"
    assert response.actions == []
    assert composer_stages == ["knowledge_composer"]


def test_alignment_verifier_replan_path_includes_feedback():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
        ),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    planner_prompts: list[str] = []
    frames = [
        make_weekly_plan_spec(),
        make_support_plan_spec(
            user_need="answer report format question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            evidence_query="月报 年化收益率 展示规则",
            ambiguity_slots=[],
        ),
    ]

    class SequentialPlanner:
        async def kickoff_async(self, prompt, response_format):
            del response_format
            planner_prompts.append(prompt)
            return SimpleNamespace(pydantic=frames.pop(0), raw="")

    class FakeEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
            from market_support_crewai_agent.runtime.evidence import (
                EvidenceFact,
                evidence_facts_from_preflight,
            )

            if plan.response_mode == "knowledge_answer":
                facts = [
                    EvidenceFact(
                        fact_type="document_context",
                        value="月报不展示年化收益率。",
                        source_type="document_mcp",
                        source_id="doc-1",
                    )
                ]
                preflight = AdapterPreflightSnapshot.empty()
            else:
                preflight = AdapterPreflightSnapshot(
                    items=[
                        resolved_item(
                            "weekly_report",
                            resolve_ref="weekly:ref",
                            period="20260529",
                            report_date="2026-05-29",
                        )
                    ]
                )
                facts = evidence_facts_from_preflight(preflight)
            return SimpleNamespace(
                preflight=preflight,
                evidence_facts=facts,
                business_facts=derive_business_facts(facts, request),
            )

    class FakeComposer:
        async def kickoff_async(self, prompt, response_format):
            del prompt, response_format
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    reply=PrimaryReply(kind="answer", text="月报不展示年化收益率。"),
                    actions=[],
                ),
                raw="",
            )

    class ReplanThenPassVerifier:
        def __init__(self):
            self.calls = 0

        async def verify(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert kwargs["response"].actions[0].type == "send_weekly_report"
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="wrong_artifact",
                    remediation="replan",
                    planner_feedback="This is a report-format knowledge question.",
                )
            return ReplyAlignmentVerdict(aligned=True, safe_to_return=True, confidence=0.9)

    verifier = ReplanThenPassVerifier()
    runtime._build_planner_agent = lambda: SequentialPlanner()  # type: ignore[method-assign]
    runtime.evidence_executor = FakeEvidenceExecutor()
    runtime._build_agent = lambda *_args, **_kwargs: FakeComposer()  # type: ignore[method-assign]
    runtime.alignment_verifier = verifier

    response = asyncio.run(runtime.reply(ReplyRequestShim("月报里为什么没有年化收益率").payload()))

    assert response.actions == []
    assert response.reply.text == "月报不展示年化收益率。"
    assert len(planner_prompts) == 2
    assert "Previous alignment verdict JSON" in planner_prompts[1]


def test_alignment_replan_failure_returns_unable_instead_of_raising():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    planner_agents = [
        FakePlannerAgent(make_weekly_plan_spec()),
        SimpleNamespace(
            kickoff_async=lambda _prompt, _response_format: asyncio.sleep(
                0,
                result=SimpleNamespace(pydantic=None, raw=""),
            )
        ),
    ]

    class ReplanVerifier:
        async def verify(self, **kwargs):
            assert kwargs["response"].actions[0].type == "send_weekly_report"
            return ReplyAlignmentVerdict(
                aligned=False,
                safe_to_return=False,
                failure_code="wrong_intent",
                remediation="replan",
            )

    runtime._build_planner_agent = lambda: planner_agents.pop(0)  # type: ignore[method-assign]
    runtime.alignment_verifier = ReplanVerifier()

    response = asyncio.run(runtime.reply(ReplyRequestShim("报告发我一下").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert response.actions == []


def test_alignment_verifier_refetches_document_context_with_refined_query():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="answer report format question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            evidence_query="年化收益率",
            ambiguity_slots=[],
        ),
    )
    queries: list[str | None] = []

    class RefetchEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
            from market_support_crewai_agent.runtime.evidence import EvidenceFact

            del canonical_context, policy, action_history
            queries.append(plan.evidence_query)
            facts = []
            if plan.evidence_query == "月报 年化收益率 展示规则":
                facts = [
                    EvidenceFact(
                        fact_type="document_context",
                        value="月报采用区间收益展示，不展示年化收益率。",
                        source_type="document_mcp",
                        source_id="doc-1",
                    )
                ]
            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=facts,
                business_facts=derive_business_facts(facts, request),
            )

    class FakeComposer:
        async def kickoff_async(self, prompt, response_format):
            del prompt, response_format
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    reply=PrimaryReply(
                        kind="answer",
                        text="月报采用区间收益展示，不展示年化收益率。",
                    ),
                    actions=[],
                ),
                raw="",
            )

    class RefetchThenPassVerifier:
        def __init__(self):
            self.calls = 0

        async def verify(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert kwargs["response"].reply.kind == "unable_to_answer"
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="missing_evidence",
                    remediation="refetch_document_context",
                    refined_evidence_query="月报 年化收益率 展示规则",
                )
            return ReplyAlignmentVerdict(aligned=True, safe_to_return=True, confidence=0.9)

    verifier = RefetchThenPassVerifier()
    runtime.evidence_executor = RefetchEvidenceExecutor()
    runtime._build_agent = lambda *_args, **_kwargs: FakeComposer()  # type: ignore[method-assign]
    runtime.alignment_verifier = verifier

    response = asyncio.run(runtime.reply(ReplyRequestShim("月报里为什么没有年化收益率").payload()))

    assert queries == ["年化收益率", "月报 年化收益率 展示规则"]
    assert response.reply.text == "月报采用区间收益展示，不展示年化收益率。"
    assert response.actions == []


def test_alignment_verifier_refetches_report_scope_products_from_typed_refetch():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="answer weekly report product list",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["weekly_report"],
            evidence_query=None,
            ambiguity_slots=[],
        ),
    )
    queries: list[str | None] = []

    class ReportScopeEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
            from market_support_crewai_agent.runtime.evidence import EvidenceFact

            del canonical_context, policy, action_history
            queries.append(plan.evidence_query)
            facts = [
                EvidenceFact(
                    fact_type="weekly_report_resolvable",
                    value=True,
                    source_type="adapter_resolve",
                    source_id="weekly_report",
                    resolve_type="weekly_report",
                    metadata={
                        "status": "resolved",
                        "resolve_ref": "weekly:ref",
                        "period": "20260605",
                        "report_date": "2026-06-05",
                        "period_start": "2026-06-02",
                        "period_end": "2026-06-05",
                    },
                ),
                EvidenceFact(
                    fact_type="report_period",
                    value="20260605",
                    source_type="adapter_resolve",
                    source_id="weekly_report",
                    resolve_type="weekly_report",
                    metadata={
                        "period": "20260605",
                        "report_date": "2026-06-05",
                        "period_start": "2026-06-02",
                        "period_end": "2026-06-05",
                    },
                ),
            ]
            if plan.evidence_query == "report_scope_products":
                facts.append(
                    EvidenceFact(
                        fact_type="report_scope_products",
                        value=True,
                        source_type="adapter_report_scope",
                        source_id="weekly_report",
                        resolve_type="weekly_report",
                        metadata={
                            "period": "20260605",
                            "products": [
                                {
                                    "product_name": "Product1",
                                    "report_section": "IndexPlus",
                                    "source_pdf_status": "found",
                                    "final_report_status": "generated",
                                },
                                {
                                    "product_name": "Product2",
                                    "report_section": "IndexPlus",
                                    "source_pdf_status": "found",
                                    "final_report_status": "generated",
                                },
                            ],
                            "product_total_count": 2,
                        },
                    )
                )
            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=facts,
                business_facts=derive_business_facts(facts, request),
            )

    class FakeComposer:
        async def kickoff_async(self, prompt, response_format):
            del response_format
            text = (
                "weekly report products: Product1, Product2"
                if "Product1" in prompt
                else "weekly report product scope unavailable"
            )
            return SimpleNamespace(
                pydantic=ReplyResponse(reply=PrimaryReply(kind="answer", text=text), actions=[]),
                raw="",
            )

    class TypedRefetchThenPassVerifier:
        def __init__(self):
            self.calls = 0

        async def verify(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="missing_answer",
                    remediation="refetch_report_scope",
                    refined_evidence_query="report_scope_products",
                )
            return ReplyAlignmentVerdict(aligned=True, safe_to_return=True, confidence=0.9)

    verifier = TypedRefetchThenPassVerifier()
    runtime.evidence_executor = ReportScopeEvidenceExecutor()
    runtime._build_agent = lambda *_args, **_kwargs: FakeComposer()  # type: ignore[method-assign]
    runtime.alignment_verifier = verifier

    response = asyncio.run(runtime.reply(ReplyRequestShim("weekly report products?").payload()))

    assert queries == [None, "report_scope_products"]
    assert response.reply.text == "weekly report products: Product1, Product2"


def test_alignment_verdict_rejects_free_text_report_scope_refetch_query():
    try:
        ReplyAlignmentVerdict(
            aligned=False,
            safe_to_return=False,
            failure_code="missing_answer",
            remediation="refetch_report_scope",
            refined_evidence_query="weekly report product list scope",
        )
    except ValueError as exc:
        error = exc
    else:
        raise AssertionError("free-text report-scope refetch query must be rejected")

    assert "report_scope_products or report_scope_summary" in str(error)


def test_alignment_verifier_failure_does_not_return_action():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
    )
    install_fake_planner(runtime, make_weekly_plan_spec())

    class FailingVerifier:
        async def verify(self, **kwargs):
            del kwargs
            raise ValueError("invalid verifier output")

    runtime.alignment_verifier = FailingVerifier()

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert response.actions == []


def test_alignment_verifier_failure_preserves_non_compliant_refusal_text():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="refuse private contact request",
            artifact_kind="refusal",
            action_intent="refuse",
            ambiguity_slots=[],
            requested_capabilities=[],
            compliance={
                "is_compliant": False,
                "reason_code": "private_contact_request",
                "reason": "asks to add private contact",
            },
        ),
    )

    class FailingVerifier:
        async def verify(self, **kwargs):
            del kwargs
            raise ValueError("invalid verifier output")

    runtime.alignment_verifier = FailingVerifier()

    response = asyncio.run(runtime.reply(ReplyRequestShim("加你微信了，通过一下").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert response.reply.text == "老师请问具体是什么产品需求？业务问题请在当前群内沟通，便于留痕和统一回复。"
    assert response.actions == []


def test_alignment_verifier_recompose_once():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            evidence_query="衍复 公司介绍",
            ambiguity_slots=[],
        ),
    )
    composer_calls: list[str] = []

    class FakeComposer:
        async def kickoff_async(self, prompt, response_format):
            del response_format
            composer_calls.append(prompt)
            if len(composer_calls) == 1:
                text = "这是一段没有回答问题的文字。"
            else:
                text = "衍复是一家量化私募管理人。"
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id=f"resp-knowledge-{len(composer_calls)}",
                    reply=PrimaryReply(kind="answer", text=text),
                    actions=[],
                ),
                raw="",
            )

    class FakeEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
            from market_support_crewai_agent.runtime.evidence import EvidenceFact

            del canonical_context, plan, policy, action_history
            facts = [
                EvidenceFact(
                    fact_type="document_context",
                    value="衍复是一家量化私募管理人。",
                    source_type="document_mcp",
                    source_id="doc-1",
                )
            ]
            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=facts,
                business_facts=derive_business_facts(facts, request),
            )

    class RecomposeThenValidVerifier:
        def __init__(self):
            self.calls = 0

        async def verify(self, **kwargs):
            assert "response" in kwargs
            self.calls += 1
            if self.calls == 1:
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="composer_drift",
                    remediation="recompose",
                    composer_feedback="answer the company introduction question",
                )
            return ReplyAlignmentVerdict(aligned=True, safe_to_return=True, confidence=0.9)

    verifier = RecomposeThenValidVerifier()
    runtime.evidence_executor = FakeEvidenceExecutor()
    runtime.alignment_verifier = verifier
    runtime._build_agent = lambda *_args, **_kwargs: FakeComposer()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("介绍一下衍复").payload()))

    assert response.reply.text == "衍复是一家量化私募管理人。"
    assert len(composer_calls) == 2
    assert verifier.calls == 2
    assert "Previous alignment verdict JSON" in composer_calls[1]



def test_runtime_uses_smalltalk_composer_for_triggered_greeting():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="greeting",
            artifact_kind="smalltalk",
            action_intent="none",
            ambiguity_slots=[],
            compliance={
                "is_compliant": True,
                "reason_code": "unrelated_request",
                "reason": "greeting",
            },
        ),
    )
    prompts: list[str] = []
    stages: list[str] = []

    class FakeSmalltalkComposer:
        async def kickoff_async(self, prompt, response_format):
            prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-smalltalk",
                    reply=PrimaryReply(kind="answer", text="smalltalk response"),
                    actions=[],
                ),
                raw="",
            )

    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": stages.append(stage) or FakeSmalltalkComposer()
    )

    response = asyncio.run(runtime.reply(ReplyRequestShim("hi").payload()))

    assert stages == ["smalltalk_composer"]
    assert response.reply.kind == "answer"
    assert response.reply.text == "smalltalk response"
    assert response.actions == []
    assert prompts
    assert "base.smalltalk_composer" in prompts[0]
    assert '"actions": []' in prompts[0]

def test_runtime_skips_knowledge_composer_without_document_evidence():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            ambiguity_slots=[],
        ),
    )

    class ComposerShouldNotRun:
        async def kickoff_async(self, prompt, response_format):
            raise AssertionError("composer should not run without document evidence")

    class EmptyEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts

            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=[],
                business_facts=derive_business_facts([], request),
            )

    runtime.evidence_executor = EmptyEvidenceExecutor()
    runtime._build_agent = lambda *_args, **_kwargs: ComposerShouldNotRun()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("介绍一下衍复").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert response.actions == []


def test_runtime_raises_when_composer_returns_invalid_reply_contract():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local:23000",
            reply_alignment_verifier_enabled=False,
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            ambiguity_slots=[],
        ),
    )

    class BadComposer:
        async def kickoff_async(self, prompt, response_format):
            return SimpleNamespace(pydantic=None, raw='{"not": "a reply"}')

    class FakeEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.domain.business_facts import derive_business_facts
            from market_support_crewai_agent.runtime.evidence import EvidenceFact

            facts = [
                EvidenceFact(
                    fact_type="document_context",
                    value="文档证据",
                    source_type="document_mcp",
                    source_id="doc-1",
                )
            ]
            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=facts,
                business_facts=derive_business_facts(facts, request),
            )

    runtime.evidence_executor = FakeEvidenceExecutor()
    runtime._build_agent = lambda *_args, **_kwargs: BadComposer()  # type: ignore[method-assign]

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("介绍一下衍复").payload()))
    except AgentRuntimeError as exc:
        error = exc
    else:
        raise AssertionError("invalid composer output must raise")

    assert "invalid ReplyResponse contract" in str(error)


def test_same_conversation_key_reuses_prior_turns():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=EmptyPreflightService(),
    )
    prompts: list[str] = []
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_support_plan_spec(),
        prompts,
    )
    install_fake_clarification_composer(runtime, text="请确认一下具体需求。")

    asyncio.run(runtime.reply(ReplyRequestShim("first question").payload()))
    asyncio.run(runtime.reply(ReplyRequestShim("follow up").payload()))

    assert "Current user message:\nfirst question" in prompts[0]
    assert '"role": "user"' in prompts[1]
    assert "first question" in prompts[1]
    assert "Current user message:\nfollow up" in prompts[1]


def test_different_conversation_key_does_not_share_history():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=EmptyPreflightService(),
    )
    prompts: list[str] = []
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_support_plan_spec(),
        prompts,
    )
    install_fake_clarification_composer(runtime, text="请确认一下具体需求。")

    asyncio.run(runtime.reply(ReplyRequestShim("group one history").payload()))
    asyncio.run(
        runtime.reply(
            ReplyRequestShim(
                "group two current",
                conversation_key="wecom:group-2:sender-1",
                group_id="group-2",
            ).payload()
        )
    )

    assert "group one history" not in prompts[1]
    assert "Current user message:\ngroup two current" in prompts[1]


def test_adapter_execution_history_is_in_planner_prompt():
    store = ConversationStore(max_messages=12)
    ledger = ActionLedger()
    ledger.record_feedback(
        ActionFeedbackRequest.model_validate(
            {
                "conversation_key": "wecom:group-1:sender-1",
                "group_id": "group-1",
                "sender_id": "sender-1",
                "context_id": "msg-1",
                "response_id": "resp-1",
                "executions": [
                    {
                        "action_type": "send_weekly_report",
                        "status": "executed",
                        "action_id": "act-1",
                        "resolve_ref": "weekly:resolve-ref",
                        "material_type": "weekly",
                        "version": "20260529",
                        "adapter_result": {"ok": True, "private": "not prompted"},
                    }
                ],
            }
        )
    )
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        action_ledger=ledger,
        preflight_service=EmptyPreflightService(),
    )
    planner_prompts: list[str] = []
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        prompts=planner_prompts,
    )
    install_fake_clarification_composer(runtime, text="请确认一下具体需求。")

    asyncio.run(runtime.reply(ReplyRequestShim("刚才发了吗").payload()))

    assert '"ledger_summary"' in planner_prompts[0]
    assert '"has_recent_executed_actions": true' in planner_prompts[0]
    assert '"weekly"' in planner_prompts[0]
    assert "weekly:resolve-ref" not in planner_prompts[0]
    assert "not prompted" not in planner_prompts[0]


def test_runtime_passes_projected_context_to_planner_without_raw_huge_history():
    store = ConversationStore(max_messages=12)
    huge = "HUGE-HISTORY-CONTENT" * 500
    store.save_turn("wecom:group-1:sender-1", "old user", huge)
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            reply_alignment_verifier_enabled=False,
            agent_context_recent_turns_verbatim_count=0,
        ),
        conversation_store=store,
        preflight_service=EmptyPreflightService(),
    )
    prompts: list[str] = []
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_support_plan_spec(),
        prompts,
    )
    install_fake_clarification_composer(runtime, text="请确认一下具体需求。")

    asyncio.run(runtime.reply(ReplyRequestShim("current question").payload()))

    assert prompts
    assert "model-visible-context" in prompts[0]
    assert "compacted_summary" in prompts[0]
    assert huge not in prompts[0]
    assert "Current user message:\ncurrent question" in prompts[0]


def test_history_trims_to_max_messages():
    store = ConversationStore(max_messages=3)

    store.save_turn("key", "u1", "a1")
    store.save_turn("key", "u2", "a2")

    messages = store.get_recent("key")
    assert [message.content for message in messages] == ["a1", "u2", "a2"]


def test_sessions_expired_by_ttl_are_deleted():
    now = datetime(2026, 5, 22, tzinfo=timezone.utc)
    store = ConversationStore(ttl_seconds=10, now_factory=lambda: now)

    store.save_turn("expired", "u", "a")
    now = now + timedelta(seconds=11)

    assert store.cleanup_expired() == 1
    assert store.get_recent("expired") == []
    assert store.session_count() == 0


def test_max_sessions_cap_prevents_unbounded_growth():
    now = datetime(2026, 5, 22, tzinfo=timezone.utc)

    def tick():
        nonlocal now
        current = now
        now = now + timedelta(seconds=1)
        return current

    store = ConversationStore(max_sessions=2, now_factory=tick)

    store.save_turn("first", "u1", "a1")
    store.save_turn("second", "u2", "a2")
    store.save_turn("third", "u3", "a3")

    assert store.session_count() == 2
    assert store.get_recent("first") == []
    assert [message.content for message in store.get_recent("second")] == ["u2", "a2"]
    assert [message.content for message in store.get_recent("third")] == ["u3", "a3"]


def _test_settings() -> Settings:
    return Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False)


def _invalid_weekly_plan_raw() -> str:
    payload = make_weekly_plan_spec().model_dump(mode="json")
    payload["plan_units"][0].pop("selected_capability_id", None)
    return json.dumps(payload, ensure_ascii=False)


def _missing_ref_weekly_plan_raw() -> str:
    payload = make_weekly_plan_spec().model_dump(mode="json")
    payload["plan_units"][0].pop("evidence_contract_ref", None)
    payload["plan_units"][0]["evidence_contract"] = None
    return json.dumps(payload, ensure_ascii=False)


class ReplyRequestShim:
    def __init__(
        self,
        message: str,
        conversation_key: str = "wecom:group-1:sender-1",
        group_id: str = "group-1",
        sender_id: str = "sender-1",
    ) -> None:
        self.message = message
        self.conversation_key = conversation_key
        self.group_id = group_id
        self.sender_id = sender_id

    def payload(self):
        from market_support_crewai_agent.schemas import ReplyRequest

        return ReplyRequest.model_validate(
            make_payload(
                self.message,
                conversation_key=self.conversation_key,
                group_id=self.group_id,
                sender_id=self.sender_id,
            )
        )
