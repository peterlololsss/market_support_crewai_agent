from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.audit import AuditStore
from market_support_crewai_agent.runtime.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.guardrails import ReplyContractError
from market_support_crewai_agent.runtime.input_guardrails import InputGuardrailError
from market_support_crewai_agent.runtime.planning import IntentFrame
from market_support_crewai_agent.runtime.reply_agent import (
    AgentRuntimeError,
    CrewAIReplyRuntime,
    build_reply,
)
from market_support_crewai_agent.runtime.response_ids import ensure_response_ids
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    AdapterResolveResult,
    PrimaryReply,
    ReplyResponse,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.server.main import app
from market_support_crewai_agent.settings import Settings

client = TestClient(app)


class FakePlannerAgent:
    def __init__(self, frame: IntentFrame | None = None, prompts: list[str] | None = None):
        self.frame = frame or make_intent_frame(
            artifact_kind="unclear",
            action_intent="none",
            report_scope="none",
            ambiguity_slots=["request_meaning"],
        )
        self.prompts = prompts

    async def kickoff_async(self, prompt, response_format):
        if self.prompts is not None:
            self.prompts.append(prompt)
        return SimpleNamespace(pydantic=self.frame, raw="")


def make_intent_frame(**overrides) -> IntentFrame:
    payload = {
        "user_need": "answer current market support request",
        "artifact_kind": "unclear",
        "action_intent": "none",
        "report_scope": "none",
        "ambiguity_slots": ["request_meaning"],
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal product or support request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    return IntentFrame.model_validate(payload)


def make_weekly_frame(**overrides) -> IntentFrame:
    payload = {
        "user_need": "send weekly report",
        "artifact_kind": "weekly_report",
        "action_intent": "send",
        "report_scope": "channel_all",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal weekly report request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    return IntentFrame.model_validate(payload)


def install_fake_planner(runtime: CrewAIReplyRuntime, frame: IntentFrame | None = None):
    runtime._build_planner_agent = lambda: FakePlannerAgent(frame)  # type: ignore[method-assign]


def make_payload(message: str = "hello", **overrides):
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": message,
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": [],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return payload


def resolved_item(resolve_type: str, **overrides) -> AdapterPreflightItem:
    payload = {
        "contract_version": "adapter-resolve",
        "resolve_type": resolve_type,
        "status": "resolved",
        "display_name": "测试渠道",
        "reason_code": "ok",
        "candidates": [],
        "channel_type": "bank",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["指增"],
        "resolved_at": 1,
        "resolve_ref": f"{resolve_type}:ref",
    }
    payload.update(overrides)
    return AdapterPreflightItem(
        resolve_type=resolve_type,
        result=AdapterResolveResult.model_validate(payload),
    )


class ResolvedWeeklyPreflight:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_strategies=None,
    ):
        del request, canonical_context, resolve_types, resolve_strategies
        return AdapterPreflightSnapshot(
            items=[
                resolved_item(
                    "weekly_report",
                    resolve_ref="weekly:ref",
                    period="20260529",
                    report_date="2026-05-29",
                    scope_status="included",
                    contains_strategy=True,
                )
            ]
        )


class MissingWeeklyWithSalesPreflight:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_strategies=None,
    ):
        del request, canonical_context, resolve_types, resolve_strategies
        missing = {
            "contract_version": "adapter-resolve",
            "resolve_type": "weekly_report",
            "status": "missing",
            "display_name": "测试渠道",
            "reason_code": "weekly_report_unavailable",
            "candidates": [],
            "channel_type": "bank",
            "available_materials": [],
            "available_strategies": [],
            "resolved_at": 1,
        }
        return AdapterPreflightSnapshot(
            items=[
                AdapterPreflightItem(
                    resolve_type="weekly_report",
                    result=AdapterResolveResult.model_validate(missing),
                ),
                resolved_item("sales_mention", resolve_ref="sales:ref"),
            ]
        )


class EmptyPreflightService:
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_strategies=None,
    ):
        del request, canonical_context, resolve_types, resolve_strategies
        return AdapterPreflightSnapshot.empty()


def test_crewai_agents_disable_execution_planning_without_delegation():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            llm_timeout_seconds=7,
            crewai_max_retry_limit=4,
        )
    )

    planner = runtime._build_planner_agent()
    composer = runtime._build_agent()

    assert planner.planning is False
    assert planner.allow_delegation is False
    assert planner.inject_date is True
    assert planner.llm.timeout == 7
    assert planner.max_retry_limit == 4
    assert composer.planning is False
    assert composer.allow_delegation is False
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
                report_scope="channel_all",
                strategy=None,
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
                "report_scope": "channel_all",
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
        Settings(llm_api_key="test-key", llm_timeout_seconds=0.01),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )

    class SlowPlannerAgent:
        async def kickoff_async(self, prompt, response_format):
            await asyncio.sleep(1)
            return SimpleNamespace(pydantic=make_intent_frame(), raw="")

    class ComposerShouldNotRun:
        async def kickoff_async(self, prompt, response_format):
            raise AssertionError("composer should not run after planner timeout")

    runtime._build_planner_agent = lambda: SlowPlannerAgent()  # type: ignore[method-assign]
    runtime._build_agent = lambda: ComposerShouldNotRun()  # type: ignore[method-assign]

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))
    except AgentRuntimeError as exc:
        error = exc
    else:
        raise AssertionError("slow planner should time out")

    assert str(error) == "CrewAI planner timed out"


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


def test_request_contract_rejects_unknown_material_type():
    response = client.post(
        "/reply",
        json=make_payload(available_materials=["calendar"]),
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
                "report_scope": "channel_all",
                "strategy": None,
            }
        )
    except ValidationError:
        return

    raise AssertionError("report send actions must include period and report_date")


def test_strategy_scoped_report_action_requires_strategy():
    from pydantic import ValidationError

    try:
        SendWeeklyReportAction.model_validate(
            {
                "type": "send_weekly_report",
                "action_id": "act-1",
                "resolve_type": "weekly_report",
                "resolve_ref": "weekly:ref",
                "report_scope": "strategy",
                "period": "20260529",
                "report_date": "2026-05-29",
            }
        )
    except ValidationError:
        return

    raise AssertionError("strategy-scoped report actions must include strategy")


def test_build_reply_uses_custom_settings_for_default_runtime_services(monkeypatch):
    settings = Settings(
        llm_api_key="test-key",
        doc_mcp_enabled=True,
        doc_mcp_base_url="http://doc-mcp.local:23000",
        agent_conversation_max_messages=3,
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
    install_fake_planner(runtime, make_weekly_frame())

    class ComposerShouldNotRun:
        async def kickoff_async(self, prompt, response_format):
            raise AssertionError("composer should not run for action responses")

    runtime._build_agent = lambda: ComposerShouldNotRun()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.reply.kind == "answer"
    assert response.reply.text == ""
    assert response.actions[0].type == "send_weekly_report"
    assert response.actions[0].resolve_ref == "weekly:ref"
    assert response.actions[0].period == "20260529"


def test_runtime_records_audit_before_raising_reply_validation_error(monkeypatch):
    audit_store = AuditStore()
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=ResolvedWeeklyPreflight(),
        audit_store=audit_store,
    )
    install_fake_planner(runtime, make_weekly_frame())

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
                    report_scope="channel_all",
                    strategy=None,
                    period="20260529",
                    report_date="2026-05-29",
                )
            ],
        )

    monkeypatch.setattr(
        "market_support_crewai_agent.runtime.reply_agent.render_directive",
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
    install_fake_planner(runtime, make_weekly_frame())

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
        ),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_intent_frame(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            report_scope="none",
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
            from market_support_crewai_agent.runtime.business_facts import derive_business_facts
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
    runtime._build_agent = lambda: FakeComposer()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("介绍一下衍复").payload()))

    assert response.reply.text == "文档证据回答"
    assert prompts
    assert "ExecutionPlan JSON" in prompts[0]


def test_runtime_skips_knowledge_composer_without_document_evidence():
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
        make_intent_frame(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            report_scope="none",
            ambiguity_slots=[],
        ),
    )

    class ComposerShouldNotRun:
        async def kickoff_async(self, prompt, response_format):
            raise AssertionError("composer should not run without document evidence")

    class EmptyEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.business_facts import derive_business_facts

            return SimpleNamespace(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=[],
                business_facts=derive_business_facts([], request),
            )

    runtime.evidence_executor = EmptyEvidenceExecutor()
    runtime._build_agent = lambda: ComposerShouldNotRun()  # type: ignore[method-assign]

    response = asyncio.run(runtime.reply(ReplyRequestShim("介绍一下衍复").payload()))

    assert response.reply.kind == "unable_to_answer"
    assert response.actions == []


def test_runtime_raises_when_composer_returns_invalid_reply_contract():
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
        make_intent_frame(
            user_need="answer knowledge question",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["document_context"],
            report_scope="none",
            ambiguity_slots=[],
        ),
    )

    class BadComposer:
        async def kickoff_async(self, prompt, response_format):
            return SimpleNamespace(pydantic=None, raw='{"not": "a reply"}')

    class FakeEvidenceExecutor:
        async def execute(self, request, canonical_context, plan, policy, action_history=None):
            from market_support_crewai_agent.runtime.business_facts import derive_business_facts
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
    runtime._build_agent = lambda: BadComposer()  # type: ignore[method-assign]

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("介绍一下衍复").payload()))
    except AgentRuntimeError as exc:
        error = exc
    else:
        raise AssertionError("invalid composer output must raise")

    assert "invalid ReplyResponse contract" in str(error)


def test_runtime_raises_when_compiled_plan_is_invalid():
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_weekly_frame(requested_capabilities=["document_context"]),
    )

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))
    except AgentRuntimeError as exc:
        error = exc
    else:
        raise AssertionError("invalid compiled plan must raise")

    assert "compiled execution plan failed validation" in str(error)


def test_same_conversation_key_reuses_prior_turns():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=EmptyPreflightService(),
    )
    prompts: list[str] = []
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_intent_frame(),
        prompts,
    )

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
        make_intent_frame(),
        prompts,
    )

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

    asyncio.run(runtime.reply(ReplyRequestShim("刚才发了吗").payload()))

    assert '"ledger_summary"' in planner_prompts[0]
    assert '"has_recent_executed_actions": true' in planner_prompts[0]
    assert '"weekly"' in planner_prompts[0]
    assert "weekly:resolve-ref" not in planner_prompts[0]
    assert "not prompted" not in planner_prompts[0]


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
    return Settings(llm_api_key="test-key")


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
