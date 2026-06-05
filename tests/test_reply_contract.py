from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.input_guardrails import InputGuardrailError
from market_support_crewai_agent.runtime.reply_agent import (
    AgentRuntimeError,
    CrewAIReplyRuntime,
    build_reply,
)
from market_support_crewai_agent.runtime.response_ids import ensure_response_ids
from market_support_crewai_agent.runtime.planning import (
    ReplyPlan,
)
from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightItem,
    AdapterPreflightSnapshot,
    NoopAdapterPreflightService,
)
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    PrimaryReply,
    ReplyResponse,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.server.main import app
from market_support_crewai_agent.settings import Settings

client = TestClient(app)


class FakePlannerAgent:
    def __init__(self, plan: ReplyPlan | None = None, prompts: list[str] | None = None):
        self.plan = plan or make_reply_plan()
        self.prompts = prompts

    async def kickoff_async(self, prompt, response_format):
        if self.prompts is not None:
            self.prompts.append(prompt)
        return SimpleNamespace(pydantic=self.plan, raw="")


def make_reply_plan(**overrides) -> ReplyPlan:
    payload = {
        "user_need": "answer current market support request",
        "intent": "clarification",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal product or support request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    return ReplyPlan.model_validate(payload)


def install_fake_planner(runtime: CrewAIReplyRuntime, plan: ReplyPlan | None = None):
    runtime._build_planner_agent = lambda: FakePlannerAgent(plan)  # type: ignore[method-assign]


def test_crewai_planner_agent_uses_planning_config_without_delegation():
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            llm_timeout_seconds=7,
            crewai_max_retry_limit=4,
        )
    )

    planner = runtime._build_planner_agent()
    composer = runtime._build_agent()

    assert planner.planning is True
    assert planner.planning_config is not None
    assert planner.planning_config.reasoning_effort == "medium"
    assert planner.planning_config.max_attempts == 2
    assert planner.planning_config.observe_steps is False
    assert planner.allow_delegation is False
    assert planner.inject_date is True
    assert planner.llm.timeout == 7
    assert planner.max_retry_limit == 4
    assert composer.planning is False
    assert composer.planning_config is None
    assert composer.allow_delegation is False
    assert composer.llm.timeout == 7
    assert composer.max_retry_limit == 4


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
        actions=[SendWeeklyReportAction(type="send_weekly_report", action_id="act-1")],
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
            "text_format": "plain_text",
            "mentions": [],
        },
        "actions": [{"action_id": "act-1", "type": "send_weekly_report"}],
    }


def test_reply_route_rejects_message_over_configured_input_limit(monkeypatch):
    monkeypatch.setenv("AGENT_INPUT_MAX_MESSAGE_CHARS", "5")

    response = client.post("/reply", json=make_payload("abcdef"))

    assert response.status_code == 413
    assert "message exceeds configured input guardrail limit" in response.json()[
        "detail"
    ]


def test_runtime_input_guardrail_runs_before_llm_configuration():
    runtime = CrewAIReplyRuntime(
        Settings(agent_input_max_message_chars=5),
        conversation_store=ConversationStore(),
        preflight_service=NoopAdapterPreflightService(),
    )

    import asyncio

    try:
        asyncio.run(runtime.reply(ReplyRequestShim("abcdef").payload()))
    except InputGuardrailError as exc:
        error = exc
    else:
        raise AssertionError("input guardrail should reject oversized message")

    assert error.code == "message_too_long"
    assert error.metadata == {
        "message_length": 6,
        "max_message_chars": 5,
    }


def test_runtime_times_out_slow_crewai_planner_before_composer_runs():
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key", llm_timeout_seconds=0.01),
        conversation_store=ConversationStore(),
        preflight_service=NoopAdapterPreflightService(),
    )

    class SlowPlannerAgent:
        async def kickoff_async(self, prompt, response_format):
            import asyncio

            await asyncio.sleep(1)
            return SimpleNamespace(pydantic=make_reply_plan(), raw="")

    class ComposerShouldNotRun:
        async def kickoff_async(self, prompt, response_format):
            raise AssertionError("composer should not run after planner timeout")

    runtime._build_planner_agent = lambda: SlowPlannerAgent()  # type: ignore[method-assign]
    runtime._build_agent = lambda: ComposerShouldNotRun()  # type: ignore[method-assign]

    import asyncio

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


def test_request_contract_rejects_unknown_material_type():
    response = client.post(
        "/reply",
        json=make_payload(available_materials=["calendar"]),
    )

    assert response.status_code == 422


def test_request_contract_rejects_extra_fields():
    response = client.post(
        "/reply",
        json=make_payload(extra_field="not allowed"),
    )

    assert response.status_code == 422


def test_request_contract_requires_conversation_key_group_id_and_sender_id():
    for required_field in ("conversation_key", "group_id", "sender_id"):
        payload = make_payload()
        payload.pop(required_field)

        response = client.post("/reply", json=payload)

        assert response.status_code == 422


def test_request_contract_accepts_optional_context_id(monkeypatch):
    expected = ReplyResponse(
        response_id="resp-ok",
        reply=PrimaryReply(kind="answer", text="ok"),
        actions=[],
    )

    async def fake_build_reply(request):
        assert request.context_id is None
        return expected

    monkeypatch.setattr("market_support_crewai_agent.server.main.build_reply", fake_build_reply)
    payload = make_payload()
    payload.pop("context_id")

    response = client.post("/reply", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "contract_version": "reply",
        "response_id": "resp-ok",
        "reply": {
            "kind": "answer",
            "text": "ok",
            "text_format": "plain_text",
            "mentions": [],
        },
        "actions": [],
    }


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
                "contract_version": "reply.v2",
                "response_id": "resp-unsupported",
                "reply": {"kind": "answer", "text": "ok", "mentions": []},
                "actions": [],
            }
        )
    except ValidationError:
        return

    raise AssertionError("reply.v2 must not be accepted")


def test_build_reply_uses_custom_settings_for_default_runtime_services(monkeypatch):
    settings = Settings(
        llm_api_key="test-key",
        adapter_preflight_enabled=False,
        doc_mcp_enabled=True,
        doc_mcp_base_url="http://doc-mcp.local:23000",
        agent_conversation_max_messages=3,
    )
    seen: dict[str, object] = {}

    async def fake_reply(self, request):
        seen["settings"] = self.settings
        seen["preflight_enabled"] = self.preflight_service.enabled
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

    import asyncio

    response = asyncio.run(
        build_reply(ReplyRequestShim("介绍一下中证1000").payload(), settings=settings)
    )

    assert response.response_id == "resp-ok"
    assert seen["settings"] is settings
    assert seen["preflight_enabled"] is False
    assert seen["document_settings"] is settings
    assert seen["conversation_max_messages"] == 3


def test_same_conversation_key_reuses_prior_turns():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=NoopAdapterPreflightService(),
    )
    install_fake_planner(runtime)
    prompts: list[str] = []

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id=f"resp-{len(prompts)}",
                    reply=PrimaryReply(kind="answer", text=f"answer-{len(prompts)}"),
                    actions=[],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    import asyncio

    asyncio.run(runtime.reply(ReplyRequestShim("first question").payload()))
    asyncio.run(runtime.reply(ReplyRequestShim("follow up").payload()))

    assert "Current user message:\nfirst question" in prompts[0]
    assert '"role": "user"' in prompts[1]
    assert "first question" in prompts[1]
    assert "answer-1" in prompts[1]
    assert "Current user message:\nfollow up" in prompts[1]


def test_different_conversation_key_does_not_share_history():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=NoopAdapterPreflightService(),
    )
    install_fake_planner(runtime)
    prompts: list[str] = []

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-ok",
                    reply=PrimaryReply(kind="answer", text="ok"),
                    actions=[],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    import asyncio

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


def test_runtime_prompts_include_canonical_entities():
    from market_support_crewai_agent.schemas import ReplyRequest

    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=NoopAdapterPreflightService(),
    )
    planner_prompts: list[str] = []
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        make_reply_plan(),
        planner_prompts,
    )
    composer_prompts: list[str] = []

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            composer_prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-ok",
                    reply=PrimaryReply(kind="answer", text="ok"),
                    actions=[],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    import asyncio

    request = ReplyRequest.model_validate(
        make_payload(
            "1000所有号的周报我想看看",
            available_strategies=["中证500", "中证1000"],
        )
    )
    asyncio.run(runtime.reply(request))

    assert "Canonical entities JSON" in planner_prompts[0]
    assert '"selected_strategy": "中证1000"' in planner_prompts[0]
    assert "Canonical entities JSON" in composer_prompts[0]
    assert '"selected_strategy": "中证1000"' in composer_prompts[0]


def test_adapter_execution_history_is_in_runtime_prompt():
    store = ConversationStore(max_messages=12)
    ledger = ActionLedger()
    for status in ("failed", "skipped"):
        ledger.record_feedback(
            ActionFeedbackRequest.model_validate(
                {
                    "conversation_key": "wecom:group-1:sender-1",
                    "group_id": "group-1",
                    "sender_id": "sender-1",
                    "context_id": "msg-non-executed",
                    "response_id": "resp-non-executed",
                    "executions": [
                        {
                            "action_type": "send_material",
                            "status": status,
                            "action_id": "act-non-executed",
                            "material_type": "monthly",
                            "version": "202605",
                            "adapter_result": {"ok": False},
                        }
                    ],
                }
            )
        )
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
                        "action_type": "send_material",
                        "status": "executed",
                        "action_id": "act-1",
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
        preflight_service=NoopAdapterPreflightService(),
    )
    planner_prompts: list[str] = []
    runtime._build_planner_agent = lambda: FakePlannerAgent(  # type: ignore[method-assign]
        prompts=planner_prompts,
    )
    prompts: list[str] = []

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-ok",
                    reply=PrimaryReply(kind="answer", text="ok"),
                    actions=[],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    import asyncio

    asyncio.run(runtime.reply(ReplyRequestShim("刚才发了吗").payload()))

    assert '"ledger_summary"' in planner_prompts[0]
    assert '"has_recent_executed_actions": true' in planner_prompts[0]
    assert '"recent_material_types": [' in planner_prompts[0]
    assert '"weekly"' in planner_prompts[0]
    assert "Recent executed adapter actions JSON" in prompts[0]
    assert '"ledger_summary"' in prompts[0]
    assert '"has_recent_executed_actions": true' in prompts[0]
    assert '"response_id": "resp-1"' in prompts[0]
    assert '"action_id": "act-1"' in prompts[0]
    assert '"status": "executed"' in prompts[0]
    assert '"material_type": "weekly"' in prompts[0]
    assert "resp-non-executed" not in prompts[0]
    assert "act-non-executed" not in prompts[0]
    assert '"status": "failed"' not in prompts[0]
    assert '"status": "skipped"' not in prompts[0]
    assert "not prompted" not in prompts[0]
    assert "opaque" not in planner_prompts[0]
    assert "opaque" not in prompts[0]


def test_adapter_preflight_is_in_runtime_prompt():
    store = ConversationStore(max_messages=12)

    class FakePreflight:
        async def collect(
                self,
                request,
                canonical_context=None,
                resolve_types=None,
                resolve_strategies=None,
        ):
            del canonical_context, resolve_types, resolve_strategies
            from market_support_crewai_agent.schemas import AdapterResolveResult

            return AdapterPreflightSnapshot(
                items=[
                    AdapterPreflightItem(
                        resolve_type="weekly_report",
                        result=AdapterResolveResult.model_validate(
                            {
                                "contract_version": "adapter-resolve.v1",
                                "resolve_type": "weekly_report",
                                "status": "resolved",
                                "display_name": request.dist_channel_name,
                                "reason_code": "ok",
                                "candidates": [],
                                "channel_type": "bank",
                                "available_materials": ["weekly"],
                                "available_strategies": ["指增"],
                                "resolved_at": 1,
                                "period": "20260529",
                                "scope_status": "unknown",
                                "card_ref": "wecom-adapter:hidden",
                            }
                        ),
                    ),
                    AdapterPreflightItem(
                        resolve_type="material_pack",
                        error="adapter resolve request failed",
                    ),
                ]
            )

    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=FakePreflight(),
    )
    install_fake_planner(runtime)
    prompts: list[str] = []

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-ok",
                    reply=PrimaryReply(kind="answer", text="ok"),
                    actions=[],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    import asyncio

    asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert "Adapter preflight JSON" in prompts[0]
    assert "Validated ReplyPlan JSON" in prompts[0]
    assert '"resolve_type": "weekly_report"' in prompts[0]
    assert '"status": "resolved"' in prompts[0]
    assert '"period": "20260529"' in prompts[0]
    assert '"resolve_type": "material_pack"' in prompts[0]
    assert '"status": "adapter_unavailable"' in prompts[0]
    assert "wecom-adapter:hidden" not in prompts[0]


def test_runtime_applies_preflight_guardrail_to_agent_response():
    store = ConversationStore(max_messages=12)

    class FakePreflight:
        async def collect(
                self,
                request,
                canonical_context=None,
                resolve_types=None,
                resolve_strategies=None,
        ):
            del canonical_context, resolve_types, resolve_strategies
            from market_support_crewai_agent.schemas import AdapterResolveResult

            return AdapterPreflightSnapshot(
                items=[
                    AdapterPreflightItem(
                        resolve_type="weekly_report",
                        result=AdapterResolveResult.model_validate(
                            {
                                "contract_version": "adapter-resolve.v1",
                                "resolve_type": "weekly_report",
                                "status": "missing",
                                "display_name": request.dist_channel_name,
                                "reason_code": "weekly_report_unavailable",
                                "candidates": [],
                                "channel_type": "bank",
                                "available_materials": [],
                                "available_strategies": [],
                                "resolved_at": 1,
                            }
                        ),
                    ),
                    AdapterPreflightItem(
                        resolve_type="sales_mention",
                        result=AdapterResolveResult.model_validate(
                            {
                                "contract_version": "adapter-resolve.v1",
                                "resolve_type": "sales_mention",
                                "status": "resolved",
                                "display_name": request.dist_channel_name,
                                "reason_code": "ok",
                                "candidates": [],
                                "channel_type": "bank",
                                "available_materials": [],
                                "available_strategies": [],
                                "resolved_at": 1,
                            }
                        ),
                    ),
                ]
            )

    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=FakePreflight(),
    )
    install_fake_planner(
        runtime,
        make_reply_plan(
            intent="send_weekly_report",
            required_adapter_resolves=["weekly_report"],
            evidence_requests=[
                {
                    "capability": "resolve_weekly_report",
                    "reason": "confirm weekly report can be sent",
                }
            ],
            candidate_actions=[{"type": "send_weekly_report", "report_scope": "channel_all"}],
        ),
    )

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-agent",
                    reply=PrimaryReply(kind="answer", text=""),
                    actions=[
                        SendWeeklyReportAction(
                            type="send_weekly_report",
                            action_id="act-weekly",
                        )
                    ],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    import asyncio

    response = asyncio.run(runtime.reply(ReplyRequestShim("请发周报").payload()))

    assert response.reply.text.startswith("目前这个渠道下我没有看到可发送的对应材料")
    assert response.reply.kind == "human_handoff"
    assert len(response.reply.mentions) == 1
    assert response.reply.mentions[0].type == "sales"
    assert response.actions == []


def test_runtime_uses_safe_fallback_for_non_compliant_plan_without_repair():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=NoopAdapterPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_reply_plan(
            user_need="refuse expected return request",
            intent="refusal",
            compliance={
                "is_compliant": False,
                "reason_code": "expected_or_target_return",
                "reason": "expected return requests must be refused",
            },
            confidence=0.9,
        ),
    )
    calls = {"composer": 0}

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            del prompt, response_format
            calls["composer"] += 1
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    response_id="resp-agent",
                    reply=PrimaryReply(kind="answer", text="预计收益可以看周报。"),
                    actions=[
                        SendWeeklyReportAction(
                            type="send_weekly_report",
                            action_id="act-weekly",
                        )
                    ],
                ),
                raw="",
            )

    runtime._build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    import asyncio

    response = asyncio.run(
        runtime.reply(ReplyRequestShim("请问产品预计收益多少？").payload())
    )

    assert calls["composer"] == 1
    assert response.reply.kind == "unable_to_answer"
    assert "不设置预计收益" in response.reply.text
    assert response.actions == []


def test_runtime_falls_back_when_agent_returns_invalid_reply_contract():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(
        _test_settings(),
        conversation_store=store,
        preflight_service=NoopAdapterPreflightService(),
    )
    install_fake_planner(runtime)

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            return SimpleNamespace(pydantic=None, raw='{"not": "a reply"}')

    runtime._build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    import asyncio

    response = asyncio.run(runtime.reply(ReplyRequestShim("bad output").payload()))

    assert response.response_id.startswith("resp-")
    assert response.reply.kind == "no_reply"
    assert response.reply.text == ""
    assert response.actions == []


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
