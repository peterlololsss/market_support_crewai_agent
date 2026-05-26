from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.conversation_store import ConversationStore
from market_support_crewai_agent.runtime.reply_agent import (
    AgentRuntimeError,
    CrewAIReplyRuntime,
)
from market_support_crewai_agent.schemas import ReplyResponse, SendTextAction
from market_support_crewai_agent.server.main import app
from market_support_crewai_agent.settings import Settings


client = TestClient(app)


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


def test_reply_returns_runtime_response_without_business_rewrite(monkeypatch):
    expected = ReplyResponse(
        text="runtime decided text",
        actions=[SendTextAction(type="send_text", text="runtime action text")],
    )

    async def fake_build_reply(request):
        return expected

    monkeypatch.setattr("market_support_crewai_agent.server.main.build_reply", fake_build_reply)

    response = client.post("/reply", json=make_payload("any message"))

    assert response.status_code == 200
    assert response.json() == {
        "text": "runtime decided text",
        "actions": [{"type": "send_text", "text": "runtime action text"}],
    }


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
    expected = ReplyResponse(text="ok", actions=[])

    async def fake_build_reply(request):
        assert request.context_id is None
        return expected

    monkeypatch.setattr("market_support_crewai_agent.server.main.build_reply", fake_build_reply)
    payload = make_payload()
    payload.pop("context_id")

    response = client.post("/reply", json=payload)

    assert response.status_code == 200
    assert response.json() == {"text": "ok", "actions": []}


def test_request_contract_rejects_removed_trigger_and_session_fields():
    for removed_field in ("session_id", "bot_mentioned", "trigger_reason"):
        response = client.post(
            "/reply",
            json=make_payload(**{removed_field: "not allowed"}),
        )

        assert response.status_code == 422


def test_same_conversation_key_reuses_prior_turns():
    store = ConversationStore(max_messages=12)
    runtime = CrewAIReplyRuntime(_test_settings(), conversation_store=store)
    prompts: list[str] = []

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(
                    text=f"answer-{len(prompts)}",
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
    runtime = CrewAIReplyRuntime(_test_settings(), conversation_store=store)
    prompts: list[str] = []

    class FakeAgent:
        async def kickoff_async(self, prompt, response_format):
            prompts.append(prompt)
            return SimpleNamespace(
                pydantic=ReplyResponse(text="ok", actions=[]),
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


def test_history_trims_to_max_messages():
    store = ConversationStore(max_messages=3)

    store.save_turn("key", "u1", "a1")
    store.save_turn("key", "u2", "a2")

    messages = store.get_recent("key")
    assert [message.content for message in messages] == ["a1", "u2", "a2"]


def test_sessions_older_than_ttl_are_deleted():
    now = datetime(2026, 5, 22, tzinfo=timezone.utc)
    store = ConversationStore(ttl_seconds=10, now_factory=lambda: now)

    store.save_turn("old", "u", "a")
    now = now + timedelta(seconds=11)

    assert store.cleanup_expired() == 1
    assert store.get_recent("old") == []
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
