from __future__ import annotations

from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.reply_agent import AgentRuntimeError
from market_support_crewai_agent.schemas import ReplyResponse, SendTextAction
from market_support_crewai_agent.server.main import app


client = TestClient(app)


def make_payload(message: str = "hello", **overrides):
    payload = {
        "context_id": "msg-1",
        "session_id": "session-1",
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
