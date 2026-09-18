from __future__ import annotations

import os

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

import pytest

from market_support_crewai_agent.runtime.identity import VerifiedRequestEnvelopeV1
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.validation.reply_validator import (
    ReplyContractError,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from tests.helpers.reply_contract_requests import make_v2_payload
from tests.helpers.reply_contract_runtime import (
    client,
)

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_reply_requires_api_key_when_configured(monkeypatch: pytest.MonkeyPatch):
    called = False

    async def fake_build_reply(_request: VerifiedRequestEnvelopeV1):
        nonlocal called
        called = True
        return ReplyResponse(
            response_id="resp-test",
            reply=PrimaryReply(kind="answer", text="ok"),
            actions=[],
        )

    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply", fake_build_reply
    )

    response = client.post("/reply", json=make_v2_payload("any message"))

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}
    assert called is False


def test_reply_accepts_bearer_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = ReplyResponse(
        response_id="resp-test",
        reply=PrimaryReply(kind="answer", text="ok"),
        actions=[],
    )

    async def fake_build_reply(_request: VerifiedRequestEnvelopeV1):
        return expected

    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply", fake_build_reply
    )

    response = client.post(
        "/reply",
        json=make_v2_payload("any message"),
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json()["response_id"] == "resp-test"


def test_reply_returns_502_when_runtime_fails(monkeypatch: pytest.MonkeyPatch):
    async def fake_build_reply(_request: VerifiedRequestEnvelopeV1):
        raise AgentRuntimeError("runtime failed")

    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply", fake_build_reply
    )

    response = client.post(
        "/reply", json=make_v2_payload("any message"), headers={"X-API-Key": "secret"}
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "runtime failed"}


def test_reply_returns_502_when_contract_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_build_reply(_request: VerifiedRequestEnvelopeV1):
        raise ReplyContractError("invalid reply")

    monkeypatch.setenv("MARKET_AGENT_API_KEY", "secret")
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply", fake_build_reply
    )

    response = client.post(
        "/reply", json=make_v2_payload("any message"), headers={"X-API-Key": "secret"}
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "invalid reply"}
