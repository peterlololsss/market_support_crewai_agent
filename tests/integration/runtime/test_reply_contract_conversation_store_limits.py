from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"


from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from tests.helpers.reply_contract_requests import make_state_key

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_history_trims_to_max_messages():
    store = ConversationStore(max_messages=3)
    state_key = make_state_key()

    store.save_turn(state_key, "u1", "a1")
    store.save_turn(state_key, "u2", "a2")

    messages = store.get_recent(state_key)
    assert [message.content for message in messages] == ["a1", "u2", "a2"]


def test_sessions_expired_by_ttl_are_deleted():
    now = datetime(2026, 5, 22, tzinfo=UTC)
    store = ConversationStore(ttl_seconds=10, now_factory=lambda: now)
    state_key = make_state_key()

    store.save_turn(state_key, "u", "a")
    now = now + timedelta(seconds=11)

    assert store.cleanup_expired() == 1
    assert store.get_recent(state_key) == []
    assert store.session_count() == 0


def test_max_sessions_cap_prevents_unbounded_growth():
    now = datetime(2026, 5, 22, tzinfo=UTC)

    def tick():
        nonlocal now
        current = now
        now = now + timedelta(seconds=1)
        return current

    store = ConversationStore(max_sessions=2, now_factory=tick)
    first = make_state_key(request_id="req:first")
    second = make_state_key(
        request_id="req:second",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": "tenant:test",
            "group_ref": "group:second",
            "principal_ref": "principal:sender-1",
        },
    )
    third = make_state_key(
        request_id="req:third",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": "tenant:test",
            "group_ref": "group:third",
            "principal_ref": "principal:sender-1",
        },
    )

    store.save_turn(first, "u1", "a1")
    store.save_turn(second, "u2", "a2")
    store.save_turn(third, "u3", "a3")

    assert store.session_count() == 2
    assert store.get_recent(first) == []
    assert [message.content for message in store.get_recent(second)] == ["u2", "a2"]
    assert [message.content for message in store.get_recent(third)] == ["u3", "a3"]
