from __future__ import annotations

from market_support_crewai_agent.runtime.context.payload_store import (
    ScopedContextPayloadStoreV1,
)
from tests.helpers.reply_contract_requests import make_state_key


def test_payload_store_uses_stable_handles_and_retains_metadata():
    state_key = make_state_key()
    store = ScopedContextPayloadStoreV1(
        conversation_ttl_seconds=60,
        direct_audit_ttl_seconds=60,
        max_payloads=2,
    )

    handle = store.put(state_key, "payload", metadata={"source_id": "doc-1"})
    stored = store.get(state_key, handle)

    assert handle.value.startswith("cpv1:")
    assert stored is not None
    assert stored.payload == "payload"
    assert stored.metadata == {"source_id": "doc-1"}
    assert store.count() == 1


def test_payload_store_is_bounded():
    state_key = make_state_key()
    store = ScopedContextPayloadStoreV1(
        conversation_ttl_seconds=60,
        direct_audit_ttl_seconds=60,
        max_payloads=1,
    )

    first = store.put(state_key, "first", metadata={})
    second = store.put(state_key, "second", metadata={})

    assert store.get(state_key, first) is None
    second_payload = store.get(state_key, second)
    assert second_payload is not None
    assert second_payload.payload == "second"
    assert store.count() == 1
    store.clear()
    assert store.count() == 0
