from __future__ import annotations

from market_support_crewai_agent.runtime.context.payload_store import ContextPayloadStore


def test_payload_store_uses_stable_handles_and_retains_metadata():
    store = ContextPayloadStore(max_payloads=2)

    handle = store.put("payload", {"source_id": "doc-1"})
    same = store.put("payload", {"source_id": "doc-1"})

    assert handle == same
    assert handle.startswith("ctx-payload:")
    assert store.get(handle) == {
        "payload": "payload",
        "metadata": {"source_id": "doc-1"},
    }
    assert store.count() == 1


def test_payload_store_is_bounded():
    store = ContextPayloadStore(max_payloads=1)

    first = store.put("first", {})
    second = store.put("second", {})

    assert store.get(first) is None
    assert store.get(second)["payload"] == "second"
    assert store.count() == 1
    store.clear()
    assert store.count() == 0
