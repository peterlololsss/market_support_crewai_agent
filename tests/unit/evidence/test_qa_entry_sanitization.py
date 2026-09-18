from __future__ import annotations

from market_support_crewai_agent.runtime.recall.document_qa_recall import (
    parse_qa_entries,
    search_qa_entries,
)


def test_qa_entry_candidates_do_not_expose_raw_document_locators():
    entries = parse_qa_entries(
        [
            {
                "id": "file:///Users/ivan/secret.md",
                "title": "secret token=abc",
                "content": "Q：测试问题\nA：测试答案",
            }
        ]
    )

    match = search_qa_entries("测试问题", entries)
    payload = match.to_prompt_dict()

    assert match.status == "matched"
    exposed = str(payload)
    assert "file://" not in exposed
    assert "/Users/" not in exposed
    assert "token=abc" not in exposed


def test_qa_entry_candidates_do_not_expose_bare_secret_labels():
    entries = parse_qa_entries(
        [
            {
                "id": "token secretonly",
                "title": "Basic supersecret",
                "content": "Q：测试问题\nA：测试答案\nsecret supersecret\napi key deadbeef",
            }
        ]
    )

    match = search_qa_entries("测试问题", entries)
    payload = match.to_prompt_dict()

    assert match.status == "matched"
    exposed = str(payload)
    assert "token secretonly" not in exposed
    assert "secretonly" not in exposed
    assert "Basic supersecret" not in exposed
    assert "supersecret" not in exposed
    assert "api key deadbeef" not in exposed
    assert "deadbeef" not in exposed
