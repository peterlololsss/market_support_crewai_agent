from __future__ import annotations

from types import ModuleType

import pytest

from market_support_crewai_agent.runtime.recall import bm25_runtime
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    DocumentQaEntry,
    parse_qa_entries,
    search_qa_entries,
)


class _UnexpectedBm25Defect(RuntimeError):
    pass


def _entries() -> tuple[DocumentQaEntry, ...]:
    return parse_qa_entries(
        [
            {
                "id": "faq",
                "title": "常见问答",
                "content": "Q：什么是过拟合？\nA：过拟合说明。",
            }
        ]
    )


def test_search_qa_entries_fails_open_when_bm25_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the optional BM25 dependency cannot be imported.
    def unavailable_import(name: str) -> ModuleType:
        del name
        raise ImportError

    monkeypatch.setattr(bm25_runtime, "import_module", unavailable_import)

    # When: advisory QA recall attempts to rank the corpus.
    match = search_qa_entries("客户问过拟合是什么意思", _entries(), top_k=3)

    # Then: absence is represented as bounded unavailability for planner fallback.
    assert (match.status, match.reason_code) == ("unavailable", "qa_ranker_invalid")
    assert match.allow_planner_document_context_fallback is True


def test_search_qa_entries_propagates_unexpected_bm25_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: BM25 construction contains an unexpected programming defect.
    class DefectiveRetriever:
        def __init__(self) -> None:
            raise _UnexpectedBm25Defect("unexpected bm25 defect")

    fake_module = ModuleType("bm25s")
    vars(fake_module)["BM25"] = DefectiveRetriever

    def import_fake_module(name: str) -> ModuleType:
        del name
        return fake_module

    monkeypatch.setattr(bm25_runtime, "import_module", import_fake_module)

    # When/Then: the defect propagates rather than becoming provider unavailability.
    with pytest.raises(_UnexpectedBm25Defect, match="unexpected bm25 defect"):
        _ = search_qa_entries("客户问过拟合是什么意思", _entries(), top_k=3)
