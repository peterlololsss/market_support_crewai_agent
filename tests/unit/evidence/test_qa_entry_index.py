from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType

import pytest

from market_support_crewai_agent.runtime.recall import bm25_runtime
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    DocumentQaEntry,
    KnowledgeQaCandidate,
    KnowledgeQaMatch,
    parse_qa_entries,
    search_qa_entries,
)


def _bm25_entries() -> tuple[DocumentQaEntry, ...]:
    return parse_qa_entries(
        [
            {
                "id": "faq",
                "title": "常见问答",
                "content": "Q：什么是过拟合？\nA：过拟合说明。",
            }
        ]
    )


def _module_importer(module: ModuleType) -> Callable[[str], ModuleType]:
    def import_module(name: str) -> ModuleType:
        del name
        return module

    return import_module


def test_parse_qa_entries_extracts_multiple_question_answer_forms() -> None:
    entries = parse_qa_entries(
        [
            {
                "id": "faq",
                "title": "常见问答",
                "content": (
                    "Q：什么是过拟合？\n"
                    "A：过拟合指模型在样本内过度拟合，样本外失效。\n"
                    "补充：需要看样本外表现。\n\n"
                    "问题：赎回什么时候到账？\n"
                    "答案：以产品合同和管理人公告为准。"
                ),
            }
        ]
    )

    assert [entry.question for entry in entries] == [
        "什么是过拟合？",
        "赎回什么时候到账？",
    ]
    assert "补充" in entries[0].answer
    assert entries[0].doc_id == "faq"
    assert entries[0].entry_id.startswith("qa:")


def test_parse_qa_entries_splits_slash_joined_question_variants() -> None:
    entries = parse_qa_entries(
        [
            {
                "id": "灵活对冲",
                "title": "灵活对冲",
                "content": (
                    "Q：衍复灵活对冲策略的因子超额贡献分别是多少？/"
                    "衍复灵活对冲策略因子的来源都有哪些？/"
                    "灵活对冲策略与市场中性策略有什么区别？/"
                    "灵活对冲策略的市值敞口是多少？\n"
                    "A：灵活对冲策略会结合多类收益来源，并根据策略约束管理敞口。"
                ),
            }
        ]
    )

    assert [entry.question for entry in entries] == [
        "衍复灵活对冲策略的因子超额贡献分别是多少？",
        "衍复灵活对冲策略因子的来源都有哪些？",
        "灵活对冲策略与市场中性策略有什么区别？",
        "灵活对冲策略的市值敞口是多少？",
    ]
    assert {entry.answer for entry in entries} == {
        "灵活对冲策略会结合多类收益来源，并根据策略约束管理敞口。"
    }


def test_search_qa_entries_returns_compact_question_candidates_only() -> None:
    entries = parse_qa_entries(
        [
            {
                "id": "faq",
                "title": "常见问答",
                "content": "Q：什么是过拟合？\nA：这是完整答案，应该只作为预览进入规划。",
            }
        ]
    )

    match = search_qa_entries("客户问过拟合是什么意思", entries, top_k=3)

    assert match.status == "matched"
    assert [candidate.question for candidate in match.candidates] == ["什么是过拟合？"]
    assert "answer_preview" not in match.candidates[0].model_dump(mode="json")
    assert "not proof of absence" in match.planner_guidance


def test_search_qa_entries_does_not_match_only_common_question_tokens() -> None:
    entries = parse_qa_entries(
        [
            {
                "id": "faq",
                "title": "常见问答",
                "content": "Q：什么是过拟合？\nA：过拟合说明。",
            }
        ]
    )

    match = search_qa_entries("什么是", entries, top_k=3)

    assert match.status == "no_match"
    assert match.reason_code == "empty_query"


def test_search_qa_entries_fails_open_when_bm25_factory_is_not_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("bm25s")
    vars(fake_module)["BM25"] = "not-callable"
    monkeypatch.setattr(bm25_runtime, "import_module", _module_importer(fake_module))

    match = search_qa_entries("客户问过拟合是什么意思", _bm25_entries(), top_k=3)

    assert match.status == "unavailable"
    assert match.reason_code == "qa_ranker_invalid"
    assert match.allow_planner_document_context_fallback is True


def test_search_qa_entries_rejects_wrong_bm25_factory_signature_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls = 0

    def wrong_factory(required_extra: str) -> None:
        nonlocal factory_calls
        del required_extra
        factory_calls += 1
        raise AssertionError("factory should not be invoked")

    fake_module = ModuleType("bm25s")
    vars(fake_module)["BM25"] = wrong_factory
    monkeypatch.setattr(bm25_runtime, "import_module", _module_importer(fake_module))

    match = search_qa_entries("客户问过拟合是什么意思", _bm25_entries(), top_k=3)

    assert match.status == "unavailable"
    assert match.reason_code == "qa_ranker_invalid"
    assert factory_calls == 0


def test_search_qa_entries_rejects_wrong_bm25_index_signature_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_calls = 0

    class BadRetriever:
        def index(
            self,
            corpus_tokens: Sequence[Sequence[str]],
            required_extra: str,
            *,
            show_progress: bool = True,
        ) -> None:
            nonlocal index_calls
            del corpus_tokens, required_extra, show_progress
            index_calls += 1

        def retrieve(
            self,
            corpus_tokens: Sequence[Sequence[str]],
            *,
            k: int,
            show_progress: bool = True,
        ) -> tuple[tuple[int, ...], tuple[float, ...]]:
            del corpus_tokens, k, show_progress
            return ((0,), (1.0,))

    fake_module = ModuleType("bm25s")
    vars(fake_module)["BM25"] = BadRetriever
    monkeypatch.setattr(bm25_runtime, "import_module", _module_importer(fake_module))

    match = search_qa_entries("客户问过拟合是什么意思", _bm25_entries(), top_k=3)

    assert match.status == "unavailable"
    assert match.reason_code == "qa_ranker_invalid"
    assert index_calls == 0


def test_search_qa_entries_rejects_wrong_bm25_retrieve_signature_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_calls = 0
    retrieve_calls = 0

    class BadRetriever:
        def index(
            self,
            corpus_tokens: Sequence[Sequence[str]],
            *,
            show_progress: bool = True,
        ) -> None:
            nonlocal index_calls
            del corpus_tokens, show_progress
            index_calls += 1

        def retrieve(
            self,
            corpus_tokens: Sequence[Sequence[str]],
            required_extra: str,
            *,
            k: int,
            show_progress: bool = True,
        ) -> tuple[tuple[int, ...], tuple[float, ...]]:
            nonlocal retrieve_calls
            del corpus_tokens, required_extra, k, show_progress
            retrieve_calls += 1
            return ((0,), (1.0,))

    fake_module = ModuleType("bm25s")
    vars(fake_module)["BM25"] = BadRetriever
    monkeypatch.setattr(bm25_runtime, "import_module", _module_importer(fake_module))

    match = search_qa_entries("客户问过拟合是什么意思", _bm25_entries(), top_k=3)

    assert match.status == "unavailable"
    assert match.reason_code == "qa_ranker_invalid"
    assert index_calls == 0
    assert retrieve_calls == 0


def test_search_qa_entries_fails_open_for_malformed_bm25_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadRetriever:
        def index(
            self,
            corpus_tokens: Sequence[Sequence[str]],
            *,
            show_progress: bool = True,
        ) -> None:
            del corpus_tokens, show_progress

        def retrieve(
            self,
            corpus_tokens: Sequence[Sequence[str]],
            *,
            k: int,
            show_progress: bool = True,
        ) -> tuple[tuple[int, ...], tuple[float, ...]]:
            del corpus_tokens, k, show_progress
            return ((999,), (1.0,))

    fake_module = ModuleType("bm25s")
    vars(fake_module)["BM25"] = BadRetriever
    monkeypatch.setattr(bm25_runtime, "import_module", _module_importer(fake_module))

    match = search_qa_entries("客户问过拟合是什么意思", _bm25_entries(), top_k=3)

    assert match.status == "unavailable"
    assert match.reason_code == "qa_ranker_invalid"


def test_knowledge_qa_match_prompt_payload_is_bounded() -> None:
    match = KnowledgeQaMatch(
        status="matched",
        candidates=[
            KnowledgeQaCandidate(
                doc_id="faq",
                question="什么是过拟合？",
                score=0.8,
            ),
        ],
    )

    payload = match.to_prompt_dict()
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    candidate = candidates[0]
    assert isinstance(candidate, dict)

    assert payload["status"] == "matched"
    assert candidate["question"] == "什么是过拟合？"
    assert set(candidate) == {"doc_id", "question", "score"}
    assert "answer_preview" not in candidate
    assert "allow_planner_document_context_fallback" in payload
