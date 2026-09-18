from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, JsonValue

from market_support_crewai_agent.runtime.integrations.document_mcp.sanitizer import (
    sanitize_document_text_for_evidence,
)
from market_support_crewai_agent.runtime.recall.bm25_runtime import (
    Bm25QueryV1,
    normalize_bm25_score,
    rank_bm25,
)
from market_support_crewai_agent.runtime.recall.question_recall_text import (
    safe_metadata_label,
)
from market_support_crewai_agent.schemas.base import StrictModel

MAX_QA_CORPUS_DOCUMENTS = 50
_MATCH_THRESHOLD = 0.2
_QUESTION_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:Q|q|问|问题)\s*[:：]\s*(?P<text>.+?)\s*$"
)
_ANSWER_RE: re.Pattern[str] = re.compile(
    r"^\s*(?:A|a|答|答案)\s*[:：]\s*(?P<text>.*)\s*$"
)
_QUESTION_VARIANT_RE: re.Pattern[str] = re.compile(r"\s*[／/]\s*")
_TERM_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_COMMON_TERMS = frozenset(
    (
        "的",
        "了",
        "是",
        "什",
        "么",
        "吗",
        "呢",
        "和",
        "与",
        "及",
        "请",
        "问",
        "这",
        "那",
        "个",
        "一",
        "下",
        "发",
        "有",
        "在",
        "里",
        "怎",
        "样",
        "如",
        "何",
        "多",
        "少",
        "客",
        "户",
    )
)
_RANKER_INVALID_REASON = "qa_ranker_invalid"


class DocumentQaEntry(StrictModel):
    entry_id: str = Field(min_length=1, max_length=80)
    doc_id: str = Field(min_length=1, max_length=160)
    doc_title: str = Field(default="", max_length=240)
    question: str = Field(min_length=1, max_length=400)
    answer: str = Field(default="", max_length=4000)
    source_hash: str = Field(min_length=12, max_length=64)


class KnowledgeQaCandidate(StrictModel):
    doc_id: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=1, max_length=400)
    score: float = Field(ge=0.0, le=1.0)


class KnowledgeQaMatch(StrictModel):
    contract_version: Literal["knowledge-qa-match"] = "knowledge-qa-match"
    status: Literal["disabled", "unavailable", "no_match", "candidates", "matched"]
    candidates: list[KnowledgeQaCandidate] = Field(default_factory=list, max_length=8)
    reason_code: str = Field(default="", max_length=120)
    allow_planner_document_context_fallback: bool = True
    planner_guidance: str = (
        "QA recall is an advisory candidate list, not final evidence and not proof of "
        "absence. If the user asks a knowledge question, the planner may still select "
        "document_context so deterministic evidence can fetch full approved text."
    )

    def to_prompt_dict(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_none=True)


class QaCorpusShapeError(TypeError):
    pass


def parse_qa_entries(
    documents: Sequence[JsonValue], *, max_chars_per_document: int | None = None
) -> tuple[DocumentQaEntry, ...]:
    entries: list[DocumentQaEntry] = []
    for document in documents:
        if not isinstance(document, Mapping):
            raise QaCorpusShapeError
        entries.extend(
            _parse_document_entries(
                document, max_chars_per_document=max_chars_per_document
            )
        )
    return tuple(entries)


def search_qa_entries(
    query: str, entries: Sequence[DocumentQaEntry], *, top_k: int = 5
) -> KnowledgeQaMatch:
    query_tokens = _tokens(query)
    if not query_tokens:
        return KnowledgeQaMatch(status="no_match", reason_code="empty_query")
    if not entries:
        return KnowledgeQaMatch(status="no_match", reason_code="qa_entry_corpus_empty")

    ranked_documents = rank_bm25(
        Bm25QueryV1(
            corpus=_entry_corpus(entries),
            query_tokens=query_tokens,
            top_k=min(max(1, top_k), len(entries)),
        )
    )
    if ranked_documents is None:
        return KnowledgeQaMatch(
            status="unavailable", reason_code=_RANKER_INVALID_REASON
        )

    scored = _ranked_entries(entries, ranked_documents.indices, ranked_documents.scores)
    if not scored:
        return KnowledgeQaMatch(
            status="no_match", reason_code="no_candidate_question_overlap"
        )

    candidates = [_candidate(entry, score) for entry, score in scored]
    status: Literal["candidates", "matched"] = (
        "matched" if candidates[0].score >= _MATCH_THRESHOLD else "candidates"
    )
    return KnowledgeQaMatch(status=status, candidates=candidates)


def _parse_document_entries(
    document: Mapping[str, JsonValue], *, max_chars_per_document: int | None
) -> list[DocumentQaEntry]:
    doc_id = safe_metadata_label(str(document.get("id") or "").strip())
    if not doc_id:
        return []
    raw_title = str(document.get("title") or document.get("name") or doc_id).strip()
    title = safe_metadata_label(raw_title) or doc_id
    content = str(document.get("content") or document.get("text") or "")
    if max_chars_per_document is not None:
        content = content[:max_chars_per_document]
    sanitized = sanitize_document_text_for_evidence(content).text
    entries: list[DocumentQaEntry] = []
    current_question = ""
    answer_lines: list[str] = []
    for raw_line in sanitized.splitlines():
        line = raw_line.strip()
        question_match = _QUESTION_RE.match(line)
        if question_match is not None:
            _append_entry(entries, doc_id, title, current_question, answer_lines)
            current_question = question_match.group("text").strip()
            answer_lines = []
            continue
        answer_match = _ANSWER_RE.match(line)
        if answer_match is not None:
            answer_lines = [answer_match.group("text").strip()]
            continue
        if current_question and line:
            answer_lines.append(line)
    _append_entry(entries, doc_id, title, current_question, answer_lines)
    return entries


def _append_entry(
    entries: list[DocumentQaEntry],
    doc_id: str,
    title: str,
    question: str,
    answer_lines: Sequence[str],
) -> None:
    answer = "\n".join(line for line in answer_lines if line.strip()).strip()
    for normalized_question in _question_variants(question):
        source = f"{doc_id}\n{normalized_question}\n{answer}"
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        entries.append(
            DocumentQaEntry(
                entry_id=f"qa:{source_hash[:16]}",
                doc_id=doc_id,
                doc_title=title,
                question=normalized_question,
                answer=answer[:4000],
                source_hash=source_hash,
            )
        )


def _question_variants(question: str) -> tuple[str, ...]:
    normalized = question.strip()
    if not normalized:
        return ()
    variants = tuple(
        _compact_question(part)
        for part in _QUESTION_VARIANT_RE.split(normalized)
        if part.strip()
    )
    return variants or (_compact_question(normalized),)


def _compact_question(question: str) -> str:
    return question.strip()[:400].strip()


def _entry_corpus(entries: Sequence[DocumentQaEntry]) -> list[list[str]]:
    return [_tokens(f"{entry.question} {entry.doc_title}") for entry in entries]


def _ranked_entries(
    entries: Sequence[DocumentQaEntry],
    document_indices: Sequence[int],
    raw_scores: Sequence[float],
) -> list[tuple[DocumentQaEntry, float]]:
    scored: list[tuple[DocumentQaEntry, float]] = []
    for raw_index, raw_score in zip(document_indices, raw_scores, strict=False):
        score = float(raw_score)
        if score <= 0:
            continue
        index = int(raw_index)
        scored.append((entries[index], normalize_bm25_score(score)))
    return scored


def _candidate(entry: DocumentQaEntry, score: float) -> KnowledgeQaCandidate:
    return KnowledgeQaCandidate(
        doc_id=entry.doc_id,
        question=entry.question,
        score=round(score, 4),
    )


def _tokens(text: str) -> list[str]:
    normalized = text.lower()
    return [
        term
        for match in _TERM_RE.finditer(normalized)
        for term in (match.group(0),)
        if term not in _COMMON_TERMS
    ]
