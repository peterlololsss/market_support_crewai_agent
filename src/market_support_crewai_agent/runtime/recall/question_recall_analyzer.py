from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    APPROVED_KNOWLEDGE,
)
from market_support_crewai_agent.runtime.recall.bm25_runtime import (
    Bm25QueryV1,
    normalize_bm25_score,
    rank_bm25,
)
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    DocumentQaEntry,
)
from market_support_crewai_agent.runtime.recall.question_models import (
    MatchDraft,
    QuestionRecallCandidate,
    QuestionRecallDecision,
    QuestionRecallEntry,
    QuestionRecallMatch,
    make_question_recall_match,
)
from market_support_crewai_agent.runtime.recall.question_recall_text import (
    NormalizationProfile,
    answer_claims_adapter_execution,
    covered_by_ordered_terms,
    safe_metadata_label,
)
from market_support_crewai_agent.runtime.recall.question_recall_text import (
    tokens as _tokens,
)

_HIT_THRESHOLD, _HINT_THRESHOLD, _HIT_COVERAGE_THRESHOLD = 0.72, 0.55, 0.8


@dataclass(frozen=True, slots=True)
class CandidateInputs:
    entries: Sequence[QuestionRecallEntry]
    indices: Sequence[int]
    scores: Sequence[float]
    query: str
    profile: NormalizationProfile


def approved_static_entries() -> tuple[QuestionRecallEntry, ...]:
    entries: list[QuestionRecallEntry] = []
    for entry in APPROVED_KNOWLEDGE:
        surfaces = _unique(
            (entry.title, entry.semantic_purpose, *entry.user_request_examples)
        )
        for question in entry.user_request_examples or (entry.title,):
            entries.append(
                QuestionRecallEntry(
                    entry_id=entry.entry_id,
                    canonical_id=f"approved_static_knowledge.{entry.entry_id}",
                    doc_id=entry.entry_id,
                    title=entry.title,
                    question=question,
                    answer=entry.approved_answer,
                    source_type="approved_static_knowledge",
                    surfaces=surfaces,
                )
            )
    return tuple(entries)


def document_entry(entry: DocumentQaEntry) -> QuestionRecallEntry:
    doc_id = safe_metadata_label(entry.doc_id)
    title = safe_metadata_label(entry.doc_title)
    return QuestionRecallEntry(
        entry_id=entry.entry_id,
        canonical_id=f"document_mcp.{entry.entry_id}",
        doc_id=doc_id or "document",
        title=title or "document",
        question=entry.question,
        answer=entry.answer,
        source_type="document_mcp",
        surfaces=_unique((title, entry.question)),
    )


def search_question_recall(
    query: str,
    entries: Sequence[QuestionRecallEntry],
    profile: NormalizationProfile | None = None,
) -> QuestionRecallMatch:
    profile = profile or NormalizationProfile()
    query_tokens = _tokens(query, profile)
    if not query_tokens:
        return _fail_open("empty_query", profile)
    if not entries:
        return _fail_open("question_recall_corpus_empty", profile)

    ranked_documents = rank_bm25(
        Bm25QueryV1(
            corpus=[_entry_tokens(entry, profile) for entry in entries],
            query_tokens=query_tokens,
            top_k=min(5, len(entries)),
        )
    )
    if ranked_documents is None:
        return _fail_open("question_recall_ranker_invalid", profile)
    candidates = _candidates(
        CandidateInputs(
            entries,
            ranked_documents.indices,
            ranked_documents.scores,
            query,
            profile,
        )
    )
    if not candidates:
        return _fail_open("no_candidate_overlap", profile)

    top = candidates[0]
    decision = _decision(top)
    if decision == "fail_open":
        return _fail_open("no_ordered_surface_match", profile)
    entry = _entry_by_id(entries, top.entry_id)
    return make_question_recall_match(
        MatchDraft(
            status="matched" if decision == "recall_hit" else "candidates",
            decision=decision,
            reason_code=decision,
            normalizer_version=profile.version,
            lexicon_version=profile.lexicon_version,
            candidates=tuple(candidates),
            entry=entry,
        )
    )


def fail_open_match(
    reason_code: str, profile: NormalizationProfile
) -> QuestionRecallMatch:
    return _fail_open(reason_code, profile)


def unavailable_match(
    reason_code: str, profile: NormalizationProfile
) -> QuestionRecallMatch:
    return make_question_recall_match(
        MatchDraft(
            status="unavailable",
            decision="fail_open",
            reason_code=reason_code,
            normalizer_version=profile.version,
            lexicon_version=profile.lexicon_version,
        )
    )


def disabled_match(
    reason_code: str, profile: NormalizationProfile
) -> QuestionRecallMatch:
    return make_question_recall_match(
        MatchDraft(
            status="disabled",
            decision="fail_open",
            reason_code=reason_code,
            normalizer_version=profile.version,
            lexicon_version=profile.lexicon_version,
        )
    )


def _fail_open(reason_code: str, profile: NormalizationProfile) -> QuestionRecallMatch:
    return make_question_recall_match(
        MatchDraft(
            status="no_match",
            decision="fail_open",
            reason_code=reason_code,
            normalizer_version=profile.version,
            lexicon_version=profile.lexicon_version,
        )
    )


def _entry_tokens(
    entry: QuestionRecallEntry, profile: NormalizationProfile
) -> list[str]:
    surface_tokens = [profile.normalize(surface) for surface in entry.surfaces]
    return [
        entry.canonical_id,
        *_tokens(" ".join((entry.title, entry.question, *entry.surfaces)), profile),
        *surface_tokens,
    ]


def _candidates(inputs: CandidateInputs) -> list[QuestionRecallCandidate]:
    output: list[QuestionRecallCandidate] = []
    query_token_set = frozenset(_tokens(inputs.query, inputs.profile))
    for raw_index, raw_score in zip(inputs.indices, inputs.scores, strict=False):
        score = normalize_bm25_score(float(raw_score))
        if score <= 0:
            continue
        entry = inputs.entries[int(raw_index)]
        entry_tokens = frozenset(_entry_tokens(entry, inputs.profile))
        coverage = _coverage(query_token_set, entry_tokens)
        matched = _matched_surfaces(inputs.query, entry, inputs.profile)
        confidence = min(
            1.0,
            (score * 0.55) + (coverage * 0.25) + (0.25 if matched else 0.0),
        )
        if confidence < _HINT_THRESHOLD:
            continue
        output.append(
            QuestionRecallCandidate(
                entry_id=entry.entry_id,
                canonical_id=entry.canonical_id,
                doc_id=entry.doc_id,
                question=entry.question,
                source_type=entry.source_type,
                score=round(score, 4),
                confidence=round(confidence, 4),
                coverage=round(coverage, 4),
                matched_entities=matched,
                answer_available=(
                    bool(entry.answer.strip())
                    and not answer_claims_adapter_execution(
                        entry.answer, inputs.profile
                    )
                ),
            )
        )
    return output


def _decision(candidate: QuestionRecallCandidate) -> QuestionRecallDecision:
    if (
        not candidate.matched_entities
        or not candidate.answer_available
        or candidate.coverage < _HIT_COVERAGE_THRESHOLD
    ):
        return "fail_open"
    if candidate.source_type != "approved_static_knowledge":
        return "recall_hint"
    if candidate.confidence >= _HIT_THRESHOLD and candidate.answer_available:
        return "recall_hit"
    return "recall_hint"


def _entry_by_id(
    entries: Sequence[QuestionRecallEntry],
    entry_id: str,
) -> QuestionRecallEntry | None:
    for entry in entries:
        if entry.entry_id == entry_id:
            return entry
    return None


def _matched_surfaces(
    query: str,
    entry: QuestionRecallEntry,
    profile: NormalizationProfile,
) -> tuple[str, ...]:
    phrase_terms = tuple(_tokens(query, profile))
    matched = [
        profile.normalize(surface)
        for surface in entry.surfaces
        if covered_by_ordered_terms(tuple(_tokens(surface, profile)), phrase_terms)
    ]
    return tuple(_unique(matched))


def _coverage(query_tokens: frozenset[str], entry_tokens: frozenset[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & entry_tokens) / len(query_tokens)


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return tuple(output)
