from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, JsonValue

from market_support_crewai_agent.schemas.base import StrictModel

QuestionRecallDecision = Literal["recall_hit", "recall_hint", "fail_open"]
QuestionRecallStatus = Literal[
    "disabled", "unavailable", "no_match", "candidates", "matched"
]
QuestionRecallSource = Literal["document_mcp", "approved_static_knowledge"]


@dataclass(frozen=True, slots=True)
class QuestionRecallEntry:
    entry_id: str
    canonical_id: str
    doc_id: str
    title: str
    question: str
    answer: str
    source_type: QuestionRecallSource
    surfaces: tuple[str, ...]

    @property
    def evidence_text(self) -> str:
        return f"Q：{self.question}\nA：{self.answer}".strip()


class QuestionRecallCandidate(StrictModel):
    entry_id: str = Field(min_length=1, max_length=120)
    canonical_id: str = Field(min_length=1, max_length=160)
    doc_id: str = Field(min_length=1, max_length=160)
    question: str = Field(min_length=1, max_length=400)
    source_type: QuestionRecallSource
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    matched_entities: tuple[str, ...] = ()
    answer_available: bool = False


@dataclass(frozen=True, slots=True)
class MatchDraft:
    status: QuestionRecallStatus
    decision: QuestionRecallDecision
    reason_code: str
    normalizer_version: str
    lexicon_version: str
    candidates: tuple[QuestionRecallCandidate, ...] = ()
    entry: QuestionRecallEntry | None = None


class QuestionRecallMatch(StrictModel):
    contract_version: Literal["question-recall-match"] = "question-recall-match"
    status: QuestionRecallStatus
    decision: QuestionRecallDecision
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    candidates: list[QuestionRecallCandidate] = Field(
        default_factory=list, max_length=8
    )
    reason_code: str = Field(default="", max_length=120)
    reply_text: str = Field(default="", max_length=4000)
    evidence_text: str = Field(default="", max_length=5000)
    source_id: str = Field(default="", max_length=160)
    source_type: QuestionRecallSource | None = None
    trace: dict[str, JsonValue] = Field(default_factory=dict)
    allow_planner_document_context_fallback: bool = True
    planner_guidance: str = (
        "Question recall is fail-open evidence. High-confidence approved Q&A "
        "hits may answer through deterministic validators; hints and misses go "
        "through the planner/document path."
    )

    def to_prompt_dict(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude_none=True)


def make_question_recall_match(draft: MatchDraft) -> QuestionRecallMatch:
    candidates = list(draft.candidates)
    top_confidence = candidates[0].confidence if candidates else 0.0
    entry = draft.entry if draft.decision == "recall_hit" else None
    return QuestionRecallMatch(
        status=draft.status,
        decision=draft.decision,
        confidence=top_confidence,
        candidates=candidates,
        reason_code=draft.reason_code,
        reply_text=entry.answer if entry is not None else "",
        evidence_text=entry.evidence_text if entry is not None else "",
        source_id=entry.entry_id if entry is not None else "",
        source_type=entry.source_type if entry is not None else None,
        trace={
            "normalizer_version": draft.normalizer_version,
            "lexicon_version": draft.lexicon_version,
            "hit_threshold": 0.72,
        },
    )


def candidate_ids(candidates: Sequence[QuestionRecallCandidate]) -> tuple[str, ...]:
    return tuple(candidate.entry_id for candidate in candidates)
