from __future__ import annotations

from market_support_crewai_agent.runtime.recall.question_models import (
    MatchDraft,
    QuestionRecallCandidate,
    QuestionRecallEntry,
    QuestionRecallSource,
    candidate_ids,
    make_question_recall_match,
)


def _candidate(*, confidence: float = 0.91) -> QuestionRecallCandidate:
    return QuestionRecallCandidate(
        entry_id="company_contact",
        canonical_id="approved_static_knowledge.company_contact",
        doc_id="company_contact",
        question="公司网址是什么？",
        source_type="approved_static_knowledge",
        score=0.93,
        confidence=confidence,
        coverage=1.0,
        matched_entities=("company",),
        answer_available=True,
    )


def _entry(
    *, source_type: QuestionRecallSource = "approved_static_knowledge"
) -> QuestionRecallEntry:
    return QuestionRecallEntry(
        entry_id="company_contact",
        canonical_id=f"{source_type}.company_contact",
        doc_id="company_contact",
        title="公司联系信息",
        question="公司网址是什么？",
        answer="公司网址是 https://example.com。",
        source_type=source_type,
        surfaces=("公司网址",),
    )


def test_match_factory_preserves_current_hit_projection() -> None:
    # Given: the current approved-static recall draft.
    candidate = _candidate()
    draft = MatchDraft(
        status="matched",
        decision="recall_hit",
        reason_code="approved_recall_hit",
        normalizer_version="normalizer.v1",
        lexicon_version="lexicon.v1",
        candidates=(candidate,),
        entry=_entry(),
    )

    # When: the legacy public factory materializes the match.
    match = make_question_recall_match(draft)

    # Then: confidence, deterministic bodies, source, and trace retain their values.
    assert match.confidence == candidate.confidence
    assert match.reply_text == "公司网址是 https://example.com。"
    assert match.evidence_text == (
        "Q：公司网址是什么？\nA：公司网址是 https://example.com。"
    )
    assert match.source_id == "company_contact"
    assert match.source_type == "approved_static_knowledge"
    assert match.trace == {
        "normalizer_version": "normalizer.v1",
        "lexicon_version": "lexicon.v1",
        "hit_threshold": 0.72,
    }


def test_match_factory_omits_entry_bodies_for_non_hit() -> None:
    # Given: a hint draft that still carries an analyzer-internal entry.
    draft = MatchDraft(
        status="candidates",
        decision="recall_hint",
        reason_code="approved_recall_hint",
        normalizer_version="normalizer.v1",
        lexicon_version="lexicon.v1",
        candidates=(_candidate(confidence=0.61),),
        entry=_entry(),
    )

    # When: the match is materialized.
    match = make_question_recall_match(draft)

    # Then: non-hit results do not project deterministic answer/evidence bodies.
    assert match.reply_text == ""
    assert match.evidence_text == ""
    assert match.source_id == ""
    assert match.source_type is None


def test_candidate_ids_preserve_input_order_and_duplicates() -> None:
    # Given: candidates whose IDs intentionally repeat.
    first = _candidate(confidence=0.91)
    second = first.model_copy(update={"confidence": 0.72})

    # When: current candidate IDs are projected.
    ids = candidate_ids((first, second))

    # Then: projection neither sorts nor deduplicates the sequence.
    assert ids == ("company_contact", "company_contact")
