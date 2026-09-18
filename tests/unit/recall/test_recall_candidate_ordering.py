from market_support_crewai_agent.runtime.recall.outcome_models import (
    RecallCandidateViewV1,
)
from market_support_crewai_agent.runtime.recall.service import merge_recall_candidates


def test_candidate_merge_preserves_score_and_tie_order() -> None:
    # Given: duplicate static rows and an equal-confidence document candidate.
    static = (
        RecallCandidateViewV1(
            candidate_id="static:z",
            canonical_id="canonical:z",
            source_class="approved_static",
            question="静态问题",
            confidence=0.8,
            reason_code="approved_recall_hint",
        ),
        RecallCandidateViewV1(
            candidate_id="static:z-duplicate",
            canonical_id="canonical:z",
            source_class="approved_static",
            question="静态问题",
            confidence=0.7,
            reason_code="approved_recall_hint",
        ),
    )
    document = (
        RecallCandidateViewV1(
            candidate_id="document:a",
            canonical_id="canonical:a",
            source_class="document_mcp",
            question="文档问题",
            confidence=0.8,
            reason_code="document_qa_candidate",
        ),
    )

    # When: provider-neutral candidates cross the deterministic merge boundary.
    merged = merge_recall_candidates(static, document)

    # Then: higher duplicate score wins and equal scores retain source ordering.
    assert tuple(
        (candidate.candidate_id, candidate.confidence) for candidate in merged
    ) == (("static:z", 0.8), ("document:a", 0.8))
