from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import ValidationError

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaMatch,
)
from market_support_crewai_agent.runtime.recall.outcome_models import (
    RecallBranchOutcomeV1,
    RecallCandidateReasonV1,
    RecallCandidateViewV1,
    RecallDecisionV1,
    RecallModeV1,
    RecallOutcomeV1,
    RecallShortcutSummaryV1,
)
from market_support_crewai_agent.runtime.recall.question_models import (
    QuestionRecallMatch,
)
from market_support_crewai_agent.runtime.recall.question_recall_analyzer import (
    approved_static_entries,
    disabled_match,
    fail_open_match,
    search_question_recall,
    unavailable_match,
)
from market_support_crewai_agent.runtime.recall.question_recall_text import (
    NormalizationProfile,
    recall_shortcut_blocked,
)
from market_support_crewai_agent.runtime.recall.turn_state import (
    ApprovedStaticShortcutPayloadV1,
    recall_outcome_hash,
)
from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings

_DOC_CAPABILITY: Final = "query_internal_company_info"


@dataclass(frozen=True, slots=True)
class ApprovedStaticRecallCollectionV1:
    match: QuestionRecallMatch
    shortcut_payload: ApprovedStaticShortcutPayloadV1 | None = None


@dataclass(frozen=True, slots=True)
class RecallBranchCollectionV1:
    source_class: Literal["approved_static", "document_mcp"]
    candidates: tuple[RecallCandidateViewV1, ...] = ()
    rejected_count: int = 0
    unavailable: bool = False

    def outcome(self) -> RecallBranchOutcomeV1:
        accepted_count = len(self.candidates)
        if self.unavailable:
            return RecallBranchOutcomeV1(
                source_class=self.source_class,
                status="unavailable",
                accepted_count=0,
                rejected_count=0,
                reason_code="source_unavailable",
            )
        if accepted_count and self.rejected_count:
            status, reason = "partial", "candidate_rejected"
        elif self.rejected_count:
            status, reason = "invalid", "source_invalid"
        elif accepted_count:
            status, reason = "ok", "ok"
        else:
            status, reason = "ok", "no_match"
        return RecallBranchOutcomeV1(
            source_class=self.source_class,
            status=status,
            accepted_count=accepted_count,
            rejected_count=self.rejected_count,
            reason_code=reason,
        )


class QuestionRecallService:
    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.profile: NormalizationProfile = NormalizationProfile()

    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> ApprovedStaticRecallCollectionV1:
        if policy.recall_mode == "off":
            match = disabled_match("recall_mode_off", self.profile)
        else:
            match = self._collect_enabled(request, policy)
        return ApprovedStaticRecallCollectionV1(match=match)

    def _collect_enabled(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> QuestionRecallMatch:
        if (
            not policy.internal_company_knowledge_enabled
            or _DOC_CAPABILITY not in policy.allowed_read_capabilities
        ):
            return disabled_match("document_read_policy_forbidden", self.profile)
        if recall_shortcut_blocked(request.message, self.profile):
            return fail_open_match("compliance_risky_message", self.profile)

        entries = approved_static_entries()
        if not entries:
            return unavailable_match("question_recall_corpus_empty", self.profile)
        return search_question_recall(request.message, entries, self.profile)


def approved_static_branch(match: QuestionRecallMatch) -> RecallBranchCollectionV1:
    if match.status == "unavailable":
        return RecallBranchCollectionV1("approved_static", unavailable=True)
    reason_code: RecallCandidateReasonV1 = (
        "approved_recall_hit"
        if match.decision == "recall_hit"
        else "approved_recall_hint"
    )
    candidates: list[RecallCandidateViewV1] = []
    rejected_count = 0
    for candidate in match.candidates:
        if candidate.source_type == "document_mcp":
            rejected_count += 1
            continue
        candidate_reason = (
            reason_code if candidate.answer_available else "approved_recall_hint"
        )
        try:
            candidates.append(
                RecallCandidateViewV1(
                    candidate_id=candidate.entry_id,
                    canonical_id=candidate.canonical_id,
                    source_class="approved_static",
                    question=candidate.question,
                    confidence=candidate.confidence,
                    reason_code=candidate_reason,
                )
            )
        except ValidationError:
            rejected_count += 1
    return RecallBranchCollectionV1(
        "approved_static",
        tuple(candidates),
        rejected_count,
    )


def document_recall_branch(match: KnowledgeQaMatch) -> RecallBranchCollectionV1:
    if match.status == "unavailable":
        return RecallBranchCollectionV1("document_mcp", unavailable=True)
    candidates: list[RecallCandidateViewV1] = []
    rejected_count = 0
    for candidate in match.candidates:
        digest = hashlib.sha256(
            b"document-qa-candidate.v1\0" + candidate.question.encode("utf-8")
        ).hexdigest()
        try:
            candidates.append(
                RecallCandidateViewV1(
                    candidate_id=f"document:{digest[:16]}",
                    canonical_id=f"document_qa:{digest}",
                    source_class="document_mcp",
                    question=candidate.question,
                    confidence=candidate.score,
                    reason_code="document_qa_candidate",
                )
            )
        except ValidationError:
            rejected_count += 1
    return RecallBranchCollectionV1(
        "document_mcp",
        tuple(candidates),
        rejected_count,
    )


def merge_recall_candidates(
    static: tuple[RecallCandidateViewV1, ...],
    document: tuple[RecallCandidateViewV1, ...],
) -> tuple[RecallCandidateViewV1, ...]:
    merged: dict[tuple[str, str], RecallCandidateViewV1] = {}
    for candidate in (*static, *document):
        key = (candidate.source_class, candidate.canonical_id)
        current = merged.get(key)
        if current is None or candidate.confidence > current.confidence:
            merged[key] = candidate
    source_order = {"approved_static": 0, "document_mcp": 1}
    return tuple(
        sorted(
            merged.values(),
            key=lambda item: (
                -item.confidence,
                source_order[item.source_class],
                item.canonical_id,
                item.candidate_id,
            ),
        )[:5]
    )


def make_recall_outcome(
    mode: RecallModeV1,
    decision: RecallDecisionV1,
    candidates: tuple[RecallCandidateViewV1, ...],
    branches: tuple[RecallBranchOutcomeV1, RecallBranchOutcomeV1],
    summary: RecallShortcutSummaryV1 | None = None,
) -> RecallOutcomeV1:
    draft = RecallOutcomeV1(
        mode=mode,
        decision=decision,
        candidates=candidates,
        branch_outcomes=branches,
        shortcut_summary=summary,
        trace_hash="rch1:" + "0" * 64,
    )
    return draft.model_copy(update={"trace_hash": recall_outcome_hash(draft)})


__all__ = [
    "ApprovedStaticRecallCollectionV1",
    "QuestionRecallService",
    "RecallBranchCollectionV1",
    "approved_static_branch",
    "document_recall_branch",
    "make_recall_outcome",
    "merge_recall_candidates",
]
