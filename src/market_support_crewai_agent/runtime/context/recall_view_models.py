from __future__ import annotations

from typing import ClassVar, Literal, assert_never

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.recall.outcome_models import (
    CandidateIdV1,
    CanonicalIdV1,
    RecallBranchOutcomeV1,
    RecallCandidateViewV1,
    RecallDecisionV1,
    RecallModeV1,
)
from market_support_crewai_agent.schemas.base import StrictModel


class _FrozenRecallView(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class RecallPlannerShortcutSummaryViewV1(_FrozenRecallView):
    candidate_id: CandidateIdV1
    canonical_id: CanonicalIdV1
    selected_manifest_ref: ManifestRefV1
    confidence: float = Field(ge=0.72, le=1.0)
    reply_text_hash: str = Field(pattern=r"^rtx1:[0-9a-f]{64}$")
    evidence_text_hash: str = Field(pattern=r"^evx1:[0-9a-f]{64}$")


class RecallPlannerViewV1(_FrozenRecallView):
    contract_version: Literal["recall-planner-view.v1"] = "recall-planner-view.v1"
    mode: RecallModeV1
    decision: RecallDecisionV1
    candidates: tuple[RecallCandidateViewV1, ...] = Field(max_length=5)
    branch_outcomes: tuple[RecallBranchOutcomeV1, RecallBranchOutcomeV1]
    shortcut_summary: RecallPlannerShortcutSummaryViewV1 | None
    trace_hash: str = Field(pattern=r"^rch1:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_recall_projection(self) -> RecallPlannerViewV1:
        source_order = tuple(branch.source_class for branch in self.branch_outcomes)
        if source_order != ("approved_static", "document_mcp"):
            raise ContextViewInvariantError("recall_planner_branch_order_mismatch")
        match self.decision:
            case "shortcut_match":
                if self.mode != "shortcut" or self.shortcut_summary is None:
                    raise ContextViewInvariantError(
                        "recall_planner_shortcut_summary_required"
                    )
                summary_identity = (
                    self.shortcut_summary.candidate_id,
                    self.shortcut_summary.canonical_id,
                )
                candidate_identities = {
                    (candidate.candidate_id, candidate.canonical_id)
                    for candidate in self.candidates
                }
                if summary_identity not in candidate_identities:
                    raise ContextViewInvariantError(
                        "recall_planner_shortcut_candidate_mismatch"
                    )
            case "disabled":
                if self.mode != "off" or self.candidates:
                    raise ContextViewInvariantError(
                        "recall_planner_disabled_projection_mismatch"
                    )
                if self.shortcut_summary is not None:
                    raise ContextViewInvariantError(
                        "recall_planner_unexpected_shortcut_summary"
                    )
            case "advisory_candidates":
                if self.mode == "off" or not self.candidates:
                    raise ContextViewInvariantError(
                        "recall_planner_advisory_projection_mismatch"
                    )
                if self.shortcut_summary is not None:
                    raise ContextViewInvariantError(
                        "recall_planner_unexpected_shortcut_summary"
                    )
            case "no_match" | "unavailable":
                if self.mode == "off" or self.candidates:
                    raise ContextViewInvariantError(
                        "recall_planner_empty_projection_mismatch"
                    )
                if self.shortcut_summary is not None:
                    raise ContextViewInvariantError(
                        "recall_planner_unexpected_shortcut_summary"
                    )
            case unreachable:
                assert_never(unreachable)
        return self
