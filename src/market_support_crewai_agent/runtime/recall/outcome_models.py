from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.runtime.hashing import canonical_json_bytes
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.schemas.base import StrictModel

RecallSourceClassV1 = Literal["approved_static", "document_mcp"]
RecallCandidateReasonV1 = Literal[
    "approved_recall_hit", "approved_recall_hint", "document_qa_candidate"
]
RecallBranchStatusV1 = Literal["not_called", "ok", "partial", "unavailable", "invalid"]
RecallBranchReasonV1 = Literal[
    "disabled",
    "not_needed",
    "no_match",
    "candidate_rejected",
    "source_unavailable",
    "source_invalid",
    "ok",
]
RecallModeV1 = Literal["off", "advisory", "shortcut"]
RecallDecisionV1 = Literal[
    "disabled", "no_match", "advisory_candidates", "shortcut_match", "unavailable"
]
CandidateIdV1 = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,63}$")]
CanonicalIdV1 = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")]


class RecallContractError(ValueError):
    pass


class _FrozenRecallModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class RecallCandidateViewV1(_FrozenRecallModel):
    contract_version: Literal["recall-candidate.v1"] = "recall-candidate.v1"
    candidate_id: CandidateIdV1
    canonical_id: CanonicalIdV1
    source_class: RecallSourceClassV1
    question: str = Field(min_length=1, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: RecallCandidateReasonV1

    @model_validator(mode="after")
    def _validate_contract(self) -> RecallCandidateViewV1:
        if (
            len(canonical_json_bytes(self.model_dump(mode="json", exclude_none=False)))
            > 4_096
        ):
            raise RecallContractError("recall_candidate_serialized_bytes_exceeded")
        return self


class RecallBranchOutcomeV1(_FrozenRecallModel):
    source_class: RecallSourceClassV1
    status: RecallBranchStatusV1
    accepted_count: int = Field(ge=0, le=5)
    rejected_count: int = Field(ge=0, le=5)
    reason_code: RecallBranchReasonV1


class RecallShortcutSummaryV1(_FrozenRecallModel):
    candidate_id: CandidateIdV1
    canonical_id: CanonicalIdV1
    source_id: str = Field(min_length=1, max_length=160)
    selected_manifest_ref: ManifestRefV1
    confidence: float = Field(ge=0.72, le=1.0)
    reply_text_hash: str = Field(pattern=r"^rtx1:[0-9a-f]{64}$")
    evidence_text_hash: str = Field(pattern=r"^evx1:[0-9a-f]{64}$")


class RecallOutcomeV1(_FrozenRecallModel):
    contract_version: Literal["recall-outcome.v1"] = "recall-outcome.v1"
    mode: RecallModeV1
    decision: RecallDecisionV1
    candidates: tuple[RecallCandidateViewV1, ...] = Field(max_length=5)
    branch_outcomes: tuple[RecallBranchOutcomeV1, RecallBranchOutcomeV1]
    shortcut_summary: RecallShortcutSummaryV1 | None
    trace_hash: str = Field(pattern=r"^rch1:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_outcome(self) -> RecallOutcomeV1:
        source_order = tuple(branch.source_class for branch in self.branch_outcomes)
        if source_order != ("approved_static", "document_mcp"):
            raise RecallContractError("recall_branch_order_mismatch")
        candidate_payload = (
            "["
            + ",".join(
                item.model_dump_json(exclude_none=False) for item in self.candidates
            )
            + "]"
        )
        if len(candidate_payload.encode("utf-8")) > 5_120:
            raise RecallContractError("recall_candidate_list_serialized_bytes_exceeded")
        return self
