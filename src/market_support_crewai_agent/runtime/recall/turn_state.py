from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.runtime.hashing import sha256_frame
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.recall.outcome_models import (
    CandidateIdV1,
    CanonicalIdV1,
    RecallBranchOutcomeV1,
    RecallContractError,
    RecallOutcomeV1,
)
from market_support_crewai_agent.schemas.base import StrictModel


class ApprovedStaticShortcutPayloadV1(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    candidate_id: CandidateIdV1
    canonical_id: CanonicalIdV1
    selected_manifest_ref: ManifestRefV1
    fact_type: Literal["document_context"] = "document_context"
    selected_media_asset_ids: tuple[str, ...] = Field(max_length=2)
    confidence: float = Field(ge=0.72, le=1.0)
    question: str = Field(min_length=1, max_length=400)
    reply_text: str = Field(min_length=1, max_length=4_000)
    evidence_text: str = Field(min_length=1, max_length=5_000)
    source_id: str = Field(min_length=1, max_length=160)
    reply_text_hash: str = Field(pattern=r"^rtx1:[0-9a-f]{64}$")
    evidence_text_hash: str = Field(pattern=r"^evx1:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_content_hashes(self) -> ApprovedStaticShortcutPayloadV1:
        if self.reply_text_hash != recall_reply_text_hash(self.reply_text):
            raise RecallContractError("recall_reply_text_hash_mismatch")
        if self.evidence_text_hash != recall_evidence_text_hash(self.evidence_text):
            raise RecallContractError("recall_evidence_text_hash_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class RecallTurnStateV1:
    outcome: RecallOutcomeV1
    shortcut_payload: ApprovedStaticShortcutPayloadV1 | None

    def __post_init__(self) -> None:
        if self.outcome.decision == "shortcut_match":
            summary = self.outcome.shortcut_summary
            payload = self.shortcut_payload
            if (
                summary is None
                or payload is None
                or (
                    summary.candidate_id,
                    summary.canonical_id,
                    summary.source_id,
                    summary.selected_manifest_ref,
                    summary.confidence,
                    summary.reply_text_hash,
                    summary.evidence_text_hash,
                )
                != (
                    payload.candidate_id,
                    payload.canonical_id,
                    payload.source_id,
                    payload.selected_manifest_ref,
                    payload.confidence,
                    payload.reply_text_hash,
                    payload.evidence_text_hash,
                )
            ):
                raise RecallContractError("recall_shortcut_payload_mismatch")
        elif self.shortcut_payload is not None:
            raise RecallContractError("recall_unexpected_shortcut_payload")


@dataclass(frozen=True, slots=True)
class PreplannerRecallTransitionV1:
    turn_state: RecallTurnStateV1
    shortcut_plan: ExecutionPlanV2 | None


def disabled_recall_transition() -> PreplannerRecallTransitionV1:
    branches = (
        RecallBranchOutcomeV1(
            source_class="approved_static",
            status="not_called",
            accepted_count=0,
            rejected_count=0,
            reason_code="disabled",
        ),
        RecallBranchOutcomeV1(
            source_class="document_mcp",
            status="not_called",
            accepted_count=0,
            rejected_count=0,
            reason_code="disabled",
        ),
    )
    draft = RecallOutcomeV1(
        mode="off",
        decision="disabled",
        candidates=(),
        branch_outcomes=branches,
        shortcut_summary=None,
        trace_hash="rch1:" + "0" * 64,
    )
    outcome = draft.model_copy(update={"trace_hash": recall_outcome_hash(draft)})
    return PreplannerRecallTransitionV1(
        turn_state=RecallTurnStateV1(outcome=outcome, shortcut_payload=None),
        shortcut_plan=None,
    )


def recall_reply_text_hash(text: str) -> str:
    return (
        "rtx1:"
        + hashlib.sha256(b"recall-reply-text.v1\0" + text.encode("utf-8")).hexdigest()
    )


def recall_evidence_text_hash(text: str) -> str:
    return (
        "evx1:"
        + hashlib.sha256(
            b"recall-evidence-text.v1\0" + text.encode("utf-8")
        ).hexdigest()
    )


def recall_outcome_hash(outcome: RecallOutcomeV1) -> str:
    payload = outcome.model_dump(mode="json", exclude_none=False)
    del payload["trace_hash"]
    return sha256_frame("recall-outcome.v1", payload, prefix="rch1")
