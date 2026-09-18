from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field, model_validator

from market_support_crewai_agent.runtime.context.stage_inputs import (
    SanitizedAlignmentVerifierInputV1,
)
from market_support_crewai_agent.schemas.base import StrictModel

AlignmentFailureCode = Literal[
    "none",
    "wrong_intent",
    "wrong_artifact",
    "wrong_action",
    "wrong_material_pack_option",
    "wrong_report_scope",
    "missing_answer",
    "missing_evidence",
    "unsupported_claim",
    "policy_or_compliance_mismatch",
    "unsafe_action",
    "composer_drift",
    "ambiguous_request",
]

AlignmentRemediation = Literal[
    "none",
    "replan",
    "refetch_document_context",
    "refetch_report_scope",
    "recompose",
    "return_clarification",
    "return_unable",
]
ReportScopeRefetchQuery = Literal["report_scope_products", "report_scope_summary"]
_REPORT_SCOPE_REFETCH_QUERIES: frozenset[str] = frozenset(
    ("report_scope_products", "report_scope_summary")
)


@dataclass(frozen=True, slots=True)
class AlignmentVerdictContractError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


class ReplyAlignmentVerdict(StrictModel):
    contract_version: Literal["reply-alignment-verdict"] = "reply-alignment-verdict"
    aligned: bool
    safe_to_return: bool
    failure_code: AlignmentFailureCode = "none"
    rationale: str = Field(default="", max_length=400)
    remediation: AlignmentRemediation = "none"
    refined_evidence_query: str | None = Field(default=None, max_length=200)
    planner_feedback: str | None = Field(default=None, max_length=300)
    composer_feedback: str | None = Field(default=None, max_length=300)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_shape(self):
        if self.aligned:
            if not self.safe_to_return:
                raise AlignmentVerdictContractError(
                    "aligned_verdict_requires_safe_to_return"
                )
            if self.failure_code != "none" or self.remediation != "none":
                raise AlignmentVerdictContractError(
                    "aligned_verdict_requires_none_failure_and_remediation"
                )
        if (
            self.remediation in {"refetch_document_context", "refetch_report_scope"}
            and not (self.refined_evidence_query or "").strip()
        ):
            raise AlignmentVerdictContractError(
                "alignment_refetch_requires_refined_evidence_query"
            )
        if (
            self.remediation == "refetch_report_scope"
            and self.refined_evidence_query not in _REPORT_SCOPE_REFETCH_QUERIES
        ):
            raise AlignmentVerdictContractError(
                "alignment_report_refetch_query_not_allowed"
            )
        return self


class ReplyAlignmentVerifier(Protocol):
    async def verify(
        self,
        input_value: SanitizedAlignmentVerifierInputV1,
    ) -> ReplyAlignmentVerdict: ...


class NoopReplyAlignmentVerifier:
    async def verify(
        self,
        input_value: SanitizedAlignmentVerifierInputV1,
    ) -> ReplyAlignmentVerdict:
        del input_value
        return ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            confidence=1.0,
        )
