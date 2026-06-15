from __future__ import annotations

from typing import Literal, Protocol

from pydantic import Field, model_validator

from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse, StrictModel

AlignmentFailureCode = Literal[
    "none",
    "wrong_intent",
    "wrong_artifact",
    "wrong_action",
    "wrong_strategy",
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
    "recompose",
    "return_clarification",
    "return_unable",
]


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
                raise ValueError("aligned verdicts must be safe_to_return")
            if self.failure_code != "none" or self.remediation != "none":
                raise ValueError(
                    "aligned verdicts must use failure_code=none and remediation=none"
                )
        if self.remediation == "refetch_document_context" and not (
            self.refined_evidence_query or ""
        ).strip():
            raise ValueError(
                "refetch_document_context requires refined_evidence_query"
            )
        return self


class ReplyAlignmentVerifier(Protocol):
    async def verify(
        self,
        *,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        directive: ResponseDirective,
        evidence_facts: list[EvidenceFact],
        business_facts: BusinessFacts,
        response: ReplyResponse,
        attempt: int = 0,
    ) -> ReplyAlignmentVerdict: ...


class NoopReplyAlignmentVerifier:
    async def verify(
        self,
        *,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        directive: ResponseDirective,
        evidence_facts: list[EvidenceFact],
        business_facts: BusinessFacts,
        response: ReplyResponse,
        attempt: int = 0,
    ) -> ReplyAlignmentVerdict:
        del (
            request,
            canonical_context,
            plan,
            directive,
            evidence_facts,
            business_facts,
            response,
            attempt,
        )
        return ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            confidence=1.0,
        )
