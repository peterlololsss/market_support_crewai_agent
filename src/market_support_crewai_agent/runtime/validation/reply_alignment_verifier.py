from __future__ import annotations

from typing import Literal, Protocol

from pydantic import Field, model_validator

from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse, StrictModel

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
        if self.remediation in {"refetch_document_context", "refetch_report_scope"} and not (
            self.refined_evidence_query or ""
        ).strip():
            raise ValueError(
                f"{self.remediation} requires refined_evidence_query"
            )
        if (
            self.remediation == "refetch_report_scope"
            and self.refined_evidence_query not in _REPORT_SCOPE_REFETCH_QUERIES
        ):
            raise ValueError(
                "refetch_report_scope requires refined_evidence_query to be "
                "report_scope_products or report_scope_summary"
            )
        return self


class ReplyAlignmentVerifier(Protocol):
    async def verify(
        self,
        *,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        domain_context: DomainContext,
        plan: ExecutionPlan,
        directive: ResponseDirective,
        evidence_facts: list[EvidenceFact],
        business_facts: BusinessFacts,
        response: ReplyResponse,
        guardrail_decisions: list[GuardrailDecision] | None = None,
        attempt: int = 0,
    ) -> ReplyAlignmentVerdict: ...


class NoopReplyAlignmentVerifier:
    async def verify(
        self,
        *,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        domain_context: DomainContext,
        plan: ExecutionPlan,
        directive: ResponseDirective,
        evidence_facts: list[EvidenceFact],
        business_facts: BusinessFacts,
        response: ReplyResponse,
        guardrail_decisions: list[GuardrailDecision] | None = None,
        attempt: int = 0,
    ) -> ReplyAlignmentVerdict:
        del (
            request,
            canonical_context,
            domain_context,
            plan,
            directive,
            evidence_facts,
            business_facts,
            response,
            guardrail_decisions,
            attempt,
        )
        return ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            confidence=1.0,
        )
