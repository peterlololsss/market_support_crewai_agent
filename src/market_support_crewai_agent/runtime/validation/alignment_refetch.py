from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field

from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.base import StrictModel


class AlignmentRefetchRequestV1(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1, max_length=120)
    manifest_ref: ManifestRefV1
    refined_evidence_query: str = Field(min_length=1, max_length=200)
    attempt: int = Field(ge=1, le=2)


@dataclass(frozen=True, slots=True)
class AlignmentRefetchError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


def build_alignment_refetch_request_v1(
    plan: ExecutionPlanV2,
    verdict: ReplyAlignmentVerdict,
    attempt: Literal[1, 2],
) -> AlignmentRefetchRequestV1:
    query = verdict.refined_evidence_query
    if query is None:
        raise AlignmentRefetchError("alignment_refetch_query_required")
    eligible_units = tuple(
        unit
        for unit in plan.units
        if _unit_supports_refetch(unit.manifest_ref, verdict.remediation)
    )
    if len(eligible_units) != 1:
        raise AlignmentRefetchError("alignment_refetch_unit_not_unique")
    unit = eligible_units[0]
    return AlignmentRefetchRequestV1(
        unit_id=unit.unit_id,
        manifest_ref=unit.manifest_ref,
        refined_evidence_query=query,
        attempt=attempt,
    )


def validate_alignment_refetch_request_v1(
    plan: ExecutionPlanV2,
    request: AlignmentRefetchRequestV1,
) -> None:
    matches = tuple(
        unit
        for unit in plan.units
        if unit.unit_id == request.unit_id and unit.manifest_ref == request.manifest_ref
    )
    if len(matches) != 1:
        raise AlignmentRefetchError("alignment_refetch_plan_binding_mismatch")


def _unit_supports_refetch(
    manifest_ref: ManifestRefV1,
    remediation: str,
) -> bool:
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(manifest_ref.manifest_id)
    if manifest is None or manifest.manifest_version != manifest_ref.manifest_version:
        return False
    allowed_facts = set(manifest.evidence_contract.allowed_fact_types)
    if remediation == "refetch_document_context":
        return "document_context" in allowed_facts
    if remediation == "refetch_report_scope":
        return bool(
            allowed_facts
            & {"report_scope_summary", "report_scope_match", "report_scope_products"}
        )
    return False


__all__ = [
    "AlignmentRefetchRequestV1",
    "AlignmentRefetchError",
    "build_alignment_refetch_request_v1",
    "validate_alignment_refetch_request_v1",
]
