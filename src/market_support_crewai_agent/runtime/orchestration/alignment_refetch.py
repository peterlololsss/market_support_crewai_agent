from __future__ import annotations

from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)


_REPORT_SCOPE_CAPABILITIES = {"weekly_report", "monthly_report"}
_REPORT_SCOPE_SENTINELS = {"report_scope_products", "report_scope_summary"}


def report_scope_refetch_requested(
    verdict: ReplyAlignmentVerdict,
    plan: ExecutionPlan,
) -> bool:
    del plan
    return verdict.remediation == "refetch_report_scope"


def plan_can_refetch_report_scope(
    plan: ExecutionPlan,
    policy: PolicyManifest,
) -> bool:
    return any(
        capability in plan.capabilities and capability in policy.allowed_capabilities
        for capability in _REPORT_SCOPE_CAPABILITIES
    )


def report_scope_refetch_query(verdict: ReplyAlignmentVerdict) -> str:
    query = str(verdict.refined_evidence_query or "").strip()
    return query if query in _REPORT_SCOPE_SENTINELS else ""
