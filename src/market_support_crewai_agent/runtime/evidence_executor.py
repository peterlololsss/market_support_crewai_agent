from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.adapter_preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.business_facts import (
    BusinessFacts,
    derive_business_facts,
)
from market_support_crewai_agent.runtime.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.document_mcp import (
    DocumentMcpEvidenceService,
    NoopDocumentMcpEvidenceService,
)
from market_support_crewai_agent.runtime.evidence import (
    EvidenceFact,
    evidence_facts_from_action_history,
    evidence_facts_from_preflight,
)
from market_support_crewai_agent.runtime.planning import ReplyPlan
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.schemas import AdapterResolveType, ReplyRequest

_RESOLVE_BY_READ_CAPABILITY = {
    "resolve_material_pack": "material_pack",
    "resolve_weekly_report": "weekly_report",
    "resolve_monthly_report": "monthly_report",
    "resolve_sales_mention": "sales_mention",
}
_RESOLVE_BY_ACTION = {
    "send_material_pack": "material_pack",
    "send_weekly_report": "weekly_report",
    "send_monthly_report": "monthly_report",
}


@dataclass(frozen=True)
class EvidenceExecutionResult:
    preflight: AdapterPreflightSnapshot
    evidence_facts: list[EvidenceFact]
    business_facts: BusinessFacts


class EvidenceExecutor:
    """Runs deterministic evidence wrappers after plan validation.

    This boundary owns adapter resolve/preflight and feature-gated document MCP
    evidence. Wrappers run after policy and canonical entity validation, not as
    free-form CrewAI tools.
    """

    def __init__(
        self,
        preflight_service: AdapterPreflightService,
        document_evidence_service: (
            DocumentMcpEvidenceService | NoopDocumentMcpEvidenceService | None
        ) = None,
    ) -> None:
        self.preflight_service = preflight_service
        self.document_evidence_service = (
            document_evidence_service or NoopDocumentMcpEvidenceService()
        )

    async def execute(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ReplyPlan,
        policy: PolicyManifest,
        action_history: list[ActionLedgerRecord] | None = None,
    ) -> EvidenceExecutionResult:
        resolve_types = _resolve_types_for_plan(plan, policy)
        preflight = await self.preflight_service.collect(
            request,
            canonical_context=canonical_context,
            resolve_types=resolve_types,
            resolve_strategies=_resolve_strategies_for_plan(plan),
        )
        evidence_facts = evidence_facts_from_preflight(preflight)
        evidence_facts.extend(evidence_facts_from_action_history(action_history))
        evidence_facts.extend(
            await self.document_evidence_service.collect(
                request,
                canonical_context,
                plan,
                policy,
            )
        )
        business_facts = derive_business_facts(evidence_facts, request)
        return EvidenceExecutionResult(
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
        )


def _resolve_types_for_plan(
    plan: ReplyPlan,
    policy: PolicyManifest,
) -> list[AdapterResolveType]:
    requested: list[AdapterResolveType] = []

    for resolve_type in plan.required_adapter_resolves:
        if resolve_type in policy.required_adapter_resolves:
            requested.append(resolve_type)

    for evidence_request in plan.evidence_requests:
        resolve_type = _RESOLVE_BY_READ_CAPABILITY.get(evidence_request.capability)
        if resolve_type and resolve_type in policy.required_adapter_resolves:
            requested.append(resolve_type)

    for action in plan.candidate_actions:
        resolve_type = _RESOLVE_BY_ACTION.get(action.type)
        if (
            resolve_type
            and action.type in policy.allowed_side_effect_actions
            and resolve_type in policy.required_adapter_resolves
        ):
            requested.append(resolve_type)

    requested.append("sales_mention")
    return _ordered_unique_resolve_types(requested)


def _resolve_strategies_for_plan(plan: ReplyPlan) -> dict[AdapterResolveType, str]:
    strategies: dict[AdapterResolveType, str] = {}

    for evidence_request in plan.evidence_requests:
        if not evidence_request.strategy:
            continue
        resolve_type = _RESOLVE_BY_READ_CAPABILITY.get(evidence_request.capability)
        if resolve_type is not None:
            strategies.setdefault(resolve_type, evidence_request.strategy)

    for action in plan.candidate_actions:
        if not action.strategy:
            continue
        resolve_type = _RESOLVE_BY_ACTION.get(action.type)
        if resolve_type is not None:
            strategies.setdefault(resolve_type, action.strategy)

    return strategies


def _ordered_unique_resolve_types(
    resolve_types: list[AdapterResolveType],
) -> list[AdapterResolveType]:
    order: tuple[AdapterResolveType, ...] = (
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    )
    requested = set(resolve_types)
    return [resolve_type for resolve_type in order if resolve_type in requested]
