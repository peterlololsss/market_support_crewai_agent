from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.knowledge.approved_knowledge import (
    ApprovedKnowledgeEvidenceService,
    NoopApprovedKnowledgeEvidenceService,
)
from market_support_crewai_agent.runtime.domain.business_facts import (
    BusinessFacts,
    derive_business_facts,
)
from market_support_crewai_agent.runtime.domain.capabilities import (
    ordered_resolve_types,
)
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.evidence.document_mcp import (
    DocumentMcpEvidenceService,
    NoopDocumentMcpEvidenceService,
)
from market_support_crewai_agent.runtime.evidence.report_scope import (
    NoopReportScopeEvidenceService,
    ReportScopeEvidenceService,
)
from market_support_crewai_agent.runtime.evidence import (
    EvidenceFact,
    evidence_facts_from_action_history,
    evidence_facts_from_preflight,
)
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.schemas import AdapterResolveType, ReplyRequest


@dataclass(frozen=True)
class EvidenceExecutionResult:
    preflight: AdapterPreflightSnapshot
    evidence_facts: list[EvidenceFact]
    business_facts: BusinessFacts


class EvidenceExecutor:
    """Runs deterministic evidence wrappers after plan validation.

    This boundary owns adapter resolve/preflight and feature-flag-controlled document MCP
    evidence. Wrappers run after policy and canonical entity validation, not as
    free-form CrewAI tools.
    """

    def __init__(
        self,
        preflight_service: AdapterPreflightService,
        document_evidence_service: (
            DocumentMcpEvidenceService | NoopDocumentMcpEvidenceService | None
        ) = None,
        approved_knowledge_service: (
            ApprovedKnowledgeEvidenceService | NoopApprovedKnowledgeEvidenceService | None
        ) = None,
        report_scope_service: (
            ReportScopeEvidenceService | NoopReportScopeEvidenceService | None
        ) = None,
    ) -> None:
        self.preflight_service = preflight_service
        self.document_evidence_service = (
            document_evidence_service or NoopDocumentMcpEvidenceService()
        )
        self.approved_knowledge_service = (
            approved_knowledge_service or ApprovedKnowledgeEvidenceService()
        )
        self.report_scope_service = report_scope_service or ReportScopeEvidenceService()

    async def execute(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
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
        evidence_facts.extend(
            await self.report_scope_service.collect(
                request,
                canonical_context,
                plan,
                policy,
                preflight,
            )
        )
        evidence_facts.extend(
            await self.approved_knowledge_service.collect(
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
    plan: ExecutionPlan,
    policy: PolicyManifest,
) -> list[AdapterResolveType]:
    requested: list[AdapterResolveType] = []

    for resolve_spec in plan.adapter_resolves:
        if resolve_spec.resolve_type in policy.allowed_adapter_resolves:
            requested.append(resolve_spec.resolve_type)
    return ordered_resolve_types(requested)


def _resolve_strategies_for_plan(plan: ExecutionPlan) -> dict[AdapterResolveType, str]:
    strategies: dict[AdapterResolveType, str] = {}

    for resolve_spec in plan.adapter_resolves:
        if not resolve_spec.strategy:
            continue
        strategies.setdefault(resolve_spec.resolve_type, resolve_spec.strategy)

    return strategies
