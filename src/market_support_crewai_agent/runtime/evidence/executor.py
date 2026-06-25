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
from market_support_crewai_agent.runtime.domain.ontology import (
    DomainContext,
    DomainContextBuilder,
)
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
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    retrieval_source_guard,
)
from market_support_crewai_agent.runtime.validation.execution_tool_guard import (
    execution_tool_guard,
)
from market_support_crewai_agent.runtime.state.runtime_trace import trace_event, trace_span
from market_support_crewai_agent.schemas import AdapterResolveType, ReplyRequest


@dataclass(frozen=True)
class EvidenceExecutionResult:
    preflight: AdapterPreflightSnapshot
    evidence_facts: list[EvidenceFact]
    business_facts: BusinessFacts
    domain_context: DomainContext
    guardrail_decisions: list[GuardrailDecision]


class EvidenceExecutor:
    """Runs deterministic evidence wrappers after plan validation.

    This boundary owns adapter resolve/preflight and feature-flag-controlled document MCP
    evidence. Wrappers run after policy validation, not as free-form CrewAI tools.
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
        plan: ExecutionPlan,
        policy: PolicyManifest,
        action_history: list[ActionLedgerRecord] | None = None,
    ) -> EvidenceExecutionResult:
        initial_domain_context = DomainContextBuilder().build(
            request,
            conversation_metadata={
                "context_id": request.context_id,
                "conversation_key": request.conversation_key,
            },
        )
        tool_decision = execution_tool_guard(
            plan=plan,
            policy=policy,
            domain_context=initial_domain_context,
        )
        if tool_decision.outcome == "block":
            evidence_facts = evidence_facts_from_action_history(action_history)
            business_facts = derive_business_facts(evidence_facts, request)
            return EvidenceExecutionResult(
                preflight=AdapterPreflightSnapshot.empty(),
                evidence_facts=evidence_facts,
                business_facts=business_facts,
                domain_context=initial_domain_context,
                guardrail_decisions=[tool_decision],
            )

        resolve_types = _resolve_types_for_plan(plan, policy)
        trace_event("evidence.resolve_types", resolve_types=resolve_types)
        with trace_span("evidence.adapter_preflight"):
            preflight = await self.preflight_service.collect(
                request,
                resolve_types=resolve_types,
                resolve_material_pack_options=_resolve_material_pack_options_for_plan(plan),
            )
        evidence_facts = evidence_facts_from_preflight(preflight)
        evidence_facts.extend(evidence_facts_from_action_history(action_history))
        with trace_span("evidence.document_mcp"):
            evidence_facts.extend(
                await self.document_evidence_service.collect(
                    request,
                    plan,
                    policy,
                )
            )
        with trace_span("evidence.report_scope"):
            evidence_facts.extend(
                await self.report_scope_service.collect(
                    request,
                    plan,
                    policy,
                    preflight,
                )
            )
        with trace_span("evidence.approved_knowledge"):
            evidence_facts.extend(
                await self.approved_knowledge_service.collect(
                    request,
                    plan,
                    policy,
                )
            )
        with trace_span("evidence.derive_business_facts"):
            business_facts = derive_business_facts(evidence_facts, request)
        domain_context = DomainContextBuilder().build(
            request,
            available_artifacts=[preflight, *evidence_facts],
            conversation_metadata={
                "context_id": request.context_id,
                "conversation_key": request.conversation_key,
            },
        )
        with trace_span("evidence.retrieval_source_guard"):
            source_decision = retrieval_source_guard(
                plan=plan,
                policy=policy,
                evidence_facts=evidence_facts,
                domain_context=domain_context,
            )
        return EvidenceExecutionResult(
            preflight=preflight,
            evidence_facts=evidence_facts,
            business_facts=business_facts,
            domain_context=domain_context,
            guardrail_decisions=[tool_decision, source_decision],
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


def _resolve_material_pack_options_for_plan(
    plan: ExecutionPlan,
) -> dict[AdapterResolveType, str]:
    options: dict[AdapterResolveType, str] = {}

    for resolve_spec in plan.adapter_resolves:
        if not resolve_spec.material_pack_option:
            continue
        options.setdefault(resolve_spec.resolve_type, resolve_spec.material_pack_option)

    return options
