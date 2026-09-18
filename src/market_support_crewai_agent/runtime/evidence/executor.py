from __future__ import annotations

from typing import Protocol

from market_support_crewai_agent.runtime.evidence import adapter_facts, grounding
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    RegisteredMediaBindingV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    kernel_subject_ref,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.adapter.report_scope import (
    ReportScopeEvidenceProvider,
    ReportScopeEvidenceService,
)
from market_support_crewai_agent.runtime.observability.runtime_trace import trace_span
from market_support_crewai_agent.runtime.planning.models import (
    ExecutionPlanV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.policy.capabilities.mappings import (
    ordered_resolve_types,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.policy.ontology import (
    DomainContextV1Builder,
)
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
    validate_alignment_refetch_request_v1,
)
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType


class AdapterPreflightProvider(Protocol):
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot: ...


class CanonicalEvidenceExecutionError(RuntimeError):
    pass


class EvidenceExecutor:
    """Runs deterministic evidence wrappers after plan validation.

    This boundary owns adapter resolve/preflight and feature-flag-controlled document MCP
    evidence. Wrappers run after policy validation, not as free-form CrewAI tools.
    """

    def __init__(
        self,
        preflight_service: AdapterPreflightProvider,
        internal_company_knowledge_gateway: InternalCompanyKnowledgeGatewayV1
        | None = None,
        report_scope_service: ReportScopeEvidenceProvider | None = None,
    ) -> None:
        self.preflight_service: AdapterPreflightProvider = preflight_service
        self.internal_company_knowledge_gateway: (
            InternalCompanyKnowledgeGatewayV1 | None
        ) = internal_company_knowledge_gateway
        self.report_scope_service: ReportScopeEvidenceProvider | None = (
            report_scope_service
        )

    async def execute_v2(
        self,
        request: KernelReplyRequestV1,
        plan: ExecutionPlanV2,
        policy: PolicyManifestV2,
        *,
        scope_authority: BusinessScopeAuthorityV1,
        state_key_ref: str | None = None,
        document_cache_config: DocumentMcpCacheConfigV1 | None = None,
        alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
    ) -> grounding.CanonicalEvidenceExecutionResultV1:
        if request.business_scope != scope_authority.scope:
            raise CanonicalEvidenceExecutionError("business_scope_authority_mismatch")
        if policy.business_scope_hash != scope_authority.business_scope_hash:
            raise CanonicalEvidenceExecutionError("policy_business_scope_hash_mismatch")
        if alignment_refetch_request is not None:
            validate_alignment_refetch_request_v1(plan, alignment_refetch_request)
        resolve_types = _resolve_types_for_plan_v2(plan, policy)
        options = _resolve_material_pack_options_for_plan_v2(plan)
        with trace_span("evidence.adapter_preflight"):
            preflight = await self.preflight_service.collect(
                request,
                resolve_types=resolve_types,
                resolve_material_pack_options=options,
            )
        canonical_facts, resolve_bindings = adapter_facts.canonical_adapter_facts_v1(
            plan, preflight
        )
        report_scope_service = self.report_scope_service
        if report_scope_service is None and _plan_uses_adapter_report_scope_v1(plan):
            match self.preflight_service:
                case AdapterPreflightService(adapter_client=adapter_client):
                    report_scope_service = ReportScopeEvidenceService(
                        adapter_client=adapter_client
                    )
                case _:
                    raise CanonicalEvidenceExecutionError(
                        "report_scope_service_required"
                    )
        if report_scope_service is not None:
            with trace_span("evidence.adapter_report_scope"):
                report_facts = await report_scope_service.collect(
                    request,
                    plan,
                    policy,
                    preflight,
                    alignment_refetch_request=alignment_refetch_request,
                )
            canonical_facts = (*canonical_facts, *report_facts)
        media_bindings: tuple[RegisteredMediaBindingV1, ...] = ()
        if self.internal_company_knowledge_gateway is not None:
            if alignment_refetch_request is None:
                knowledge = await self.internal_company_knowledge_gateway.collect(
                    request=request,
                    plan=plan,
                    policy=policy,
                    state_key_ref=state_key_ref,
                    document_cache_config=document_cache_config,
                )
            else:
                knowledge = await self.internal_company_knowledge_gateway.collect(
                    request=request,
                    plan=plan,
                    policy=policy,
                    state_key_ref=state_key_ref,
                    document_cache_config=document_cache_config,
                    alignment_refetch_request=alignment_refetch_request,
                )
            canonical_facts = (*canonical_facts, *knowledge.facts)
            media_bindings = knowledge.media_bindings
        groundings = grounding.ground_execution_plan_v2(
            plan,
            policy,
            canonical_facts,
            resolve_bindings,
        )
        domain_context = DomainContextV1Builder().build(
            request,
            conversation_metadata={
                "context_id": request.context_id,
                "conversation_key": kernel_subject_ref(request),
            },
            scope_authority=scope_authority,
        )
        return grounding.CanonicalEvidenceExecutionResultV1(
            preflight=preflight,
            canonical_facts=canonical_facts,
            resolve_bindings=resolve_bindings,
            groundings=groundings,
            domain_context=domain_context,
            media_bindings=media_bindings,
        )


def _resolve_types_for_plan_v2(
    plan: ExecutionPlanV2,
    policy: PolicyManifestV2,
) -> list[AdapterResolveType]:
    return ordered_resolve_types(
        [
            resolve.resolve_type
            for resolve in plan.adapter_resolves
            if resolve.resolve_type in policy.allowed_adapter_resolves
        ]
    )


def _plan_uses_adapter_report_scope_v1(plan: ExecutionPlanV2) -> bool:
    for unit in plan.units:
        manifest = CAPABILITY_MANIFEST_REGISTRY.find(unit.manifest_ref.manifest_id)
        if (
            manifest is not None
            and manifest.manifest_version == unit.manifest_ref.manifest_version
            and "adapter_report_scope"
            in manifest.evidence_contract.allowed_source_types
        ):
            return True
    return False


def _resolve_material_pack_options_for_plan_v2(
    plan: ExecutionPlanV2,
) -> dict[AdapterResolveType, str]:
    return {
        resolve.resolve_type: resolve.material_pack_option
        for resolve in plan.adapter_resolves
        if resolve.material_pack_option is not None
    }
