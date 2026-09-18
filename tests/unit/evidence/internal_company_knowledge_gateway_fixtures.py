from __future__ import annotations

import anyio
from typing_extensions import TypedDict, Unpack, final, override

from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
    InternalCompanyKnowledgeResultV1,
    PostPlanDocumentKnowledgeProvider,
    PostPlanStaticKnowledgeProvider,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.planning import (
    ExecutionPlanV2,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from tests.helpers.reply_contract_requests import make_v2_envelope


@final
class DocumentProvider(PostPlanDocumentKnowledgeProvider):
    def __init__(self, contexts: tuple[GatewayDocumentContextV1, ...]) -> None:
        self.contexts: tuple[GatewayDocumentContextV1, ...] = contexts
        self.calls: int = 0
        self.cache_authorities: list[DocumentMcpCacheAuthorityV1 | None] = []

    @override
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[GatewayDocumentContextV1, ...]:
        del request, evidence_query
        self.calls += 1
        self.cache_authorities.append(cache_authority)
        return self.contexts


@final
class StaticProvider(PostPlanStaticKnowledgeProvider):
    def __init__(self, contexts: tuple[GatewayStaticContextV1, ...]) -> None:
        self.contexts: tuple[GatewayStaticContextV1, ...] = contexts
        self.calls: int = 0

    @override
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
    ) -> tuple[GatewayStaticContextV1, ...]:
        del request, evidence_query
        self.calls += 1
        return self.contexts


@final
class EmptyPreflight:
    async def collect(
        self,
        request: KernelReplyRequestV1,
        resolve_types: list[AdapterResolveType] | None = None,
        resolve_material_pack_options: dict[AdapterResolveType, str] | None = None,
    ) -> AdapterPreflightSnapshot:
        del request, resolve_types, resolve_material_pack_options
        return AdapterPreflightSnapshot.empty()


class _GatewayCollectArguments(TypedDict):
    request: KernelReplyRequestV1
    plan: ExecutionPlanV2
    policy: PolicyManifestV2
    state_key_ref: str | None
    document_cache_config: DocumentMcpCacheConfigV1 | None


def direct_inputs(
    *, enabled: bool, manifest_id: str = "answer_internal_company_knowledge"
) -> tuple[KernelReplyRequestV1, PolicyManifestV2, ExecutionPlanV2, str]:
    envelope = make_v2_envelope(
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:test",
            "direct_thread_ref": "direct:thread-1",
            "principal_ref": "principal:sender-1",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "test user",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"] if enabled else [],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
    request = envelope.request
    scope = business_scope_authority_v1(request.business_scope)
    policy = PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(request, scope, group_recall_mode="off"),
        policy_ledger_summary_v1((), 0),
    )
    plan_policy = policy
    if not enabled:
        _, plan_policy, _, _ = direct_inputs(enabled=True)
    plan = finalize_execution_plan_v2(
        DeterministicPlanOriginInputV1(
            user_need="company public fact",
            units=(
                DeterministicPlanUnitV1(
                    unit_id="company-fact",
                    manifest_id=manifest_id,
                    answerability_policy="answer",
                    evidence_query="company public fact",
                ),
            ),
        ),
        plan_policy,
        scope,
        origin="deterministic",
    )
    return request, policy, plan, envelope.state_key_ref


def cache_config() -> DocumentMcpCacheConfigV1:
    return DocumentMcpCacheConfigV1(
        client_contract_version="document-mcp-client.v1",
        corpus_version="company-public.v1",
        request_schema_hash="osh1:" + "1" * 64,
        response_schema_hash="osh1:" + "2" * 64,
        timeout_milliseconds=1_000,
        max_candidates=8,
        ttl_seconds=30,
        capacity=8,
    )


def group_inputs() -> tuple[
    KernelReplyRequestV1,
    PolicyManifestV2,
    ExecutionPlanV2,
    str,
]:
    envelope = make_v2_envelope()
    request = envelope.request
    scope = business_scope_authority_v1(request.business_scope)
    policy = PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(request, scope, group_recall_mode="off"),
        policy_ledger_summary_v1((), 0),
    )
    plan = finalize_execution_plan_v2(
        DeterministicPlanOriginInputV1(
            user_need="channel strategy summary",
            units=(
                DeterministicPlanUnitV1(
                    unit_id="strategy-summary",
                    manifest_id="answer_internal_company_knowledge",
                    answerability_policy="answer",
                    evidence_query="strategy summary",
                ),
            ),
        ),
        policy,
        scope,
        origin="deterministic",
    )
    return request, policy, plan, envelope.state_key_ref


def collect_gateway(
    gateway: InternalCompanyKnowledgeGatewayV1,
    **kwargs: Unpack[_GatewayCollectArguments],
) -> InternalCompanyKnowledgeResultV1:
    async def run() -> InternalCompanyKnowledgeResultV1:
        return await gateway.collect(**kwargs)

    return anyio.run(run)
