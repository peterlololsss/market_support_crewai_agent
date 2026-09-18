from __future__ import annotations

from dataclasses import dataclass
from typing import final

from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.planning import finalize_execution_plan_v2
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    ApprovedKnowledgeSelection,
)
from market_support_crewai_agent.runtime.recall.approved_static_selector import (
    ApprovedKnowledgeCandidate,
)
from tests.helpers.reply_contract_requests import make_v2_envelope


@dataclass(frozen=True, slots=True)
class DirectMediaInputs:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    plan: ExecutionPlanV2
    state_key_ref: str


def direct_media_inputs(*, enabled: bool) -> DirectMediaInputs:
    envelope = make_v2_envelope(
        "公司股权结构是什么？",
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
    authority = business_scope_authority_v1(envelope.request.business_scope)
    policy = PolicyManifestV2.from_core(
        compile_policy_authority_core_v1(
            envelope.request,
            authority,
            group_recall_mode="off",
        ),
        policy_ledger_summary_v1((), 0),
    )
    plan_policy = policy if enabled else direct_media_inputs(enabled=True).policy
    plan = finalize_execution_plan_v2(
        DeterministicPlanOriginInputV1(
            user_need="company shareholding",
            units=(
                DeterministicPlanUnitV1(
                    unit_id="company-knowledge",
                    manifest_id="answer_internal_company_knowledge",
                    answerability_policy="answer",
                    evidence_query="company shareholding",
                ),
            ),
        ),
        plan_policy,
        authority,
        origin="deterministic",
    )
    return DirectMediaInputs(envelope.request, policy, plan, envelope.state_key_ref)


def document_mcp_cache_config() -> DocumentMcpCacheConfigV1:
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


@final
class DocumentProvider:
    contexts: tuple[GatewayDocumentContextV1, ...]
    calls: int
    cache: dict[str, str]

    def __init__(self, contexts: tuple[GatewayDocumentContextV1, ...] = ()) -> None:
        self.contexts = contexts
        self.calls = 0
        self.cache = {"warm": "document"}

    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[GatewayDocumentContextV1, ...]:
        del request, evidence_query, cache_authority
        self.calls += 1
        return self.contexts


@final
class StaticProvider:
    contexts: tuple[GatewayStaticContextV1, ...]
    calls: int
    cache: dict[str, str]

    def __init__(self, contexts: tuple[GatewayStaticContextV1, ...] = ()) -> None:
        self.contexts = contexts
        self.calls = 0
        self.cache = {"warm": "static"}

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
class DirectImageBudgetSelector:
    max_images: list[int]

    def __init__(self) -> None:
        self.max_images = []

    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection:
        del user_message, evidence_query, catalog_manifest, max_entries
        self.max_images.append(max_images)
        return ApprovedKnowledgeSelection()
