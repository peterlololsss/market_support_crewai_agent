from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from market_support_crewai_agent.runtime.context.payload_store import (
    ScopedContextPayloadStoreV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    ApprovedStaticKnowledgeGatewayAdapter,
    DocumentMcpGatewayAdapter,
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.client import (
    DocumentMcpClient,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import V2Composer
from market_support_crewai_agent.runtime.state.action_ledger import (
    ActionLedger,
    get_action_ledger,
)
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.state.coordinator_protocol import (
    ReplyTurnStateCoordinatorV1,
)
from market_support_crewai_agent.runtime.state.coordinator_provider import (
    get_reply_state_coordinator,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerifier,
)
from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings


class AgentRuntimeError(RuntimeError):
    """Raised when the CrewAI runtime cannot produce a valid reply."""


_APP_SETTINGS = get_settings()
_APP_CONVERSATION_STORE = ConversationStore.from_settings(_APP_SETTINGS)
_APP_ACTION_LEDGER = get_action_ledger()
_APP_ADAPTER_PREFLIGHT = AdapterPreflightService(settings=_APP_SETTINGS)
_APP_CONTEXT_PAYLOAD_STORE = ScopedContextPayloadStoreV1(
    conversation_ttl_seconds=_APP_SETTINGS.agent_conversation_ttl_seconds,
    direct_audit_ttl_seconds=_APP_SETTINGS.agent_direct_audit_ttl_seconds,
)


@dataclass(frozen=True, slots=True)
class RuntimeDeps:
    settings: Settings
    conversation_store: ConversationStore
    action_ledger: ActionLedger
    preflight_service: AdapterPreflightService
    evidence_executor: EvidenceExecutor
    context_payload_store: ScopedContextPayloadStoreV1
    coordinator: ReplyTurnStateCoordinatorV1
    document_cache_config: DocumentMcpCacheConfigV1 | None = None
    alignment_verifier: ReplyAlignmentVerifier | None = None
    v2_composer: V2Composer | None = None


def build_runtime_deps(
    *,
    settings: Settings | None = None,
    conversation_store: ConversationStore | None = None,
    action_ledger: ActionLedger | None = None,
    preflight_service: AdapterPreflightService | None = None,
    evidence_executor: EvidenceExecutor | None = None,
    context_payload_store: ScopedContextPayloadStoreV1 | None = None,
    coordinator: ReplyStateTransactionCoordinatorV1 | None = None,
    internal_company_knowledge_gateway: InternalCompanyKnowledgeGatewayV1 | None = None,
    document_cache_config: DocumentMcpCacheConfigV1 | None = None,
    alignment_verifier: ReplyAlignmentVerifier | None = None,
    v2_composer: V2Composer | None = None,
) -> RuntimeDeps:
    use_app_singletons = settings is None
    resolved_settings = settings or _APP_SETTINGS
    resolved_preflight_service = preflight_service or (
        _APP_ADAPTER_PREFLIGHT
        if use_app_singletons
        else AdapterPreflightService(settings=resolved_settings)
    )
    resolved_gateway, resolved_cache_config = _internal_knowledge_dependencies(
        resolved_settings,
        gateway=internal_company_knowledge_gateway,
        cache_config=document_cache_config,
    )

    return RuntimeDeps(
        settings=resolved_settings,
        conversation_store=conversation_store
        if conversation_store is not None
        else (
            _APP_CONVERSATION_STORE
            if use_app_singletons
            else ConversationStore.from_settings(resolved_settings)
        ),
        action_ledger=action_ledger or _APP_ACTION_LEDGER,
        preflight_service=resolved_preflight_service,
        evidence_executor=evidence_executor
        if evidence_executor is not None
        else EvidenceExecutor(
            resolved_preflight_service,
            internal_company_knowledge_gateway=resolved_gateway,
        ),
        context_payload_store=context_payload_store
        if context_payload_store is not None
        else (
            _APP_CONTEXT_PAYLOAD_STORE
            if use_app_singletons
            else ScopedContextPayloadStoreV1(
                conversation_ttl_seconds=resolved_settings.agent_conversation_ttl_seconds,
                direct_audit_ttl_seconds=resolved_settings.agent_direct_audit_ttl_seconds,
            )
        ),
        coordinator=coordinator or get_reply_state_coordinator(),
        document_cache_config=resolved_cache_config,
        alignment_verifier=alignment_verifier,
        v2_composer=v2_composer,
    )


def _internal_knowledge_dependencies(
    settings: Settings,
    *,
    gateway: InternalCompanyKnowledgeGatewayV1 | None,
    cache_config: DocumentMcpCacheConfigV1 | None,
) -> tuple[InternalCompanyKnowledgeGatewayV1 | None, DocumentMcpCacheConfigV1 | None]:
    config = cache_config or _sealed_document_cache_config(settings)
    if gateway is not None:
        return gateway, config
    document_provider = _document_gateway_provider(settings, config)
    return (
        InternalCompanyKnowledgeGatewayV1(
            document_provider=document_provider,
            static_provider=ApprovedStaticKnowledgeGatewayAdapter(),
        ),
        config,
    )


def _document_gateway_provider(
    settings: Settings,
    cache_config: DocumentMcpCacheConfigV1 | None,
) -> DocumentMcpGatewayAdapter | _NoopDocumentGatewayProvider:
    if not settings.doc_mcp_enabled or not settings.doc_mcp_base_url:
        return _NoopDocumentGatewayProvider()
    if settings.doc_mcp_cache_ttl_seconds > 0 and cache_config is None:
        return _NoopDocumentGatewayProvider()
    return DocumentMcpGatewayAdapter(DocumentMcpClient(settings))


class _NoopDocumentGatewayProvider:
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[GatewayDocumentContextV1, ...]:
        del request, evidence_query, cache_authority
        return ()


def _sealed_document_cache_config(
    settings: Settings,
) -> DocumentMcpCacheConfigV1 | None:
    if settings.doc_mcp_cache_ttl_seconds <= 0:
        return None
    try:
        return DocumentMcpCacheConfigV1(
            client_contract_version="document-mcp-client.v1",
            corpus_version="document-mcp-corpus.v1",
            request_schema_hash="osh1:4a056b7f023a7b26ba65b73db3e2e296f7d2ba889fe9fd54c3a5d9221bd0b5a1",
            response_schema_hash="osh1:6ffdc9963cee4d339749e972ab0a22cbee1e34ccdcbd1a4595cf84a0e4d357f7",
            timeout_milliseconds=round(settings.doc_mcp_timeout_seconds * 1_000),
            max_candidates=50,
            ttl_seconds=max(1, ceil(settings.doc_mcp_cache_ttl_seconds)),
            capacity=256,
        )
    except ValueError:
        return None
