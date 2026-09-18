from __future__ import annotations

from market_support_crewai_agent.runtime.evidence import (
    internal_company_knowledge_models as knowledge_models,
)
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_admission import (
    document_fact,
    static_fact,
)
from market_support_crewai_agent.runtime.evidence.internal_knowledge_facts import (
    document_context_from_chunk,
)
from market_support_crewai_agent.runtime.evidence.internal_knowledge_selection import (
    approved_entry_by_id,
    knowledge_targets,
    validate_selection_for_gateway,
    validated_state_key_ref,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.client import (
    DocumentContextClient,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.parsing import (
    DocumentMcpError,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.recall.approved_static_catalog import (
    validated_catalog_manifest_ref,
)
from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    ApprovedKnowledgeSelector,
    DirectApprovedKnowledgeSelector,
    approved_knowledge_manifest,
)
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)


class DocumentMcpGatewayAdapter:
    def __init__(self, client: DocumentContextClient) -> None:
        self._client: DocumentContextClient = client

    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[knowledge_models.GatewayDocumentContextV1, ...]:
        try:
            chunks = await self._client.fetch_context_async(
                request,
                evidence_query=evidence_query,
                cache_authority=cache_authority,
            )
        except DocumentMcpError:
            return ()
        return tuple(document_context_from_chunk(chunk) for chunk in chunks)


class ApprovedStaticKnowledgeGatewayAdapter:
    def __init__(self, selector: ApprovedKnowledgeSelector | None = None) -> None:
        self._selector: ApprovedKnowledgeSelector = (
            selector or DirectApprovedKnowledgeSelector()
        )

    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
    ) -> tuple[knowledge_models.GatewayStaticContextV1, ...]:
        max_images = 0 if request.identity.scene == "direct" else 2
        selection = await self._selector.select(
            user_message=request.message,
            evidence_query=evidence_query,
            catalog_manifest=approved_knowledge_manifest(),
            max_entries=1,
            max_images=max_images,
        )
        selected = validate_selection_for_gateway(
            selection,
            max_entries=1,
            max_images=max_images,
        )
        if not selected.selected_entry_ids:
            return ()
        contexts: list[knowledge_models.GatewayStaticContextV1] = []
        for entry_id in selected.selected_entry_ids:
            entry = approved_entry_by_id(entry_id)
            if entry is None:
                continue
            manifest_ref = validated_catalog_manifest_ref(entry)
            if manifest_ref is None:
                continue
            asset_ids = tuple(
                asset_id
                for asset_id in selected.selected_image_asset_ids
                if asset_id in entry.image_asset_ids
            )
            text = entry.to_document_context(asset_ids)
            if text:
                contexts.append(
                    knowledge_models.GatewayStaticContextV1(
                        entry_id=entry.entry_id,
                        manifest_ref=manifest_ref,
                        text=text,
                        selected_asset_ids=asset_ids,
                    )
                )
        return tuple(contexts)


class InternalCompanyKnowledgeGatewayV1:
    def __init__(
        self,
        *,
        document_provider: knowledge_models.PostPlanDocumentKnowledgeProvider,
        static_provider: knowledge_models.PostPlanStaticKnowledgeProvider,
    ) -> None:
        self._document_provider: knowledge_models.PostPlanDocumentKnowledgeProvider = (
            document_provider
        )
        self._static_provider: knowledge_models.PostPlanStaticKnowledgeProvider = (
            static_provider
        )

    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        plan: ExecutionPlanV2,
        policy: PolicyManifestV2,
        state_key_ref: str | None,
        document_cache_config: DocumentMcpCacheConfigV1 | None,
        alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
    ) -> knowledge_models.InternalCompanyKnowledgeResultV1:
        if not policy.internal_company_knowledge_enabled:
            return knowledge_models.InternalCompanyKnowledgeResultV1((), ())
        targets = knowledge_targets(plan, policy)
        if not targets:
            return knowledge_models.InternalCompanyKnowledgeResultV1((), ())

        facts: list[CanonicalEvidenceFactV1] = []
        bindings: list[knowledge_models.RegisteredMediaBindingV1] = []
        for unit, manifest_ref in targets:
            query = evidence_query_for_unit(
                unit_id=unit.unit_id,
                unit_manifest_ref=unit.manifest_ref,
                unit_evidence_query=unit.evidence_query,
                alignment_refetch_request=alignment_refetch_request,
            )
            if query is None:
                continue
            cache_authority = document_cache_authority(
                state_key_ref=state_key_ref,
                document_cache_config=document_cache_config,
                policy=policy,
                manifest_ref=f"{manifest_ref.manifest_id}@{manifest_ref.manifest_version}",
            )
            documents = await self._document_provider.collect(
                request=request,
                evidence_query=query,
                cache_authority=cache_authority,
            )
            facts.extend(
                fact
                for context in documents
                if (fact := document_fact(context, unit)) is not None
            )
            static_contexts = await self._static_provider.collect(
                request=request,
                evidence_query=query,
            )
            for context in static_contexts:
                fact, context_bindings = static_fact(
                    context,
                    unit,
                    media_allowed=request.identity.scene == "group",
                )
                if fact is not None:
                    facts.append(fact)
                    bindings.extend(context_bindings)

        return deduplicate_knowledge_result(facts, bindings)


def evidence_query_for_unit(
    *,
    unit_id: str,
    unit_manifest_ref: ManifestRefV1,
    unit_evidence_query: str | None,
    alignment_refetch_request: AlignmentRefetchRequestV1 | None,
) -> str | None:
    query = (
        alignment_refetch_request.refined_evidence_query
        if alignment_refetch_request is not None
        and alignment_refetch_request.unit_id == unit_id
        and alignment_refetch_request.manifest_ref == unit_manifest_ref
        else unit_evidence_query
    )
    if query is None or not query.strip():
        return None
    return query


def document_cache_authority(
    *,
    state_key_ref: str | None,
    document_cache_config: DocumentMcpCacheConfigV1 | None,
    policy: PolicyManifestV2,
    manifest_ref: str,
) -> DocumentMcpCacheAuthorityV1 | None:
    if state_key_ref is None or document_cache_config is None:
        return None
    validated_ref = validated_state_key_ref(state_key_ref)
    if validated_ref is None:
        return None
    return DocumentMcpCacheAuthorityV1(
        state_key_ref=validated_ref,
        policy_id=policy.policy_id,
        manifest_ref=manifest_ref,
        business_scope_hash=policy.business_scope_hash,
        source_cache_config=document_cache_config,
    )


def deduplicate_knowledge_result(
    facts: list[CanonicalEvidenceFactV1],
    bindings: list[knowledge_models.RegisteredMediaBindingV1],
) -> knowledge_models.InternalCompanyKnowledgeResultV1:
    fact_by_id = {fact.evidence_id: fact for fact in facts}
    binding_by_ref = {
        (binding.evidence_id, binding.asset_id): binding
        for binding in bindings
        if binding.evidence_id in fact_by_id
    }
    return knowledge_models.InternalCompanyKnowledgeResultV1(
        facts=tuple(fact_by_id.values()),
        media_bindings=tuple(binding_by_ref.values()),
    )
