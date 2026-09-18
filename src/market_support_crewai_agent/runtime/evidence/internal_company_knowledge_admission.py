from __future__ import annotations

import re
from typing import Final, assert_never

from pydantic import ValidationError

from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    ApprovedStaticEvidenceSourceRecordKeyV1,
    DistributionEvidenceScopeIdentityV1,
    DocumentMcpEvidenceSourceRecordKeyV1,
    EvidenceProvenanceCanonicalV1,
    UnscopedEvidenceScopeIdentityV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
    build_canonical_evidence_fact_v1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    EvidenceContentValueV1,
    ScalarEvidencePayloadCanonicalV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
    RegisteredMediaBindingV1,
)
from market_support_crewai_agent.runtime.hashing import (
    evidence_scope_ref,
    evidence_source_record_ref,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.sanitizer import (
    sanitize_document_text_for_evidence,
)
from market_support_crewai_agent.runtime.planning.models import (
    DistributionExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    UnscopedExecutionDomainScopeV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.recall.approved_static_catalog import (
    _APPROVED_IMAGE_ASSETS_BY_ID,
    _APPROVED_KNOWLEDGE_BY_ID,
    APPROVED_STATIC_MANIFEST_REF,
    ApprovedKnowledgeEntry,
    validated_catalog_manifest_ref,
)

_CATALOG_VERSION: Final = "approved-static-catalog.v1"
_GATEWAY_VERSION: Final = "internal-company-knowledge-gateway.v1"
_MARKER_PATTERN: Final = re.compile(r"%%([^%\r\n]{1,160})%%")


def document_fact(
    context: GatewayDocumentContextV1,
    unit: ExecutionPlanUnitV2,
) -> CanonicalEvidenceFactV1 | None:
    text = sanitize_document_text_for_evidence(context.text).text
    if not text:
        return None
    scope = _document_scope(unit)
    source = DocumentMcpEvidenceSourceRecordKeyV1(
        client_contract_version=context.client_contract_version,
        corpus_version=context.corpus_version,
        document_id=context.document_id,
        entry_id=context.document_id,
    )
    provenance = EvidenceProvenanceCanonicalV1(
        source_class="document_mcp",
        source_record_ref=evidence_source_record_ref(source),
        producer_contract_version=_GATEWAY_VERSION,
        retrieval_operation="document_query",
        scope_ref=evidence_scope_ref(scope),
        as_of_epoch_seconds=None,
        public_url_hashes=(),
    )
    return build_canonical_evidence_fact_v1(
        fact_type="document_context",
        source_type="document_mcp",
        artifact_type="document_context",
        resolve_type=None,
        payload=ScalarEvidencePayloadCanonicalV1(
            value=EvidenceContentValueV1(media_type="text/markdown", text=text)
        ),
        scope=scope,
        provenance=provenance,
        observed_at_epoch_seconds=None,
    )


def static_fact(
    context: GatewayStaticContextV1,
    unit: ExecutionPlanUnitV2,
    *,
    media_allowed: bool = True,
) -> tuple[CanonicalEvidenceFactV1 | None, tuple[RegisteredMediaBindingV1, ...]]:
    if not media_allowed and (context.selected_asset_ids or "%%" in context.text):
        return None, ()
    entry = _APPROVED_KNOWLEDGE_BY_ID.get(context.entry_id)
    if entry is None or not _static_manifest_is_admitted(context, entry, unit):
        return None, ()
    if not _static_assets_are_selected(context, entry.image_asset_ids):
        return None, ()
    if _selected_markers(context.text, context.selected_asset_ids) is None:
        return None, ()
    sanitized = sanitize_document_text_for_evidence(context.text).text
    if (
        not sanitized
        or _selected_markers(sanitized, context.selected_asset_ids) is None
    ):
        return None, ()

    scope = _document_scope(unit)
    source = ApprovedStaticEvidenceSourceRecordKeyV1(
        catalog_version=_CATALOG_VERSION,
        canonical_id=context.entry_id,
    )
    provenance = EvidenceProvenanceCanonicalV1(
        source_class="approved_static",
        source_record_ref=evidence_source_record_ref(source),
        producer_contract_version=_GATEWAY_VERSION,
        retrieval_operation="approved_static_query",
        scope_ref=evidence_scope_ref(scope),
        as_of_epoch_seconds=None,
        public_url_hashes=(),
    )
    fact = build_canonical_evidence_fact_v1(
        fact_type="document_context",
        source_type="approved_static_knowledge",
        artifact_type="document_context",
        resolve_type=None,
        payload=ScalarEvidencePayloadCanonicalV1(
            value=EvidenceContentValueV1(media_type="text/markdown", text=sanitized)
        ),
        scope=scope,
        provenance=provenance,
        observed_at_epoch_seconds=None,
    )
    bindings = tuple(
        RegisteredMediaBindingV1(
            evidence_id=fact.evidence_id,
            asset_id=asset_id,
            marker=_APPROVED_IMAGE_ASSETS_BY_ID[asset_id].marker,
        )
        for asset_id in context.selected_asset_ids
    )
    return fact, bindings


def _static_manifest_is_admitted(
    context: GatewayStaticContextV1,
    entry: ApprovedKnowledgeEntry,
    unit: ExecutionPlanUnitV2,
) -> bool:
    context_ref = _validated_manifest_ref(getattr(context, "manifest_ref", None))
    entry_ref = validated_catalog_manifest_ref(entry)
    if context_ref is None or entry_ref is None:
        return False
    if (
        context_ref != APPROVED_STATIC_MANIFEST_REF
        or entry_ref != context_ref
        or unit.manifest_ref != context_ref
    ):
        return False
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(context_ref.manifest_id)
    if manifest is None or manifest.manifest_version != context_ref.manifest_version:
        return False
    contract = manifest.evidence_contract
    return (
        "document_context" in contract.allowed_fact_types
        and "document_context" in contract.allowed_artifact_types
        and "approved_static_knowledge" in contract.allowed_source_types
    )


def _validated_manifest_ref(value: ManifestRefV1 | None) -> ManifestRefV1 | None:
    if value is None:
        return None
    try:
        payload = value.model_dump(mode="json", exclude_none=False)
    except AttributeError:
        return None
    try:
        return ManifestRefV1.model_validate(payload)
    except ValidationError:
        return None


def _static_assets_are_selected(
    context: GatewayStaticContextV1,
    entry_asset_ids: tuple[str, ...],
) -> bool:
    return (
        len(set(context.selected_asset_ids)) == len(context.selected_asset_ids)
        and set(context.selected_asset_ids) <= set(entry_asset_ids)
        and all(
            asset_id in _APPROVED_IMAGE_ASSETS_BY_ID
            for asset_id in context.selected_asset_ids
        )
    )


def _selected_markers(
    text: str,
    selected_asset_ids: tuple[str, ...],
) -> frozenset[str] | None:
    selected = frozenset(
        _APPROVED_IMAGE_ASSETS_BY_ID[asset_id].marker for asset_id in selected_asset_ids
    )
    observed: set[str] = set()
    cursor = 0
    while (start := text.find("%%", cursor)) >= 0:
        match = _MARKER_PATTERN.match(text, start)
        if match is None:
            return None
        end = match.end()
        if (start > 0 and text[start - 1] == "%") or (
            end < len(text) and text[end] == "%"
        ):
            return None
        marker = match.group(0)
        if marker not in selected:
            return None
        observed.add(marker)
        cursor = end
    if observed != selected:
        return None
    return frozenset(observed)


def _document_scope(
    unit: ExecutionPlanUnitV2,
) -> DistributionEvidenceScopeIdentityV1 | UnscopedEvidenceScopeIdentityV1:
    match unit.scope:
        case DistributionExecutionDomainScopeV2(
            business_scope_ref=business_scope_ref,
            channel_kind=channel_kind,
            product_ids=product_ids,
            material_pack_option=material_pack_option,
            time_range=time_range,
        ):
            return DistributionEvidenceScopeIdentityV1(
                business_scope_ref=business_scope_ref,
                channel_kind=channel_kind,
                product_ids=product_ids,
                artifact_type="document_context",
                material_option=material_pack_option,
                period=time_range.period if time_range is not None else None,
            )
        case UnscopedExecutionDomainScopeV2():
            return UnscopedEvidenceScopeIdentityV1(artifact_type="document_context")
        case unreachable:
            assert_never(unreachable)
