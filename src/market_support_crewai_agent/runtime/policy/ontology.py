from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Final

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy import ontology_models
from market_support_crewai_agent.runtime.policy.ontology_adapter import (
    artifact_from_adapter_result,
    build_artifact,
    build_strategy,
    products_from_fact,
)
from market_support_crewai_agent.runtime.policy.ontology_inputs import (
    artifact_map,
    available_artifact_payloads,
    channel_from_payload,
    clean,
    is_artifact_type,
    material_pack_options_from_available,
    nested_runtime_items,
    payload_dict,
    product_map,
    strategy_map,
    string_key_object_mapping,
    time_range_from_metadata,
)
from market_support_crewai_agent.schemas.adapter import AdapterResolveResult

if TYPE_CHECKING:
    from market_support_crewai_agent.runtime.evidence.scope_authority import (
        BusinessScopeAuthorityV1,
    )

_ARTIFACT_TYPES_BY_RESOLVE: Final[dict[str, ontology_models.ArtifactType]] = {
    "material_pack": "material_pack",
    "weekly_report": "weekly_report",
    "monthly_report": "monthly_report",
    "sales_mention": "adapter_context",
}


class DomainContextV1Builder:
    def build(
        self,
        adapter_channel_payload: KernelReplyRequestV1 | Mapping[str, object] | None,
        available_artifacts: Sequence[object] | None = None,
        conversation_metadata: Mapping[str, object] | None = None,
        base_context: ontology_models.DomainContextV1 | None = None,
        scope_authority: BusinessScopeAuthorityV1 | None = None,
    ) -> ontology_models.DomainContextV1:
        if (
            scope_authority is not None
            and isinstance(adapter_channel_payload, KernelReplyRequestV1)
            and adapter_channel_payload.business_scope != scope_authority.scope
        ):
            raise ValueError("business_scope_authority_mismatch")
        payload = payload_dict(adapter_channel_payload)
        channel = channel_from_payload(payload, base_context)
        if scope_authority is not None and scope_authority.scope.kind == "distribution":
            channel = replace(channel, id=scope_authority.business_scope_ref)
        strategies = strategy_map(base_context)
        products = product_map(base_context)
        artifacts = artifact_map(base_context)

        for available in available_artifact_payloads(payload):
            artifact_type_text = str(available.get("type") or "unknown")
            scope = ontology_models.ArtifactScope(
                channel_id=channel.id,
                source_id=("adapter_channel.available_artifacts:" + artifact_type_text),
                provenance="adapter_channel_payload",
            )
            artifact = build_artifact(
                artifact_type=normalize_artifact_type(artifact_type_text),
                scope=scope,
                title=artifact_type_text,
                source_type="adapter_channel_payload",
                fact_types=(),
            )
            _ = artifacts.setdefault(artifact.id, artifact)

        for item in available_artifacts or ():
            for artifact, new_products in self._artifacts_from_runtime_item(
                item,
                channel,
                strategies,
            ):
                for product in new_products:
                    _ = products.setdefault(product.id, product)
                _ = artifacts.setdefault(artifact.id, artifact)

        metadata = dict(base_context.metadata) if base_context is not None else {}
        material_pack_options = material_pack_options_from_available(payload)
        if material_pack_options:
            metadata["material_pack_options"] = material_pack_options
        metadata.update(
            {
                str(key): value
                for key, value in (conversation_metadata or {}).items()
                if value is not None
            }
        )
        return ontology_models.DomainContextV1(
            channel=channel,
            strategies=tuple(strategies.values()),
            products=tuple(products.values()),
            artifacts=tuple(artifacts.values()),
            metadata=metadata,
        )

    def _artifacts_from_runtime_item(
        self,
        item: object,
        channel: ontology_models.DistributionChannel,
        strategies: dict[str, ontology_models.Strategy],
    ) -> list[tuple[ontology_models.Artifact, tuple[ontology_models.Product, ...]]]:
        nested_items = nested_runtime_items(item)
        if isinstance(nested_items, list):
            output: list[
                tuple[ontology_models.Artifact, tuple[ontology_models.Product, ...]]
            ] = []
            for nested in nested_items:
                output.extend(
                    self._artifacts_from_runtime_item(nested, channel, strategies)
                )
            return output
        result = getattr(item, "result", None)
        if isinstance(result, AdapterResolveResult):
            return [_artifact_from_adapter_result(result, channel, strategies)]
        if isinstance(item, AdapterResolveResult):
            return [_artifact_from_adapter_result(item, channel, strategies)]

        fact_type = clean(getattr(item, "fact_type", ""))
        if not fact_type:
            return []
        artifact_type = normalize_artifact_type(
            getattr(item, "artifact_type", None),
            resolve_type=getattr(item, "resolve_type", None),
            source_type=getattr(item, "source_type", None),
            fact_type=fact_type,
        )
        metadata = string_key_object_mapping(getattr(item, "metadata", {}))
        scope = getattr(item, "scope", None)
        if (
            not isinstance(scope, ontology_models.ArtifactScope)
            or scope.channel_id != channel.id
        ):
            scope = artifact_scope_for_evidence(
                channel_id=channel.id,
                artifact_type=artifact_type,
                resolve_type=getattr(item, "resolve_type", None),
                source_id=getattr(item, "source_id", ""),
                source_type=getattr(item, "source_type", ""),
                metadata=metadata,
                strategies=strategies,
            )
        fact_products = products_from_fact(item, channel, scope)
        product_ids = tuple(product.id for product in fact_products)
        if product_ids and not scope.product_ids:
            scope = replace(scope, product_ids=product_ids)
        artifact = build_artifact(
            artifact_type=artifact_type,
            scope=scope,
            title=str(getattr(item, "source_id", "") or fact_type),
            source_type=str(getattr(item, "source_type", "") or ""),
            fact_types=(fact_type,),
        )
        return [(artifact, fact_products)]


def artifact_scope_for_evidence(
    *,
    channel_id: str,
    artifact_type: ontology_models.ArtifactType = "unknown",
    resolve_type: object | None = None,
    source_id: object | None = None,
    source_type: object | None = None,
    metadata: Mapping[str, object] | None = None,
    strategies: dict[str, ontology_models.Strategy] | None = None,
) -> ontology_models.ArtifactScope:
    metadata = metadata or {}
    strategy = clean(metadata.get("strategy"))
    strategy_id = None
    if strategy:
        known = build_strategy(
            channel_id,
            strategy,
            source_id=f"{source_id or resolve_type or artifact_type}:strategy",
            provenance=str(source_type or "evidence_metadata"),
        )
        if strategies is not None:
            known = strategies.setdefault(known.id, known)
        strategy_id = known.id
    source = str(source_id or resolve_type or artifact_type or "unknown")
    return ontology_models.ArtifactScope(
        channel_id=channel_id,
        strategy_id=strategy_id,
        product_ids=(),
        time_range=time_range_from_metadata(metadata),
        source_id=source,
        provenance=str(source_type or "unknown"),
    )


def normalize_artifact_type(
    value: object | None = None,
    *,
    resolve_type: object | None = None,
    source_type: object | None = None,
    fact_type: object | None = None,
) -> ontology_models.ArtifactType:
    text = clean(value)
    if is_artifact_type(text) and text != "unknown":
        return text
    resolve = clean(resolve_type)
    if resolve in _ARTIFACT_TYPES_BY_RESOLVE:
        return _ARTIFACT_TYPES_BY_RESOLVE[resolve]
    fact = clean(fact_type)
    if fact in {"document_context", "document_context_unavailable"}:
        return "document_context"
    source = clean(source_type)
    if source == "action_ledger":
        return "history"
    if source in {"document_mcp", "approved_static_knowledge"}:
        return "document_context"
    if source == "adapter_resolve":
        return (
            "adapter_context"
            if not resolve
            else normalize_artifact_type(resolve_type=resolve)
        )
    return "unknown"


def _artifact_from_adapter_result(
    result: AdapterResolveResult,
    channel: ontology_models.DistributionChannel,
    strategies: dict[str, ontology_models.Strategy],
) -> tuple[ontology_models.Artifact, tuple[ontology_models.Product, ...]]:
    artifact_type = normalize_artifact_type(resolve_type=result.resolve_type)
    scope = artifact_scope_for_evidence(
        channel_id=channel.id,
        artifact_type=artifact_type,
        resolve_type=result.resolve_type,
        source_id=result.resolve_type,
        source_type="adapter_resolve",
        metadata=result.model_dump(mode="json", exclude_none=True),
        strategies=strategies,
    )
    return artifact_from_adapter_result(
        result,
        artifact_type=artifact_type,
        scope=scope,
    )
