from __future__ import annotations

from collections.abc import Sequence

from market_support_crewai_agent.runtime.policy.ontology_inputs import (
    clean,
    is_object_mapping,
    stable_id,
    string_key_object_mapping,
)
from market_support_crewai_agent.runtime.policy.ontology_models import (
    Artifact,
    ArtifactScope,
    ArtifactType,
    DistributionChannel,
    Product,
    Strategy,
)
from market_support_crewai_agent.schemas.adapter import AdapterResolveResult


def build_strategy(
    channel_id: str,
    name: str,
    *,
    source_id: str,
    provenance: str,
) -> Strategy:
    return Strategy(
        id=stable_id("strategy", channel_id, name),
        name=name,
        channel_id=channel_id,
        source_id=source_id,
        provenance=provenance,
    )


def build_artifact(
    *,
    artifact_type: ArtifactType,
    scope: ArtifactScope,
    title: str,
    source_type: str,
    fact_types: tuple[str, ...],
) -> Artifact:
    return Artifact(
        id=stable_id(
            "artifact",
            artifact_type,
            scope.channel_id,
            scope.strategy_id or "",
            ",".join(scope.product_ids),
            scope.source_id,
            ",".join(fact_types),
        ),
        artifact_type=artifact_type,
        scope=scope,
        title=title,
        source_type=source_type,
        fact_types=fact_types,
    )


def artifact_from_adapter_result(
    result: AdapterResolveResult,
    *,
    artifact_type: ArtifactType,
    scope: ArtifactScope,
) -> tuple[Artifact, tuple[Product, ...]]:
    return (
        build_artifact(
            artifact_type=artifact_type,
            scope=scope,
            title=result.display_name,
            source_type="adapter_resolve",
            fact_types=(f"{result.resolve_type}_resolvable",),
        ),
        (),
    )


def products_from_fact(
    fact: object,
    channel: DistributionChannel,
    scope: ArtifactScope,
) -> tuple[Product, ...]:
    source_metadata = getattr(fact, "source_metadata", None)
    if (
        getattr(source_metadata, "evidence_allowed_by_default", True) is False
        or getattr(source_metadata, "artifact_type", "") == "history"
        or getattr(source_metadata, "source_type", "")
        in {"user_message", "assistant_message", "history_summary"}
    ):
        return ()
    metadata = string_key_object_mapping(getattr(fact, "metadata", {}))
    raw_products = metadata.get("products")
    if not isinstance(raw_products, Sequence) or isinstance(raw_products, str):
        return ()
    strategy_ids = (scope.strategy_id,) if scope.strategy_id else ()
    products = tuple(
        Product(
            id=stable_id("product", channel.id, name),
            name=name,
            channel_id=channel.id,
            strategy_ids=strategy_ids,
            source_id=str(getattr(fact, "source_id", "") or ""),
            provenance=str(getattr(fact, "source_type", "") or "unknown"),
        )
        for item in raw_products
        if (name := _product_name(item))
    )
    return _unique_by_id(products)


def _product_name(value: object) -> str:
    if is_object_mapping(value):
        metadata = string_key_object_mapping(value)
        for key in ("product_name", "name", "display_name"):
            text = clean(metadata.get(key))
            if text:
                return text
    return clean(value)


def _unique_by_id(values: Sequence[Product]) -> tuple[Product, ...]:
    seen: set[str] = set()
    output: list[Product] = []
    for value in values:
        if value.id in seen:
            continue
        seen.add(value.id)
        output.append(value)
    return tuple(output)
