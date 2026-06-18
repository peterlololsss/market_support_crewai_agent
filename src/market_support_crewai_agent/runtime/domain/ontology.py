from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from market_support_crewai_agent.schemas import AdapterResolveResult, ReplyRequest

ChannelKind = Literal["bank", "non_bank", "unknown"]
ArtifactType = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "document_context",
    "adapter_context",
    "history",
    "user_upload",
    "unknown",
]

_ARTIFACT_TYPES_BY_RESOLVE = {
    "material_pack": "material_pack",
    "weekly_report": "weekly_report",
    "monthly_report": "monthly_report",
    "sales_mention": "adapter_context",
}
_ARTIFACT_TYPES_BY_MATERIAL = {
    "material": "material_pack",
    "weekly": "weekly_report",
    "monthly": "monthly_report",
}


@dataclass(frozen=True)
class TimeRange:
    period: str | None = None
    start: str | None = None
    end: str | None = None
    label: str | None = None

    def to_prompt_dict(self) -> dict:
        return {
            key: value
            for key, value in {
                "period": self.period,
                "start": self.start,
                "end": self.end,
                "label": self.label,
            }.items()
            if value
        }


@dataclass(frozen=True)
class DistributionChannel:
    id: str
    name: str
    kind: ChannelKind = "unknown"
    source_id: str = ""
    provenance: str = "unknown"

    def to_prompt_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "source_id": self.source_id,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    channel_id: str
    source_id: str = ""
    provenance: str = "unknown"

    def to_prompt_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "channel_id": self.channel_id,
            "source_id": self.source_id,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    channel_id: str
    strategy_ids: tuple[str, ...] = ()
    source_id: str = ""
    provenance: str = "unknown"

    def to_prompt_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "channel_id": self.channel_id,
            "strategy_ids": list(self.strategy_ids),
            "source_id": self.source_id,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ArtifactScope:
    channel_id: str
    strategy_id: str | None = None
    product_ids: tuple[str, ...] = ()
    time_range: TimeRange | None = None
    source_id: str = ""
    provenance: str = "unknown"

    def to_prompt_dict(self) -> dict:
        return {
            "channel_id": self.channel_id,
            "strategy_id": self.strategy_id,
            "product_ids": list(self.product_ids),
            "time_range": (
                self.time_range.to_prompt_dict()
                if self.time_range is not None
                else None
            ),
            "source_id": self.source_id,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Artifact:
    id: str
    artifact_type: ArtifactType
    scope: ArtifactScope
    title: str = ""
    source_type: str = ""
    fact_types: tuple[str, ...] = ()

    def to_prompt_dict(self) -> dict:
        return {
            "id": self.id,
            "artifact_type": self.artifact_type,
            "title": self.title,
            "source_type": self.source_type,
            "fact_types": list(self.fact_types),
            "scope": self.scope.to_prompt_dict(),
        }


@dataclass(frozen=True)
class DomainContext:
    channel: DistributionChannel
    strategies: tuple[Strategy, ...] = ()
    products: tuple[Product, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def channel_kind(self) -> ChannelKind:
        return self.channel.kind

    def artifacts_by_type(self, artifact_type: ArtifactType) -> tuple[Artifact, ...]:
        return tuple(
            artifact
            for artifact in self.artifacts
            if artifact.artifact_type == artifact_type
        )

    def strategy_by_name(self, name: str | None) -> Strategy | None:
        normalized = _clean(name)
        if not normalized:
            return None
        for strategy in self.strategies:
            if _clean(strategy.name) == normalized:
                return strategy
        return None

    def to_prompt_dict(self) -> dict:
        return {
            "channel": self.channel.to_prompt_dict(),
            "strategies": [
                strategy.to_prompt_dict()
                for strategy in self.strategies
            ],
            "products": [
                product.to_prompt_dict()
                for product in self.products
            ],
            "artifacts": [
                artifact.to_prompt_dict()
                for artifact in self.artifacts
            ],
            "metadata": dict(self.metadata),
        }


class DomainContextBuilder:
    def build(
        self,
        adapter_channel_payload: ReplyRequest | Mapping[str, object] | None,
        available_artifacts: Sequence[object] | None = None,
        conversation_metadata: Mapping[str, object] | None = None,
        base_context: DomainContext | None = None,
    ) -> DomainContext:
        payload = _payload_dict(adapter_channel_payload)
        channel = _channel_from_payload(payload, base_context)
        strategies = _strategy_map(base_context)
        products = _product_map(base_context)
        artifacts = _artifact_map(base_context)

        for material_type in _string_list(payload.get("available_materials")):
            artifact_type = _ARTIFACT_TYPES_BY_MATERIAL.get(material_type, "unknown")
            scope = ArtifactScope(
                channel_id=channel.id,
                source_id=f"adapter_channel.available_materials:{material_type}",
                provenance="adapter_channel_payload",
            )
            artifact = _artifact(
                artifact_type=artifact_type,  # type: ignore[arg-type]
                scope=scope,
                title=material_type,
                source_type="adapter_channel_payload",
                fact_types=(),
            )
            artifacts.setdefault(artifact.id, artifact)

        for item in available_artifacts or ():
            for artifact, new_products in self._artifacts_from_runtime_item(
                item,
                channel,
                strategies,
            ):
                for product in new_products:
                    products.setdefault(product.id, product)
                artifacts.setdefault(artifact.id, artifact)

        metadata = dict(base_context.metadata) if base_context is not None else {}
        material_pack_options = _string_list(payload.get("material_pack_options"))
        if material_pack_options:
            metadata["material_pack_options"] = material_pack_options
        metadata.update(
            {
                str(key): value
                for key, value in (conversation_metadata or {}).items()
                if value is not None
            }
        )
        return DomainContext(
            channel=channel,
            strategies=tuple(strategies.values()),
            products=tuple(products.values()),
            artifacts=tuple(artifacts.values()),
            metadata=metadata,
        )

    def _artifacts_from_runtime_item(
        self,
        item: object,
        channel: DistributionChannel,
        strategies: dict[str, Strategy],
    ) -> list[tuple[Artifact, tuple[Product, ...]]]:
        if hasattr(item, "items") and isinstance(getattr(item, "items"), list):
            output: list[tuple[Artifact, tuple[Product, ...]]] = []
            for nested in getattr(item, "items"):
                output.extend(self._artifacts_from_runtime_item(nested, channel, strategies))
            return output

        result = getattr(item, "result", None)
        if isinstance(result, AdapterResolveResult):
            return [_artifact_from_adapter_result(result, channel, strategies)]

        if isinstance(item, AdapterResolveResult):
            return [_artifact_from_adapter_result(item, channel, strategies)]

        fact_type = _clean(getattr(item, "fact_type", ""))
        if not fact_type:
            return []
        artifact_type = normalize_artifact_type(
            getattr(item, "artifact_type", None),
            resolve_type=getattr(item, "resolve_type", None),
            source_type=getattr(item, "source_type", None),
            fact_type=fact_type,
        )
        metadata = getattr(item, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        scope = getattr(item, "scope", None)
        if not isinstance(scope, ArtifactScope) or scope.channel_id != channel.id:
            scope = artifact_scope_for_evidence(
                channel_id=channel.id,
                artifact_type=artifact_type,
                resolve_type=getattr(item, "resolve_type", None),
                source_id=getattr(item, "source_id", ""),
                source_type=getattr(item, "source_type", ""),
                metadata=metadata,
                strategies=strategies,
            )
        products = _products_from_fact(item, channel, scope, strategies)
        product_ids = tuple(product.id for product in products)
        if product_ids and not scope.product_ids:
            scope = ArtifactScope(
                channel_id=scope.channel_id,
                strategy_id=scope.strategy_id,
                product_ids=product_ids,
                time_range=scope.time_range,
                source_id=scope.source_id,
                provenance=scope.provenance,
            )
        artifact = _artifact(
            artifact_type=artifact_type,
            scope=scope,
            title=str(getattr(item, "source_id", "") or fact_type),
            source_type=str(getattr(item, "source_type", "") or ""),
            fact_types=(fact_type,),
        )
        return [(artifact, products)]


def artifact_scope_for_evidence(
    *,
    channel_id: str,
    artifact_type: ArtifactType = "unknown",
    resolve_type: object | None = None,
    source_id: object | None = None,
    source_type: object | None = None,
    metadata: Mapping[str, object] | None = None,
    strategies: Mapping[str, Strategy] | None = None,
) -> ArtifactScope:
    metadata = metadata or {}
    strategy = _clean(metadata.get("strategy"))
    strategy_id = None
    if strategy:
        known = _strategy(
            channel_id,
            strategy,
            source_id=f"{source_id or resolve_type or artifact_type}:strategy",
            provenance=str(source_type or "evidence_metadata"),
        )
        if strategies is not None:
            known = strategies.setdefault(known.id, known)
        strategy_id = known.id

    source = str(source_id or resolve_type or artifact_type or "unknown")
    return ArtifactScope(
        channel_id=channel_id,
        strategy_id=strategy_id,
        product_ids=(),
        time_range=_time_range_from_metadata(metadata),
        source_id=source,
        provenance=str(source_type or "unknown"),
    )


def normalize_artifact_type(
    value: object | None = None,
    *,
    resolve_type: object | None = None,
    source_type: object | None = None,
    fact_type: object | None = None,
) -> ArtifactType:
    text = _clean(value)
    if text in _valid_artifact_types() and text != "unknown":
        return text  # type: ignore[return-value]
    resolve = _clean(resolve_type)
    if resolve in _ARTIFACT_TYPES_BY_RESOLVE:
        return _ARTIFACT_TYPES_BY_RESOLVE[resolve]  # type: ignore[return-value]
    fact = _clean(fact_type)
    if fact in {"document_context", "document_context_unavailable"}:
        return "document_context"
    if fact.startswith("material_pack_") or fact == "material_pack_resolvable":
        return "material_pack"
    if fact.startswith("weekly_report_") or (
        resolve == "weekly_report"
        and fact.startswith("report_")
    ):
        return "weekly_report"
    if fact.startswith("monthly_report_") or (
        resolve == "monthly_report"
        and fact.startswith("report_")
    ):
        return "monthly_report"
    source = _clean(source_type)
    if source == "action_ledger":
        return "history"
    if source in {"document_mcp", "approved_static_knowledge"}:
        return "document_context"
    if source == "adapter_resolve":
        return "adapter_context" if not resolve else normalize_artifact_type(resolve_type=resolve)
    return "unknown"


def _artifact_from_adapter_result(
    result: AdapterResolveResult,
    channel: DistributionChannel,
    strategies: dict[str, Strategy],
) -> tuple[Artifact, tuple[Product, ...]]:
    artifact_type = normalize_artifact_type(resolve_type=result.resolve_type)
    metadata = result.model_dump(mode="json", exclude_none=True)
    scope = artifact_scope_for_evidence(
        channel_id=channel.id,
        artifact_type=artifact_type,
        resolve_type=result.resolve_type,
        source_id=result.resolve_type,
        source_type="adapter_resolve",
        metadata=metadata,
        strategies=strategies,
    )
    artifact = _artifact(
        artifact_type=artifact_type,
        scope=scope,
        title=result.display_name,
        source_type="adapter_resolve",
        fact_types=(f"{result.resolve_type}_resolvable",),
    )
    return artifact, ()


def _products_from_fact(
    fact: object,
    channel: DistributionChannel,
    scope: ArtifactScope,
    strategies: dict[str, Strategy],
) -> tuple[Product, ...]:
    source_metadata = getattr(fact, "source_metadata", None)
    if (
        getattr(source_metadata, "evidence_allowed_by_default", True) is False
        or getattr(source_metadata, "artifact_type", "") == "history"
        or getattr(source_metadata, "source_type", "")
        in {"user_message", "assistant_message", "history_summary"}
    ):
        return ()
    metadata = getattr(fact, "metadata", {})
    if not isinstance(metadata, Mapping):
        return ()
    raw_products = metadata.get("products")
    if not isinstance(raw_products, Sequence) or isinstance(raw_products, str):
        return ()
    products: list[Product] = []
    strategy_ids = (scope.strategy_id,) if scope.strategy_id else ()
    for item in raw_products:
        name = _product_name(item)
        if not name:
            continue
        product = Product(
            id=_stable_id("product", channel.id, name),
            name=name,
            channel_id=channel.id,
            strategy_ids=strategy_ids,
            source_id=str(getattr(fact, "source_id", "") or ""),
            provenance=str(getattr(fact, "source_type", "") or "unknown"),
        )
        products.append(product)
        for strategy_id in strategy_ids:
            if strategy_id not in strategies:
                continue
    return tuple(_unique_by_id(products))


def _product_name(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("product_name", "name", "display_name"):
            text = _clean(value.get(key))
            if text:
                return text
    return _clean(value)


def _channel_from_payload(
    payload: Mapping[str, object],
    base_context: DomainContext | None,
) -> DistributionChannel:
    if base_context is not None and not payload:
        return base_context.channel
    name = (
        _clean(payload.get("dist_channel_name"))
        or _clean(payload.get("display_name"))
        or _clean(payload.get("channel_name"))
        or _clean(payload.get("name"))
        or (base_context.channel.name if base_context is not None else "unknown")
    )
    kind = _channel_kind(payload.get("channel_type") or payload.get("kind"))
    if kind == "unknown" and base_context is not None:
        kind = base_context.channel.kind
    return DistributionChannel(
        id=_stable_id("channel", name, kind),
        name=name,
        kind=kind,
        source_id=str(payload.get("context_id") or payload.get("source_id") or "adapter_channel"),
        provenance="adapter_channel_payload" if payload else "unknown",
    )


def _strategy(
    channel_id: str,
    name: str,
    *,
    source_id: str,
    provenance: str,
) -> Strategy:
    return Strategy(
        id=_stable_id("strategy", channel_id, name),
        name=name,
        channel_id=channel_id,
        source_id=source_id,
        provenance=provenance,
    )


def _artifact(
    *,
    artifact_type: ArtifactType,
    scope: ArtifactScope,
    title: str,
    source_type: str,
    fact_types: tuple[str, ...],
) -> Artifact:
    return Artifact(
        id=_stable_id(
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


def _time_range_from_metadata(metadata: Mapping[str, object]) -> TimeRange | None:
    time_range = TimeRange(
        period=_optional_clean(metadata.get("period")),
        start=_optional_clean(metadata.get("period_start")),
        end=_optional_clean(metadata.get("period_end")),
        label=_optional_clean(metadata.get("period_label")),
    )
    return time_range if time_range.to_prompt_dict() else None


def _payload_dict(value: ReplyRequest | Mapping[str, object] | None) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, ReplyRequest):
        return value.model_dump(mode="json", exclude_none=True)
    return {str(key): item for key, item in value.items()}


def _channel_kind(value: object) -> ChannelKind:
    text = _clean(value)
    if text in {"bank", "non_bank"}:
        return text  # type: ignore[return-value]
    return "unknown"


def _strategy_map(base_context: DomainContext | None) -> dict[str, Strategy]:
    return {
        strategy.id: strategy
        for strategy in (base_context.strategies if base_context is not None else ())
    }


def _product_map(base_context: DomainContext | None) -> dict[str, Product]:
    return {
        product.id: product
        for product in (base_context.products if base_context is not None else ())
    }


def _artifact_map(base_context: DomainContext | None) -> dict[str, Artifact]:
    return {
        artifact.id: artifact
        for artifact in (base_context.artifacts if base_context is not None else ())
    }


def _string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_clean(value),) if _clean(value) else ()
    if isinstance(value, Sequence):
        return tuple(
            item
            for item in (_clean(nested) for nested in value)
            if item
        )
    return ()


def _optional_clean(value: object) -> str | None:
    text = _clean(value)
    return text or None


def _clean(value: object) -> str:
    return str(value or "").strip()


def _valid_artifact_types() -> set[str]:
    return {
        "material_pack",
        "weekly_report",
        "monthly_report",
        "document_context",
        "adapter_context",
        "history",
        "user_upload",
        "unknown",
    }


def _unique_by_id(values):
    seen: set[str] = set()
    output = []
    for value in values:
        if value.id in seen:
            continue
        seen.add(value.id)
        output.append(value)
    return tuple(output)


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"
