from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

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

JsonDict: TypeAlias = dict[str, object]


@dataclass(frozen=True)
class TimeRange:
    period: str | None = None
    start: str | None = None
    end: str | None = None
    label: str | None = None

    def to_prompt_dict(self) -> JsonDict:
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

    def to_prompt_dict(self) -> JsonDict:
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

    def to_prompt_dict(self) -> JsonDict:
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

    def to_prompt_dict(self) -> JsonDict:
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

    def to_prompt_dict(self) -> JsonDict:
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

    def to_prompt_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "artifact_type": self.artifact_type,
            "title": self.title,
            "source_type": self.source_type,
            "fact_types": list(self.fact_types),
            "scope": self.scope.to_prompt_dict(),
        }


@dataclass(frozen=True)
class DomainContextV1:
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
        normalized = str(name or "").strip()
        if not normalized:
            return None
        for strategy in self.strategies:
            if strategy.name.strip() == normalized:
                return strategy
        return None

    def to_prompt_dict(self) -> JsonDict:
        return {
            "channel": self.channel.to_prompt_dict(),
            "strategies": [strategy.to_prompt_dict() for strategy in self.strategies],
            "products": [product.to_prompt_dict() for product in self.products],
            "artifacts": [artifact.to_prompt_dict() for artifact in self.artifacts],
            "metadata": dict(self.metadata),
        }
