from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias, TypeGuard

from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.ontology_models import (
    Artifact,
    ArtifactType,
    ChannelKind,
    DistributionChannel,
    DomainContextV1,
    JsonDict,
    Product,
    Strategy,
    TimeRange,
)

PayloadJsonDict: TypeAlias = dict[str, JsonValue]

_JSON_OBJECT_ADAPTER: Final[TypeAdapter[PayloadJsonDict]] = TypeAdapter(
    dict[str, JsonValue],
)
_VALID_ARTIFACT_TYPES: Final[tuple[ArtifactType, ...]] = (
    "material_pack",
    "weekly_report",
    "monthly_report",
    "document_context",
    "adapter_context",
    "history",
    "user_upload",
    "unknown",
)


def clean(value: object) -> str:
    return str(value or "").strip()


def is_artifact_type(value: str) -> TypeGuard[ArtifactType]:
    return value in _VALID_ARTIFACT_TYPES


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def payload_dict(
    value: KernelReplyRequestV1 | Mapping[str, object] | None,
) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, KernelReplyRequestV1):
        parsed = _JSON_OBJECT_ADAPTER.validate_json(
            value.model_dump_json(exclude_none=True),
        )
        return {key: item for key, item in parsed.items()}
    return {str(key): item for key, item in value.items()}


def channel_from_payload(
    payload: Mapping[str, object],
    base_context: DomainContextV1 | None,
) -> DistributionChannel:
    if base_context is not None and not payload:
        return base_context.channel
    name = (
        clean(payload.get("dist_channel_name"))
        or clean(payload.get("display_name"))
        or clean(payload.get("channel_name"))
        or clean(payload.get("name"))
        or (base_context.channel.name if base_context is not None else "unknown")
    )
    kind = channel_kind(payload.get("channel_type") or payload.get("kind"))
    if kind == "unknown" and base_context is not None:
        kind = base_context.channel.kind
    return DistributionChannel(
        id=stable_id("channel", name, kind),
        name=name,
        kind=kind,
        source_id=str(
            payload.get("context_id") or payload.get("source_id") or "adapter_channel"
        ),
        provenance="adapter_channel_payload" if payload else "unknown",
    )


def channel_kind(value: object) -> ChannelKind:
    text = clean(value)
    if _is_channel_kind(text):
        return text
    return "unknown"


def available_artifact_payloads(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    artifacts = payload.get("available_artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, str):
        return ()
    return tuple(
        {str(key): nested for key, nested in item.items()}
        for item in artifacts
        if _is_object_dict(item) and clean(item.get("type"))
    )


def material_pack_options_from_available(
    payload: Mapping[str, object],
) -> tuple[str, ...]:
    for artifact in available_artifact_payloads(payload):
        if artifact.get("type") == "material_pack":
            return string_list(artifact.get("options"))
    return ()


def string_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (clean(value),) if clean(value) else ()
    if isinstance(value, Sequence):
        return tuple(item for item in (clean(nested) for nested in value) if item)
    return ()


def string_key_object_mapping(value: object) -> JsonDict:
    if not is_object_mapping(value):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def nested_runtime_items(value: object) -> list[object] | None:
    items = getattr(value, "items", None)
    if not _is_object_list(items):
        return None
    return [item for item in items]


def strategy_map(base_context: DomainContextV1 | None) -> dict[str, Strategy]:
    return {
        strategy.id: strategy
        for strategy in (base_context.strategies if base_context is not None else ())
    }


def product_map(base_context: DomainContextV1 | None) -> dict[str, Product]:
    return {
        product.id: product
        for product in (base_context.products if base_context is not None else ())
    }


def artifact_map(base_context: DomainContextV1 | None) -> dict[str, Artifact]:
    return {
        artifact.id: artifact
        for artifact in (base_context.artifacts if base_context is not None else ())
    }


def time_range_from_metadata(metadata: Mapping[str, object]) -> TimeRange | None:
    time_range = TimeRange(
        period=_optional_clean(metadata.get("period")),
        start=_optional_clean(metadata.get("period_start")),
        end=_optional_clean(metadata.get("period_end")),
        label=_optional_clean(metadata.get("period_label")),
    )
    return time_range if time_range.to_prompt_dict() else None


def _optional_clean(value: object) -> str | None:
    text = clean(value)
    return text or None


def _is_channel_kind(value: str) -> TypeGuard[Literal["bank", "non_bank"]]:
    return value in {"bank", "non_bank"}


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)
