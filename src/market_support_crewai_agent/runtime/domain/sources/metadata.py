from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    TimeRange,
)

SourceContextType = Literal[
    "current_artifact",
    "adapter_context",
    "user_message",
    "assistant_message",
    "history_summary",
    "retrieved_doc",
    "tool_result",
]


@dataclass(frozen=True)
class SourceMetadata:
    source_id: str
    source_type: SourceContextType
    artifact_type: str | None = None
    channel_id: str | None = None
    strategy_id: str | None = None
    product_ids: tuple[str, ...] = ()
    time_range: TimeRange | None = None
    created_at: datetime | str | None = None
    observed_at: datetime | str | None = None
    provenance: str = "unknown"
    evidence_allowed_by_default: bool = False

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "source_id": self.source_id,
                "source_type": self.source_type,
                "artifact_type": self.artifact_type,
                "channel_id": self.channel_id,
                "strategy_id": self.strategy_id,
                "product_ids": list(self.product_ids),
                "time_range": (
                    self.time_range.to_prompt_dict()
                    if self.time_range is not None
                    else None
                ),
                "created_at": _isoformat(self.created_at),
                "observed_at": _isoformat(self.observed_at),
                "provenance": self.provenance,
                "evidence_allowed_by_default": self.evidence_allowed_by_default,
            }.items()
            if value not in (None, "", [], {})
        }


def source_metadata_from_mapping(value: Mapping[str, object]) -> SourceMetadata:
    source_type = _coerce_source_context_type(value.get("source_type"))
    return SourceMetadata(
        source_id=str(value.get("source_id") or ""),
        source_type=source_type,  # type: ignore[arg-type]
        artifact_type=_optional_str(value.get("artifact_type")),
        channel_id=_optional_str(value.get("channel_id")),
        strategy_id=_optional_str(value.get("strategy_id")),
        product_ids=_string_tuple(value.get("product_ids")),
        time_range=_time_range_from_mapping(value.get("time_range")),
        created_at=_optional_str(value.get("created_at")),
        observed_at=_optional_str(value.get("observed_at")),
        provenance=str(value.get("provenance") or "unknown"),
        evidence_allowed_by_default=bool(value.get("evidence_allowed_by_default")),
    )


def source_metadata_for_evidence(
    *,
    fact_source_type: object,
    source_id: object,
    artifact_type: object,
    fact_type: object,
    metadata: Mapping[str, object],
    scope: ArtifactScope,
) -> SourceMetadata:
    source_type = _source_context_type(
        fact_source_type=fact_source_type,
        artifact_type=artifact_type,
        fact_type=fact_type,
        metadata=metadata,
    )
    artifact = _clean(artifact_type)
    channel_id = _first_non_unknown(
        scope.channel_id,
        metadata.get("channel_id"),
        metadata.get("channel"),
    )
    strategy_id = _first_non_unknown(scope.strategy_id, metadata.get("strategy_id"))
    time_range = scope.time_range or _time_range_from_mapping(metadata.get("time_range"))
    observed_at = (
        metadata.get("observed_at")
        or metadata.get("resolved_at")
        or metadata.get("received_at")
    )
    provenance = str(scope.provenance or "").strip()
    if not provenance or provenance == "unknown":
        provenance = str(fact_source_type or "unknown")
    return SourceMetadata(
        source_id=str(source_id or metadata.get("source_id") or "").strip(),
        source_type=source_type,
        artifact_type=artifact if artifact and artifact != "unknown" else None,
        channel_id=channel_id,
        strategy_id=strategy_id,
        product_ids=tuple(scope.product_ids) or _string_tuple(metadata.get("product_ids")),
        time_range=time_range,
        created_at=_optional_str(metadata.get("created_at")),
        observed_at=_optional_str(observed_at),
        provenance=provenance,
        evidence_allowed_by_default=_evidence_allowed_by_default(
            source_type,
            artifact_type=artifact,
        ),
    )


def source_metadata_for_conversation_message(
    *,
    conversation_key: str,
    role: Literal["user", "assistant"],
    created_at: datetime,
) -> SourceMetadata:
    return SourceMetadata(
        source_id=f"{conversation_key}:{role}:{created_at.isoformat()}",
        source_type="user_message" if role == "user" else "assistant_message",
        artifact_type="history",
        created_at=created_at,
        observed_at=created_at,
        provenance="conversation_store",
        evidence_allowed_by_default=False,
    )


def is_history_source(metadata: SourceMetadata | None) -> bool:
    if metadata is None:
        return False
    return (
        metadata.source_type in {"user_message", "assistant_message", "history_summary"}
        or metadata.artifact_type == "history"
    )


def source_metadata_prompt_dict(metadata: SourceMetadata | Mapping[str, object] | None) -> dict:
    if isinstance(metadata, SourceMetadata):
        return metadata.to_prompt_dict()
    if isinstance(metadata, Mapping):
        return source_metadata_from_mapping(metadata).to_prompt_dict()
    return {}


_SOURCE_CONTEXT_TYPE_BY_VALUE: dict[str, SourceContextType] = {
    "current_artifact": "current_artifact",
    "adapter_context": "adapter_context",
    "user_message": "user_message",
    "assistant_message": "assistant_message",
    "history_summary": "history_summary",
    "retrieved_doc": "retrieved_doc",
    "tool_result": "tool_result",
}


def _source_context_type(
    *,
    fact_source_type: object,
    artifact_type: object,
    fact_type: object,
    metadata: Mapping[str, object],
) -> SourceContextType:
    explicit = _coerce_source_context_type(metadata.get("source_metadata_type"))
    if explicit != "tool_result":
        return explicit

    artifact = _clean(artifact_type)
    source = _clean(fact_source_type)
    if artifact == "history":
        role = _clean(metadata.get("role"))
        if role == "user":
            return "user_message"
        if role == "assistant":
            return "assistant_message"
        if source == "conversation_history":
            return "history_summary"
        return "tool_result" if source == "action_ledger" else "assistant_message"
    if source in {"document_mcp", "approved_static_knowledge"}:
        return "retrieved_doc"
    if source == "conversation_history":
        return "history_summary"
    if source == "user_upload":
        return "user_message"
    if source in {"adapter_resolve", "adapter_report_scope", "adapter_material_pack_content"}:
        if artifact in {"material_pack", "weekly_report", "monthly_report"}:
            return "current_artifact"
        return "adapter_context"
    del fact_type
    return "tool_result"


def _coerce_source_context_type(value: object) -> SourceContextType:
    return _SOURCE_CONTEXT_TYPE_BY_VALUE.get(str(value or "").strip(), "tool_result")


def _evidence_allowed_by_default(
    source_type: SourceContextType,
    *,
    artifact_type: str,
) -> bool:
    if source_type in {"user_message", "assistant_message", "history_summary"}:
        return False
    if artifact_type == "history":
        return False
    return source_type in {"current_artifact", "adapter_context", "retrieved_doc", "tool_result"}


def _time_range_from_mapping(value: object) -> TimeRange | None:
    if isinstance(value, TimeRange):
        return value
    if not isinstance(value, Mapping):
        return None
    return TimeRange(
        period=_optional_str(value.get("period")),
        start=_optional_str(value.get("start")),
        end=_optional_str(value.get("end")),
        label=_optional_str(value.get("label")),
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    return ()


def _first_non_unknown(*values: object) -> str | None:
    for value in values:
        text = _optional_str(value)
        if text and text != "unknown":
            return text
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean(value: object) -> str:
    return str(value or "").strip()


def _isoformat(value: datetime | str | None) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return _optional_str(value)
