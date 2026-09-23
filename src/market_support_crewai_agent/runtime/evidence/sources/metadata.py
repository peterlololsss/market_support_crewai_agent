from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from market_support_crewai_agent.runtime.evidence.sources import metadata_parsing
from market_support_crewai_agent.runtime.policy.ontology_models import (
    ArtifactScope,
    TimeRange,
)


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    source_id: str
    source_type: metadata_parsing.SourceContextType
    artifact_type: str | None = None
    channel_id: str | None = None
    strategy_id: str | None = None
    product_ids: tuple[str, ...] = ()
    time_range: TimeRange | None = None
    created_at: datetime | str | None = None
    observed_at: datetime | str | None = None
    provenance: str = "unknown"
    evidence_allowed_by_default: bool = False

    def to_prompt_dict(self) -> metadata_parsing.SourcePromptDict:
        result: metadata_parsing.SourcePromptDict = {
            "source_type": self.source_type,
            "provenance": self.provenance,
            "evidence_allowed_by_default": self.evidence_allowed_by_default,
        }
        if self.source_id:
            result["source_id"] = self.source_id
        if self.artifact_type:
            result["artifact_type"] = self.artifact_type
        if self.channel_id:
            result["channel_id"] = self.channel_id
        if self.strategy_id:
            result["strategy_id"] = self.strategy_id
        if self.product_ids:
            result["product_ids"] = list(self.product_ids)
        time_range = metadata_parsing.time_range_prompt_dict(self.time_range)
        if time_range:
            result["time_range"] = time_range
        created_at = metadata_parsing.isoformat_metadata_value(self.created_at)
        if created_at:
            result["created_at"] = created_at
        observed_at = metadata_parsing.isoformat_metadata_value(self.observed_at)
        if observed_at:
            result["observed_at"] = observed_at
        return result


def source_metadata_from_mapping(
    value: Mapping[str, metadata_parsing.SourceMetadataValue],
) -> SourceMetadata:
    source_type = metadata_parsing.coerce_source_context_type(value.get("source_type"))
    return SourceMetadata(
        source_id=str(value.get("source_id") or ""),
        source_type=source_type,
        artifact_type=metadata_parsing.optional_metadata_string(
            value.get("artifact_type")
        ),
        channel_id=metadata_parsing.optional_metadata_string(value.get("channel_id")),
        strategy_id=metadata_parsing.optional_metadata_string(value.get("strategy_id")),
        product_ids=metadata_parsing.source_product_ids(value.get("product_ids")),
        time_range=metadata_parsing.time_range_from_metadata(value.get("time_range")),
        created_at=metadata_parsing.optional_metadata_string(value.get("created_at")),
        observed_at=metadata_parsing.optional_metadata_string(value.get("observed_at")),
        provenance=str(value.get("provenance") or "unknown"),
        evidence_allowed_by_default=bool(value.get("evidence_allowed_by_default")),
    )


def source_metadata_for_evidence(
    *,
    fact_source_type: metadata_parsing.SourceMetadataValue,
    source_id: metadata_parsing.SourceMetadataValue,
    artifact_type: metadata_parsing.SourceMetadataValue,
    fact_type: metadata_parsing.SourceMetadataValue,
    metadata: Mapping[str, metadata_parsing.SourceMetadataValue],
    scope: ArtifactScope,
) -> SourceMetadata:
    source_type = _source_context_type(
        fact_source_type=fact_source_type,
        artifact_type=artifact_type,
        fact_type=fact_type,
        metadata=metadata,
    )
    artifact = metadata_parsing.clean_metadata_value(artifact_type)
    channel_id = metadata_parsing.first_non_unknown(
        scope.channel_id,
        metadata.get("channel_id"),
        metadata.get("channel"),
    )
    strategy_id = metadata_parsing.first_non_unknown(
        scope.strategy_id, metadata.get("strategy_id")
    )
    time_range = scope.time_range or metadata_parsing.time_range_from_metadata(
        metadata.get("time_range")
    )
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
        product_ids=tuple(scope.product_ids)
        or metadata_parsing.source_product_ids(metadata.get("product_ids")),
        time_range=time_range,
        created_at=metadata_parsing.optional_metadata_string(
            metadata.get("created_at")
        ),
        observed_at=metadata_parsing.optional_metadata_string(observed_at),
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


def source_metadata_prompt_dict(
    metadata: SourceMetadata
    | Mapping[str, metadata_parsing.SourceMetadataValue]
    | None,
) -> metadata_parsing.SourcePromptDict:
    if isinstance(metadata, SourceMetadata):
        return metadata.to_prompt_dict()
    if isinstance(metadata, Mapping):
        return source_metadata_from_mapping(metadata).to_prompt_dict()
    return {}


def _source_context_type(
    *,
    fact_source_type: metadata_parsing.SourceMetadataValue,
    artifact_type: metadata_parsing.SourceMetadataValue,
    fact_type: metadata_parsing.SourceMetadataValue,
    metadata: Mapping[str, metadata_parsing.SourceMetadataValue],
) -> metadata_parsing.SourceContextType:
    explicit = metadata_parsing.coerce_source_context_type(
        metadata.get("source_metadata_type")
    )
    if explicit != "tool_result":
        return explicit

    artifact = metadata_parsing.clean_metadata_value(artifact_type)
    source = metadata_parsing.clean_metadata_value(fact_source_type)
    if artifact == "history":
        role = metadata_parsing.clean_metadata_value(metadata.get("role"))
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
    if source in {"adapter_resolve", "adapter_report_scope"}:
        if artifact in {"material_pack", "weekly_report", "monthly_report"}:
            return "current_artifact"
        return "adapter_context"
    del fact_type
    return "tool_result"


def _evidence_allowed_by_default(
    source_type: metadata_parsing.SourceContextType,
    *,
    artifact_type: str,
) -> bool:
    if source_type in {"user_message", "assistant_message", "history_summary"}:
        return False
    if artifact_type == "history":
        return False
    return source_type in {
        "current_artifact",
        "adapter_context",
        "retrieved_doc",
        "tool_result",
    }
