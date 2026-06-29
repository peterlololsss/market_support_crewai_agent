from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.evidence.adapter_preflight import AdapterPreflightSnapshot
from market_support_crewai_agent.runtime.domain.capabilities import (
    capability_by_resolve_type,
)
from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    ArtifactType,
    artifact_scope_for_evidence,
    normalize_artifact_type,
)
from market_support_crewai_agent.runtime.domain.sources.metadata import (
    SourceMetadata,
    source_metadata_for_evidence,
    source_metadata_from_mapping,
)
from market_support_crewai_agent.schemas import AdapterResolveType

EvidenceFactType = Literal[
    "material_pack_resolvable",
    "material_pack_open_calendar",
    "weekly_report_resolvable",
    "monthly_report_resolvable",
    "sales_mention_resolvable",
    "report_period",
    "report_scope_summary",
    "report_scope_match",
    "report_scope_products",
    "report_scope_unavailable",
    "recent_executed_action",
    "document_context",
    "document_context_unavailable",
]
EvidenceFactValue = bool | str | None
EvidenceSourceType = Literal[
    "adapter_resolve",
    "adapter_report_scope",
    "adapter_material_pack_content",
    "action_ledger",
    "document_mcp",
    "approved_static_knowledge",
    "conversation_history",
    "user_upload",
    "current_artifact",
    "adapter_context",
    "user_message",
    "assistant_message",
    "history_summary",
    "retrieved_doc",
    "tool_result",
]


@dataclass(frozen=True)
class EvidenceFact:
    fact_type: EvidenceFactType
    value: EvidenceFactValue
    source_type: EvidenceSourceType = "adapter_resolve"
    source_id: str = ""
    resolve_type: AdapterResolveType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_type: ArtifactType = "unknown"
    scope: ArtifactScope = field(default_factory=lambda: ArtifactScope(channel_id="unknown"))
    source_metadata: SourceMetadata | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        artifact_type = normalize_artifact_type(
            self.artifact_type,
            resolve_type=self.resolve_type,
            source_type=self.source_type,
            fact_type=self.fact_type,
        )
        if artifact_type != self.artifact_type:
            object.__setattr__(self, "artifact_type", artifact_type)
        if self.scope.channel_id == "unknown" and artifact_type != "unknown":
            object.__setattr__(
                self,
                "scope",
                artifact_scope_for_evidence(
                    channel_id="unknown",
                    artifact_type=artifact_type,
                    resolve_type=self.resolve_type,
                    source_id=self.source_id,
                    source_type=self.source_type,
                    metadata=self.metadata,
                ),
            )
        if isinstance(self.source_metadata, SourceMetadata):
            return
        if isinstance(self.source_metadata, dict):
            object.__setattr__(
                self,
                "source_metadata",
                source_metadata_from_mapping(self.source_metadata),
            )
            return
        object.__setattr__(
            self,
            "source_metadata",
            source_metadata_for_evidence(
                fact_source_type=self.source_type,
                source_id=self.source_id,
                artifact_type=self.artifact_type,
                fact_type=self.fact_type,
                metadata=self.metadata,
                scope=self.scope,
            ),
        )


def evidence_facts_from_preflight(
        preflight: AdapterPreflightSnapshot,
) -> list[EvidenceFact]:
    facts: list[EvidenceFact] = []
    for item in preflight.items:
        capability = capability_by_resolve_type(item.resolve_type)
        fact_type = capability.resolvable_fact_type if capability is not None else None
        if fact_type is not None:
            result = item.result
            facts.append(
                _fact(
                    fact_type=fact_type,  # type: ignore[arg-type]
                    value=item.status == "resolved",
                    source_id=item.resolve_type,
                    resolve_type=item.resolve_type,
                    metadata={
                        "status": item.status,
                        "resolve_ref": (
                            result.resolve_ref if result is not None else None
                        ),
                        "candidates": (
                            result.candidates if result is not None else []
                        ),
                        "reason_code": (
                            result.reason_code if result is not None else ""
                        ),
                        "material_pack_option": (
                            result.material_pack_option if result is not None else None
                        ),
                        "period": (
                            result.period if result is not None else None
                        ),
                        "report_date": (
                            result.report_date if result is not None else None
                        ),
                        "period_start": (
                            result.period_start if result is not None else None
                        ),
                        "period_end": (
                            result.period_end if result is not None else None
                        ),
                        "period_label": (
                            result.period_label if result is not None else None
                        ),
                        "scope_complete": (
                            result.scope_complete if result is not None else None
                        ),
                        "expected_product_count": (
                            result.expected_product_count if result is not None else None
                        ),
                        "generated_product_count": (
                            result.generated_product_count if result is not None else None
                        ),
                        "missing_product_count": (
                            result.missing_product_count if result is not None else None
                        ),
                        "report_sections": (
                            [
                                section.model_dump(mode="json", exclude_none=True)
                                for section in result.report_sections
                            ]
                            if result is not None
                            else []
                        ),
                    },
                    result=result,
                )
            )

        if item.result is None:
            continue

        result = item.result
        capability = capability_by_resolve_type(result.resolve_type)
        if capability is not None and capability.is_report:
            if result.period:
                facts.append(
                    _fact(
                        fact_type="report_period",
                        value=result.period,
                        source_id=result.resolve_type,
                        resolve_type=result.resolve_type,
                        metadata={
                            "status": result.status,
                            "period": result.period,
                            "report_date": result.report_date,
                            "period_start": result.period_start,
                            "period_end": result.period_end,
                            "period_label": result.period_label,
                        },
                        result=result,
                    )
                )
    return facts


def evidence_facts_from_action_history(
        action_history: list[ActionLedgerRecord] | None,
) -> list[EvidenceFact]:
    facts: list[EvidenceFact] = []
    for record in action_history or []:
        execution = record.execution
        if execution.status != "executed":
            continue
        artifact = (
            execution.artifact.model_dump(mode="json", exclude_none=True)
            if execution.artifact is not None
            else None
        )
        facts.append(
            _fact(
                fact_type="recent_executed_action",
                value=True,
                source_type="action_ledger",
                source_id=record.response_id or record.context_id or "",
                metadata={
                    "context_id": record.context_id,
                    "response_id": record.response_id,
                    "action_id": execution.action_id,
                    "action_type": execution.action_type,
                    "artifact": artifact,
                    "received_at": record.received_at.isoformat(),
                },
                artifact_type="history",
            )
        )
    return facts


def fact_value(
        facts: list[EvidenceFact],
        fact_type: EvidenceFactType,
        resolve_type: AdapterResolveType | None = None,
) -> EvidenceFactValue:
    fact = find_fact(facts, fact_type, resolve_type)
    if fact is None:
        return None
    return fact.value


def find_fact(
        facts: list[EvidenceFact],
        fact_type: EvidenceFactType,
        resolve_type: AdapterResolveType | None = None,
) -> EvidenceFact | None:
    for fact in facts:
        if fact.fact_type != fact_type:
            continue
        if resolve_type is not None and fact.resolve_type != resolve_type:
            continue
        return fact
    return None


def _fact(
        *,
        fact_type: EvidenceFactType,
        value: EvidenceFactValue,
        source_type: EvidenceSourceType = "adapter_resolve",
        source_id: str = "",
        resolve_type: AdapterResolveType | None = None,
        metadata: dict[str, Any] | None = None,
        result=None,
        artifact_type: ArtifactType | None = None,
) -> EvidenceFact:
    metadata = metadata or {}
    resolved_artifact_type = artifact_type or normalize_artifact_type(
        resolve_type=resolve_type,
        source_type=source_type,
        fact_type=fact_type,
    )
    scope = artifact_scope_for_evidence(
        channel_id=_channel_scope_id(result),
        artifact_type=resolved_artifact_type,
        resolve_type=resolve_type,
        source_id=source_id,
        source_type=source_type,
        metadata=metadata,
    )
    return EvidenceFact(
        fact_type=fact_type,
        value=value,
        source_type=source_type,
        source_id=source_id,
        resolve_type=resolve_type,
        metadata=metadata,
        artifact_type=resolved_artifact_type,
        scope=scope,
    )


def _channel_scope_id(result) -> str:
    if result is None:
        return "unknown"
    display_name = str(getattr(result, "display_name", "") or "").strip()
    channel_type = str(getattr(result, "channel_type", "") or "unknown").strip()
    if not display_name:
        return "unknown"
    return f"adapter_channel:{channel_type}:{display_name}"
