from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.evidence.adapter_preflight import AdapterPreflightSnapshot
from market_support_crewai_agent.runtime.domain.capabilities import (
    capability_by_resolve_type,
)
from market_support_crewai_agent.schemas import AdapterResolveType

EvidenceFactType = Literal[
    "material_pack_resolvable",
    "weekly_report_resolvable",
    "monthly_report_resolvable",
    "sales_mention_resolvable",
    "report_contains_strategy",
    "report_scope_status",
    "recent_executed_action",
    "document_context",
    "document_context_unavailable",
]
EvidenceFactValue = bool | str | None


@dataclass(frozen=True)
class EvidenceFact:
    fact_type: EvidenceFactType
    value: EvidenceFactValue
    source_type: Literal[
        "adapter_resolve",
        "action_ledger",
        "document_mcp",
        "approved_static_knowledge",
    ] = "adapter_resolve"
    source_id: str = ""
    resolve_type: AdapterResolveType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
                EvidenceFact(
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
                        "strategy": (
                            result.strategy if result is not None else None
                        ),
                        "period": (
                            result.period if result is not None else None
                        ),
                        "report_date": (
                            result.report_date if result is not None else None
                        ),
                    },
                )
            )

        if item.result is None:
            continue

        result = item.result
        capability = capability_by_resolve_type(result.resolve_type)
        if capability is not None and capability.is_report:
            if result.contains_strategy is not None:
                facts.append(
                    EvidenceFact(
                        fact_type="report_contains_strategy",
                        value=result.contains_strategy,
                        source_id=result.resolve_type,
                        resolve_type=result.resolve_type,
                        metadata={
                            "strategy": result.strategy,
                            "period": result.period,
                            "report_date": result.report_date,
                        },
                    )
                )
            if result.scope_status is not None:
                facts.append(
                    EvidenceFact(
                        fact_type="report_scope_status",
                        value=result.scope_status,
                        source_id=result.resolve_type,
                        resolve_type=result.resolve_type,
                        metadata={
                            "strategy": result.strategy,
                            "period": result.period,
                            "report_date": result.report_date,
                        },
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
        facts.append(
            EvidenceFact(
                fact_type="recent_executed_action",
                value=True,
                source_type="action_ledger",
                source_id=record.response_id or record.context_id or "",
                metadata={
                    "context_id": record.context_id,
                    "response_id": record.response_id,
                    "action_id": execution.action_id,
                    "action_type": execution.action_type,
                    "resolve_ref": execution.resolve_ref,
                    "resolve_ref_available": bool(execution.resolve_ref),
                    "material_type": execution.material_type,
                    "strategy": execution.strategy,
                    "version": execution.version,
                    "material_ref_available": bool(execution.material_id),
                    "received_at": record.received_at.isoformat(),
                },
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
