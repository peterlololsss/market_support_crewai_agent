from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from market_support_crewai_agent.runtime.evidence import EvidenceFact, find_fact
from market_support_crewai_agent.schemas import AdapterResolveType, ReplyRequest

AvailabilityStatus = Literal["available", "ambiguous", "unavailable", "unknown"]
ReportScopeStatus = Literal["included", "excluded", "unknown"]
UserPermissionStatus = Literal["allowed", "denied", "unknown"]


@dataclass(frozen=True)
class ResolvableState:
    status: AvailabilityStatus = "unknown"
    candidates: tuple[str, ...] = ()
    reason_code: str = ""
    source_id: str = ""
    strategy: str | None = None

    @property
    def resolvable(self) -> bool:
        return self.status == "available"


@dataclass(frozen=True)
class ReportState(ResolvableState):
    contains_strategy: bool | None = None
    scope_status: ReportScopeStatus = "unknown"
    strategy: str | None = None
    period: str | None = None


@dataclass(frozen=True)
class ExecutedActionState:
    action_type: str = ""
    material_type: str | None = None
    strategy: str | None = None
    version: str | None = None
    action_id: str | None = None
    response_id: str | None = None
    context_id: str | None = None
    material_ref_available: bool = False
    received_at: str = ""


@dataclass(frozen=True)
class BusinessFacts:
    material_pack: ResolvableState = field(default_factory=ResolvableState)
    weekly_report: ReportState = field(default_factory=ReportState)
    monthly_report: ReportState = field(default_factory=ReportState)
    sales_mention: ResolvableState = field(default_factory=ResolvableState)
    recent_executed_actions: tuple[ExecutedActionState, ...] = ()
    requested_strategy_status: AvailabilityStatus = "unknown"
    user_permission: UserPermissionStatus = "unknown"
    evidence_fact_count: int = 0

    @classmethod
    def empty(cls) -> BusinessFacts:
        return cls()

    def resolve_state(self, resolve_type: AdapterResolveType) -> ResolvableState:
        if resolve_type == "material_pack":
            return self.material_pack
        if resolve_type == "weekly_report":
            return self.weekly_report
        if resolve_type == "monthly_report":
            return self.monthly_report
        return self.sales_mention

    def report_state(self, resolve_type: AdapterResolveType) -> ReportState | None:
        if resolve_type == "weekly_report":
            return self.weekly_report
        if resolve_type == "monthly_report":
            return self.monthly_report
        return None

    def to_prompt_dict(self) -> dict:
        return {
            "material_pack": _state_dict(self.material_pack),
            "weekly_report": _state_dict(self.weekly_report),
            "monthly_report": _state_dict(self.monthly_report),
            "sales_mention": _state_dict(self.sales_mention),
            "recent_executed_actions": [
                _executed_action_dict(action)
                for action in self.recent_executed_actions
            ],
            "requested_strategy_status": self.requested_strategy_status,
            "user_permission": self.user_permission,
            "evidence_fact_count": self.evidence_fact_count,
        }


_RESOLVABLE_FACT_BY_RESOLVE: dict[AdapterResolveType, str] = {
    "material_pack": "material_pack_resolvable",
    "weekly_report": "weekly_report_resolvable",
    "monthly_report": "monthly_report_resolvable",
    "sales_mention": "sales_mention_resolvable",
}


def derive_business_facts(
        evidence_facts: list[EvidenceFact],
        request: ReplyRequest | None = None,
) -> BusinessFacts:
    material_pack = _derive_resolvable_state(evidence_facts, "material_pack")
    weekly_report = _derive_report_state(evidence_facts, "weekly_report")
    monthly_report = _derive_report_state(evidence_facts, "monthly_report")
    sales_mention = _derive_resolvable_state(evidence_facts, "sales_mention")

    return BusinessFacts(
        material_pack=material_pack,
        weekly_report=weekly_report,
        monthly_report=monthly_report,
        sales_mention=sales_mention,
        recent_executed_actions=_derive_recent_executed_actions(evidence_facts),
        requested_strategy_status=_derive_requested_strategy_status(
            request,
            material_pack,
            weekly_report,
            monthly_report,
        ),
        user_permission="unknown",
        evidence_fact_count=len(evidence_facts),
    )


def _derive_recent_executed_actions(
        evidence_facts: list[EvidenceFact],
) -> tuple[ExecutedActionState, ...]:
    actions: list[ExecutedActionState] = []
    for fact in evidence_facts:
        if fact.fact_type != "recent_executed_action":
            continue
        metadata = fact.metadata
        actions.append(
            ExecutedActionState(
                action_type=str(metadata.get("action_type") or ""),
                material_type=_optional_str(metadata.get("material_type")),
                strategy=_optional_str(metadata.get("strategy")),
                version=_optional_str(metadata.get("version")),
                action_id=_optional_str(metadata.get("action_id")),
                response_id=_optional_str(metadata.get("response_id")),
                context_id=_optional_str(metadata.get("context_id")),
                material_ref_available=bool(metadata.get("material_ref_available")),
                received_at=str(metadata.get("received_at") or ""),
            )
        )
    return tuple(actions)


def _derive_report_state(
        evidence_facts: list[EvidenceFact],
        resolve_type: AdapterResolveType,
) -> ReportState:
    base_state = _derive_resolvable_state(evidence_facts, resolve_type)
    contains_fact = find_fact(
        evidence_facts,
        "report_contains_strategy",
        resolve_type,
    )
    scope_fact = find_fact(
        evidence_facts,
        "report_scope_status",
        resolve_type,
    )
    metadata = {}
    if contains_fact is not None:
        metadata.update(contains_fact.metadata)
    if scope_fact is not None:
        metadata.update(scope_fact.metadata)

    return ReportState(
        status=base_state.status,
        candidates=base_state.candidates,
        reason_code=base_state.reason_code,
        source_id=base_state.source_id,
        contains_strategy=(
            contains_fact.value if isinstance(contains_fact.value, bool) else None
        )
        if contains_fact is not None
        else None,
        scope_status=_normalize_scope_status(scope_fact.value if scope_fact else None),
        strategy=_optional_str(metadata.get("strategy")) or base_state.strategy,
        period=_optional_str(metadata.get("period")),
    )


def _derive_resolvable_state(
        evidence_facts: list[EvidenceFact],
        resolve_type: AdapterResolveType,
) -> ResolvableState:
    fact = find_fact(
        evidence_facts,
        _RESOLVABLE_FACT_BY_RESOLVE[resolve_type],
        resolve_type,
    )
    if fact is None:
        return ResolvableState()

    status = str(fact.metadata.get("status") or "")
    return ResolvableState(
        status=_availability_from_fact(fact.value, status),
        candidates=tuple(str(candidate) for candidate in fact.metadata.get("candidates", [])),
        reason_code=str(fact.metadata.get("reason_code") or ""),
        source_id=fact.source_id,
        strategy=_optional_str(fact.metadata.get("strategy")),
    )


def _derive_requested_strategy_status(
        request: ReplyRequest | None,
        material_pack: ResolvableState,
        weekly_report: ReportState,
        monthly_report: ReportState,
) -> AvailabilityStatus:
    if request is None:
        return "unknown"
    if material_pack.status in {"available", "ambiguous", "unavailable"}:
        return material_pack.status
    for report in (weekly_report, monthly_report):
        if report.contains_strategy is True or report.scope_status == "included":
            return "available"
        if report.contains_strategy is False or report.scope_status == "excluded":
            return "unavailable"
    return "unknown"


def _availability_from_fact(value: object, status: str) -> AvailabilityStatus:
    if value is True:
        return "available"
    if status == "ambiguous":
        return "ambiguous"
    if value is False:
        return "unavailable"
    return "unknown"


def _normalize_scope_status(value: object) -> ReportScopeStatus:
    if value in {"included", "excluded"}:
        return value
    return "unknown"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _state_dict(state: ResolvableState) -> dict:
    payload = {
        "status": state.status,
        "resolvable": state.resolvable,
        "candidates": list(state.candidates),
        "reason_code": state.reason_code,
        "source_id": state.source_id,
        "strategy": state.strategy,
    }
    if isinstance(state, ReportState):
        payload.update(
            {
                "contains_strategy": state.contains_strategy,
                "scope_status": state.scope_status,
                "strategy": state.strategy,
                "period": state.period,
            }
        )
    return payload


def _executed_action_dict(action: ExecutedActionState) -> dict:
    return {
        "action_type": action.action_type,
        "material_type": action.material_type,
        "strategy": action.strategy,
        "version": action.version,
        "action_id": action.action_id,
        "response_id": action.response_id,
        "context_id": action.context_id,
        "material_ref_available": action.material_ref_available,
        "received_at": action.received_at,
    }
