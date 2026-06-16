from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_REGISTRY,
    CapabilitySpec,
    capability_by_business_state_field,
    capability_by_resolve_type,
    resolvable_fact_type_for_resolve,
    resolvable_business_state_fields,
)
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
    resolve_ref: str | None = None
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
    report_date: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_label: str | None = None
    scope_complete: bool | None = None
    expected_product_count: int | None = None
    generated_product_count: int | None = None
    missing_product_count: int | None = None
    report_sections: tuple[dict, ...] = ()


@dataclass(frozen=True)
class ExecutedActionState:
    action_type: str = ""
    resolve_ref: str | None = None
    resolve_ref_available: bool = False
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
        capability = capability_by_resolve_type(resolve_type)
        if capability is None or capability.business_state_field is None:
            return ResolvableState()
        state = getattr(self, capability.business_state_field, None)
        return state if isinstance(state, ResolvableState) else ResolvableState()

    def report_state(self, resolve_type: AdapterResolveType) -> ReportState | None:
        capability = capability_by_resolve_type(resolve_type)
        if capability is None or not capability.is_report:
            return None
        state = getattr(self, capability.business_state_field or "", None)
        return state if isinstance(state, ReportState) else None

    def to_prompt_dict(self) -> dict:
        payload = {
            field_name: _state_dict(getattr(self, field_name))
            for field_name in resolvable_business_state_fields()
        }
        payload.update(
            {
            "recent_executed_actions": [
                _executed_action_dict(action)
                for action in self.recent_executed_actions
            ],
            "requested_strategy_status": self.requested_strategy_status,
            "user_permission": self.user_permission,
            "evidence_fact_count": self.evidence_fact_count,
            }
        )
        return payload


def derive_business_facts(
        evidence_facts: list[EvidenceFact],
        request: ReplyRequest | None = None,
) -> BusinessFacts:
    states = _derive_registry_states(evidence_facts)

    return BusinessFacts(
        material_pack=_state_for_field(states, "material_pack", ResolvableState),
        weekly_report=_state_for_field(states, "weekly_report", ReportState),
        monthly_report=_state_for_field(states, "monthly_report", ReportState),
        sales_mention=_state_for_field(states, "sales_mention", ResolvableState),
        recent_executed_actions=_derive_recent_executed_actions(evidence_facts),
        requested_strategy_status=_derive_requested_strategy_status(
            request,
            states,
        ),
        user_permission="unknown",
        evidence_fact_count=len(evidence_facts),
    )


def _derive_registry_states(evidence_facts: list[EvidenceFact]) -> dict[str, ResolvableState]:
    states: dict[str, ResolvableState] = {}
    for field_name in resolvable_business_state_fields():
        capability = capability_by_business_state_field(field_name)
        if capability is None or capability.resolve_type is None:
            continue
        if capability.is_report:
            states[field_name] = _derive_report_state(
                evidence_facts,
                capability.resolve_type,
            )
        else:
            states[field_name] = _derive_resolvable_state(
                evidence_facts,
                capability.resolve_type,
            )
    return states


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
                resolve_ref=_optional_str(metadata.get("resolve_ref")),
                resolve_ref_available=bool(metadata.get("resolve_ref_available")),
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
    base_fact_type = resolvable_fact_type_for_resolve(resolve_type)
    base_fact = (
        find_fact(evidence_facts, base_fact_type, resolve_type)  # type: ignore[arg-type]
        if base_fact_type is not None
        else None
    )
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
    period_fact = find_fact(
        evidence_facts,
        "report_period",
        resolve_type,
    )
    metadata = {}
    if base_fact is not None:
        metadata.update(base_fact.metadata)
    if period_fact is not None:
        metadata.update(period_fact.metadata)
    if contains_fact is not None:
        metadata.update(contains_fact.metadata)
    if scope_fact is not None:
        metadata.update(scope_fact.metadata)

    return ReportState(
        status=base_state.status,
        candidates=base_state.candidates,
        reason_code=base_state.reason_code,
        source_id=base_state.source_id,
        resolve_ref=base_state.resolve_ref,
        contains_strategy=(
            contains_fact.value if isinstance(contains_fact.value, bool) else None
        )
        if contains_fact is not None
        else None,
        scope_status=_normalize_scope_status(scope_fact.value if scope_fact else None),
        strategy=_optional_str(metadata.get("strategy")) or base_state.strategy,
        period=_optional_str(metadata.get("period")),
        report_date=_optional_str(metadata.get("report_date")),
        period_start=_optional_str(metadata.get("period_start")),
        period_end=_optional_str(metadata.get("period_end")),
        period_label=_optional_str(metadata.get("period_label")),
        scope_complete=(
            bool(metadata.get("scope_complete"))
            if metadata.get("scope_complete") is not None
            else None
        ),
        expected_product_count=_optional_int(metadata.get("expected_product_count")),
        generated_product_count=_optional_int(metadata.get("generated_product_count")),
        missing_product_count=_optional_int(metadata.get("missing_product_count")),
        report_sections=tuple(
            section
            for section in metadata.get("report_sections", [])
            if isinstance(section, dict)
        ),
    )


def _derive_resolvable_state(
        evidence_facts: list[EvidenceFact],
        resolve_type: AdapterResolveType,
) -> ResolvableState:
    fact_type = resolvable_fact_type_for_resolve(resolve_type)
    if fact_type is None:
        return ResolvableState()
    fact = find_fact(
        evidence_facts,
        fact_type,  # type: ignore[arg-type]
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
        resolve_ref=_optional_str(fact.metadata.get("resolve_ref")),
        strategy=_optional_str(fact.metadata.get("strategy")),
    )


def _derive_requested_strategy_status(
        request: ReplyRequest | None,
        states: dict[str, ResolvableState],
) -> AvailabilityStatus:
    if request is None:
        return "unknown"
    for capability in _strategy_status_capabilities(is_report=False):
        if capability.business_state_field is None:
            continue
        state = states.get(capability.business_state_field)
        if state is not None and state.status in {"available", "ambiguous", "unavailable"}:
            return state.status
    for capability in _strategy_status_capabilities(is_report=True):
        if capability.business_state_field is None:
            continue
        report = states.get(capability.business_state_field)
        if not isinstance(report, ReportState):
            continue
        if report.contains_strategy is True or report.scope_status == "included":
            return "available"
        if report.contains_strategy is False or report.scope_status == "excluded":
            return "unavailable"
    return "unknown"


def _strategy_status_capabilities(*, is_report: bool) -> tuple[CapabilitySpec, ...]:
    return tuple(
        capability
        for capability in CAPABILITY_REGISTRY
        if capability.business_state_field is not None
        and capability.resolve_type is not None
        and capability.is_report is is_report
    )


def _state_for_field(
        states: dict[str, ResolvableState],
        field_name: str,
        state_type: type[ResolvableState],
):
    state = states.get(field_name)
    if isinstance(state, state_type):
        return state
    return state_type()


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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _state_dict(state: ResolvableState) -> dict:
    payload = {
        "status": state.status,
        "resolvable": state.resolvable,
        "candidates": list(state.candidates),
        "reason_code": state.reason_code,
        "source_id": state.source_id,
        "resolve_ref_available": bool(state.resolve_ref),
        "strategy": state.strategy,
    }
    if isinstance(state, ReportState):
        payload.update(
            {
                "contains_strategy": state.contains_strategy,
                "scope_status": state.scope_status,
                "strategy": state.strategy,
                "period": state.period,
                "report_date": state.report_date,
                "period_start": state.period_start,
                "period_end": state.period_end,
                "period_label": state.period_label,
                "scope_complete": state.scope_complete,
                "expected_product_count": state.expected_product_count,
                "generated_product_count": state.generated_product_count,
                "missing_product_count": state.missing_product_count,
                "report_sections": list(state.report_sections),
            }
        )
    return payload


def _executed_action_dict(action: ExecutedActionState) -> dict:
    return {
        "action_type": action.action_type,
        "resolve_ref_available": action.resolve_ref_available,
        "material_type": action.material_type,
        "strategy": action.strategy,
        "version": action.version,
        "action_id": action.action_id,
        "response_id": action.response_id,
        "context_id": action.context_id,
        "material_ref_available": action.material_ref_available,
        "received_at": action.received_at,
    }
