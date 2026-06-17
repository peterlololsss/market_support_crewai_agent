from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    EvidenceContract,
)
from market_support_crewai_agent.runtime.domain.ontology import (
    ArtifactScope,
    DomainContext,
    TimeRange,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    adapter_channel_id,
    evidence_artifact_type,
    evidence_id,
    is_history_fact,
    ordered_unique,
    requested_scope,
    source_metadata_for_fact,
    source_provenance_missing,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    EvidenceSelection,
    GuardrailDecision,
    RequestedScope,
    make_decision,
)


def retrieval_source_guard(
    *,
    plan: object,
    policy: PolicyManifest,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
) -> GuardrailDecision:
    selection = select_evidence_for_plan(
        plan=plan,
        policy=policy,
        evidence_facts=evidence_facts,
        domain_context=domain_context,
    )
    required = evidence_required_for_plan(plan)
    seen = [evidence_id(fact) for fact in selection.accepted]
    if selection.has_evidence or not required:
        return make_decision(
            "allow",
            "retrieval_source",
            "retrieval_sources_allowed",
            evidence_required=required,
            evidence_seen=seen,
            source_scopes=[fact.scope.to_prompt_dict() for fact in selection.accepted],
        )

    rejected_reason = (
        selection.decisions[0].reason_code
        if selection.decisions
        else "required_evidence_missing"
    )
    return make_decision(
        "abstain",
        "retrieval_source",
        rejected_reason,
        human_reason="Required evidence was missing or outside the allowed source scope.",
        evidence_required=required,
        evidence_seen=[evidence_id(fact) for fact in evidence_facts],
        source_scopes=[fact.scope.to_prompt_dict() for fact in evidence_facts],
    )


def select_evidence_for_plan(
    *,
    plan: object,
    policy: PolicyManifest,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
) -> EvidenceSelection:
    del policy
    answer_capabilities = tuple(getattr(plan, "answer_capabilities", ()) or ())
    if not answer_capabilities:
        return EvidenceSelection((), (), ())

    accepted: list[EvidenceFact] = []
    rejected: list[EvidenceFact] = []
    decisions: list[GuardrailDecision] = []
    for fact in evidence_facts:
        capability_decisions = [
            fact_decision_for_capability(
                fact=fact,
                capability=capability,
                plan=plan,
                domain_context=domain_context,
            )
            for capability in answer_capabilities
        ]
        if any(decision.outcome == "allow" for decision in capability_decisions):
            accepted.append(fact)
            continue
        if any(
            fact_is_relevant_to_capability(fact, capability, plan)
            for capability in answer_capabilities
        ):
            rejected.append(fact)
            decisions.extend(capability_decisions)

    return EvidenceSelection(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        decisions=tuple(
            decision
            for decision in decisions
            if decision.outcome != "allow"
        ),
    )


def fact_decision_for_capability(
    *,
    fact: EvidenceFact,
    capability: str,
    plan: object,
    domain_context: DomainContext | None,
) -> GuardrailDecision:
    if not fact_is_relevant_to_capability(fact, capability, plan):
        return make_decision(
            "block",
            "retrieval_source",
            "fact_not_relevant_to_capability",
            capability_id=capability,
        )

    manifests = manifests_for_capability(capability, plan)
    rejection_codes: list[str] = []
    for manifest in manifests:
        contract = effective_evidence_contract(manifest, plan)
        code = contract_rejection_code(
            fact=fact,
            capability=capability,
            plan=plan,
            domain_context=domain_context,
            manifest_id=manifest.id,
            allowed_fact_types=tuple(
                contract.required_fact_types
                + contract.any_of_fact_types
            ),
            allowed_source_types=tuple(contract.allowed_source_types),
            forbidden_source_types=tuple(contract.forbidden_source_types),
            allow_history=contract.allow_history,
            allowed_artifact_types=tuple(contract.allowed_artifact_types),
            required_artifact_types=tuple(contract.required_artifact_types),
            required_scope_match=tuple(contract.required_scope_match),
            provenance_required=contract.provenance_required,
        )
        if code:
            rejection_codes.append(code)
            continue
        return make_decision(
            "allow",
            "retrieval_source",
            "evidence_source_allowed",
            capability_id=manifest.id,
            artifact_ids=[fact.source_id] if fact.source_id else [],
            evidence_seen=[evidence_id(fact)],
            source_scopes=[fact.scope.to_prompt_dict()],
        )

    return make_decision(
        "block",
        "retrieval_source",
        rejection_codes[0] if rejection_codes else "evidence_contract_mismatch",
        capability_id=capability,
        artifact_ids=[fact.source_id] if fact.source_id else [],
        evidence_seen=[evidence_id(fact)],
        source_scopes=[fact.scope.to_prompt_dict()],
    )


def evidence_required_for_plan(plan: object) -> list[str]:
    required: list[str] = []
    for capability in getattr(plan, "answer_capabilities", []) or []:
        for manifest in manifests_for_capability(str(capability), plan):
            contract = effective_evidence_contract(manifest, plan)
            required.extend(contract.required_fact_types)
            required.extend(contract.any_of_fact_types)
    return ordered_unique(required)


def fact_is_relevant_to_capability(
    fact: EvidenceFact,
    capability: str,
    plan: object | None = None,
) -> bool:
    return any(
        fact.fact_type in set(effective_evidence_contract(manifest, plan).required_fact_types)
        | set(effective_evidence_contract(manifest, plan).any_of_fact_types)
        for manifest in manifests_for_capability(capability, plan)
    )


def manifests_for_capability(capability: str, plan: object | None = None):
    manifests = tuple(
        manifest
        for manifest in CAPABILITY_MANIFEST_REGISTRY.list()
        if manifest.runtime_capability == capability
    )
    selected_id = str(
        getattr(getattr(plan, "plan_spec", None), "selected_capability_id", "") or ""
    )
    if selected_id:
        selected = CAPABILITY_MANIFEST_REGISTRY.find(selected_id)
        if selected is not None and selected.runtime_capability == capability:
            return (selected,)
    mode = str(getattr(plan, "response_mode", "") or "")
    answer_capabilities = set(getattr(plan, "answer_capabilities", ()) or ())
    if capability in answer_capabilities:
        filtered = tuple(
            manifest
            for manifest in manifests
            if manifest.capability_type in {"answer", "summary"}
        )
        return filtered or manifests
    if mode == "action":
        filtered = tuple(
            manifest for manifest in manifests if manifest.capability_type == "action"
        )
        return filtered or manifests
    if mode == "knowledge_answer":
        filtered = tuple(
            manifest
            for manifest in manifests
            if manifest.capability_type in {"answer", "summary"}
        )
        return filtered or manifests
    if mode == "handoff":
        filtered = tuple(
            manifest for manifest in manifests if manifest.capability_type == "handoff"
        )
        return filtered or manifests
    return manifests


def contract_rejection_code(
    *,
    fact: EvidenceFact,
    capability: str,
    plan: object,
    domain_context: DomainContext | None,
    manifest_id: str,
    allowed_fact_types: tuple[str, ...],
    allowed_source_types: tuple[str, ...],
    forbidden_source_types: tuple[str, ...],
    allow_history: bool,
    allowed_artifact_types: tuple[str, ...],
    required_artifact_types: tuple[str, ...],
    required_scope_match: tuple[str, ...],
    provenance_required: bool,
) -> str:
    del manifest_id
    if allowed_fact_types and fact.fact_type not in allowed_fact_types:
        return "fact_type_not_allowed"
    if fact.value is False or fact.value is None:
        return "evidence_value_empty"
    source_metadata = source_metadata_for_fact(fact)
    if not allow_history and is_history_fact(fact):
        return "history_source_not_current_artifact"
    if allow_history and is_history_fact(fact):
        history_scope_code = history_scope_rejection_code(
            fact,
            plan,
            domain_context,
            required_scope_match,
        )
        if history_scope_code:
            return history_scope_code
        if provenance_required and source_provenance_missing(fact):
            return "evidence_provenance_missing"
    if source_metadata is not None and not source_metadata.evidence_allowed_by_default:
        if not allow_history:
            return "source_not_evidence_by_default"
    if fact.source_type in forbidden_source_types:
        return "forbidden_source_type"
    if allowed_source_types and fact.source_type not in allowed_source_types:
        return "source_type_not_allowed"
    artifact_types = contract_artifact_types(
        required_artifact_types or allowed_artifact_types
    )
    if artifact_types and evidence_artifact_type(fact) not in artifact_types:
        return "artifact_type_not_allowed"
    if wrong_channel(fact.scope, domain_context):
        return "channel_scope_mismatch"
    if wrong_strategy(fact, plan, domain_context):
        return "strategy_scope_mismatch"
    if wrong_time_range(fact.scope, requested_scope_from_plan(plan)):
        return "time_range_scope_mismatch"
    if capability in {"weekly_report", "monthly_report"} and fact.resolve_type != capability:
        return "resolve_type_mismatch"
    return ""


def effective_evidence_contract(manifest, plan: object | None) -> EvidenceContract:
    plan_spec = getattr(plan, "plan_spec", None)
    if (
        plan_spec is not None
        and getattr(plan_spec, "selected_capability_id", None) == manifest.id
        and getattr(plan_spec, "evidence_contract", None) is not None
    ):
        return plan_spec.evidence_contract
    return manifest.evidence_contract


def history_scope_rejection_code(
    fact: EvidenceFact,
    plan: object,
    domain_context: DomainContext | None,
    required_scope_match: tuple[str, ...],
) -> str:
    fields = set(required_scope_match)
    if "channel_id" in fields and history_channel_mismatch(fact, domain_context):
        return "channel_scope_mismatch"
    if (
        {"strategy_id", "strategy_name"} & fields
        and wrong_strategy(fact, plan, domain_context)
    ):
        return "strategy_scope_mismatch"
    if "time_range" in fields and history_time_range_mismatch_or_missing(fact, plan):
        return "time_range_scope_mismatch"
    return ""


def history_channel_mismatch(
    fact: EvidenceFact,
    domain_context: DomainContext | None,
) -> bool:
    if domain_context is None:
        return False
    source_metadata = source_metadata_for_fact(fact)
    actual = (
        source_metadata.channel_id
        if source_metadata is not None and source_metadata.channel_id
        else fact.scope.channel_id
    )
    if not actual or actual == "unknown":
        return True
    allowed = {
        domain_context.channel.id,
        domain_context.channel.name,
        adapter_channel_id(domain_context),
    }
    return actual not in allowed


def history_time_range_mismatch_or_missing(
    fact: EvidenceFact,
    plan: object,
) -> bool:
    expected = expected_time_range_from_plan(plan)
    if expected is None:
        return False
    source_metadata = source_metadata_for_fact(fact)
    actual = (
        source_metadata.time_range
        if source_metadata is not None and source_metadata.time_range is not None
        else fact.scope.time_range
    )
    if actual is None:
        return True
    for attr in ("period", "start", "end"):
        expected_value = getattr(expected, attr)
        actual_value = getattr(actual, attr)
        if expected_value and not actual_value:
            return True
        if expected_value and actual_value and expected_value != actual_value:
            return True
    return False


def expected_time_range_from_plan(plan: object) -> TimeRange | None:
    plan_spec = getattr(plan, "plan_spec", None)
    if plan_spec is not None:
        domain_scope = getattr(plan_spec, "domain_scope", None)
        if domain_scope is not None and getattr(domain_scope, "time_range", None) is not None:
            return domain_scope.time_range
    scope = requested_scope_from_plan(plan)
    if scope is None or not scope.period:
        return None
    return TimeRange(
        period=scope.period,
        start=scope.time_range_start,
        end=scope.time_range_end,
    )


def wrong_channel(
    scope: ArtifactScope,
    domain_context: DomainContext | None,
) -> bool:
    if domain_context is None or scope.channel_id in {"", "unknown"}:
        return False
    allowed = {
        domain_context.channel.id,
        domain_context.channel.name,
        adapter_channel_id(domain_context),
    }
    return scope.channel_id not in allowed


def wrong_strategy(
    fact: EvidenceFact,
    plan: object,
    domain_context: DomainContext | None,
) -> bool:
    expected_name = str(getattr(plan, "selected_strategy", "") or "").strip()
    if not expected_name:
        return False
    expected_ids = {expected_name}
    if domain_context is not None:
        expected_ids.update(
            strategy.id
            for strategy in domain_context.strategies
            if strategy.name == expected_name
        )
    actual_strategy_name = str(fact.metadata.get("strategy") or "").strip()
    if actual_strategy_name:
        return actual_strategy_name != expected_name
    actual_strategy_id = fact.scope.strategy_id
    if domain_context is not None and actual_strategy_id and actual_strategy_id not in expected_ids:
        return True
    return False


def wrong_time_range(
    scope: ArtifactScope,
    requested: RequestedScope | None,
) -> bool:
    if requested is None or not requested.period:
        return False
    time_range = scope.time_range or TimeRange()
    return bool(time_range.period and time_range.period != requested.period)


def requested_scope_from_plan(plan: object) -> RequestedScope | None:
    return requested_scope(plan)


def contract_artifact_types(values: tuple[str, ...]) -> set[str]:
    return {value for value in values if value}
