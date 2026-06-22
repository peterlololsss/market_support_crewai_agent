from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    CapabilityRegistry,
    EvidenceContract,
)
from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec, PlanUnit
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    evidence_artifact_type,
    evidence_id,
    infer_abstained,
    is_history_fact,
    payload_dict,
    reply_kind,
    schema_errors,
    source_metadata_for_fact,
    source_provenance_missing,
    string_values,
)

PlanSpecValidationCode = Literal[
    "plan_spec_invalid_schema",
    "selected_capability_not_found",
    "required_artifact_missing",
    "forbidden_artifact_used",
    "output_schema_invalid",
    "required_evidence_missing",
    "evidence_count_below_minimum",
    "evidence_source_disallowed",
    "evidence_artifact_disallowed",
    "history_evidence_not_allowed",
    "evidence_scope_mismatch",
    "evidence_provenance_missing",
    "citation_required",
    "abstention_output_invalid",
]
PlanSpecValidationSeverity = Literal["error", "fatal"]


@dataclass(frozen=True)
class PlanSpecValidationIssue:
    code: PlanSpecValidationCode
    message: str
    severity: PlanSpecValidationSeverity = "error"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanSpecValidationResult:
    valid: bool
    issues: tuple[PlanSpecValidationIssue, ...] = ()


def verify_plan_spec(
    plan_spec: PlanSpec | Mapping[str, object],
    *,
    output_payload: object,
    evidence_facts: list[EvidenceFact],
    runtime_inputs: Mapping[str, object] | object | None = None,
    registry: CapabilityRegistry | None = None,
    available_artifacts: list[str] | tuple[str, ...] | None = None,
    used_artifacts: list[str] | tuple[str, ...] = (),
    cited_evidence_ids: list[str] | tuple[str, ...] | None = None,
    abstained: bool | None = None,
) -> PlanSpecValidationResult:
    del runtime_inputs
    try:
        spec = (
            plan_spec
            if isinstance(plan_spec, PlanSpec)
            else PlanSpec.model_validate(plan_spec)
        )
    except ValueError as exc:
        return PlanSpecValidationResult(
            valid=False,
            issues=(
                PlanSpecValidationIssue(
                    code="plan_spec_invalid_schema",
                    message="PlanSpec schema validation failed",
                    severity="fatal",
                    metadata={"error": str(exc)},
                ),
            ),
        )

    selected_registry = registry or CAPABILITY_MANIFEST_REGISTRY
    payload = payload_dict(output_payload)
    inferred_abstained = infer_abstained(payload) if abstained is None else abstained
    issues: list[PlanSpecValidationIssue] = []
    for unit in spec.plan_units:
        manifest = selected_registry.find(unit.selected_capability_id)
        if manifest is None:
            issues.append(
                PlanSpecValidationIssue(
                    code="selected_capability_not_found",
                    message="PlanSpec selected capability does not exist",
                    severity="fatal",
                    metadata={
                        "unit_id": unit.unit_id,
                        "selected_capability_id": unit.selected_capability_id,
                    },
                )
            )
            continue
        contract = manifest.evidence_contract
        output_schema = unit.output_schema or manifest.output_schema
        unit_issues: list[PlanSpecValidationIssue] = []
        unit_issues.extend(_check_output_schema(output_schema, payload))
        unit_issues.extend(_check_step_artifacts(unit))
        unit_issues.extend(
            _check_required_artifacts(
                unit,
                evidence_facts,
                available_artifacts=available_artifacts,
                abstained=bool(inferred_abstained),
                allowed_abstention_kinds=manifest.abstention_policy.abstention_reply_kinds,
                payload=payload,
                contract=contract,
            )
        )
        unit_issues.extend(
            _check_evidence_contract(
                unit,
                contract,
                evidence_facts,
                cited_evidence_ids=cited_evidence_ids,
                used_artifacts=used_artifacts,
                abstained=bool(inferred_abstained),
                payload=payload,
            )
        )
        issues.extend(_with_unit_metadata(unit, unit_issues))
    return PlanSpecValidationResult(valid=not issues, issues=tuple(issues))


def _with_unit_metadata(
    unit: PlanUnit,
    issues: list[PlanSpecValidationIssue],
) -> list[PlanSpecValidationIssue]:
    return [
        PlanSpecValidationIssue(
            code=issue.code,
            message=issue.message,
            severity=issue.severity,
            metadata={
                "unit_id": unit.unit_id,
                "selected_capability_id": unit.selected_capability_id,
                **issue.metadata,
            },
        )
        for issue in issues
    ]


def _check_output_schema(
    schema: Mapping[str, object],
    payload: dict[str, object],
) -> list[PlanSpecValidationIssue]:
    return [
        PlanSpecValidationIssue(
            code="output_schema_invalid",
            message=error,
            severity="fatal",
        )
        for error in schema_errors(schema, payload)
    ]


def _check_step_artifacts(unit: PlanUnit) -> list[PlanSpecValidationIssue]:
    forbidden = set(unit.forbidden_artifacts)
    issues: list[PlanSpecValidationIssue] = []
    for step in unit.steps:
        used = set(step.uses_artifacts) | set(step.required_artifacts)
        forbidden_used = sorted(used & forbidden)
        if forbidden_used:
            issues.append(
                PlanSpecValidationIssue(
                    code="forbidden_artifact_used",
                    message="PlanSpec step uses forbidden artifact types",
                    metadata={
                        "step_id": step.step_id,
                        "forbidden_artifacts": forbidden_used,
                    },
                )
            )
    return issues


def _check_required_artifacts(
    unit: PlanUnit,
    evidence_facts: list[EvidenceFact],
    *,
    available_artifacts: list[str] | tuple[str, ...] | None,
    abstained: bool,
    allowed_abstention_kinds: list[str],
    payload: dict[str, object],
    contract: EvidenceContract,
) -> list[PlanSpecValidationIssue]:
    required = set(unit.required_artifacts)
    if not required:
        return []
    available = set(available_artifacts or _available_artifact_types(evidence_facts))
    missing = sorted(required - available)
    if not missing:
        return []
    if _missing_evidence_abstention_allowed(
        unit,
        payload,
        abstained=abstained,
        allowed_abstention_kinds=allowed_abstention_kinds,
        contract=contract,
    ):
        return []
    return [
        PlanSpecValidationIssue(
            code="required_artifact_missing",
            message="required artifacts are missing and the plan did not abstain",
            metadata={"missing_artifacts": missing},
        )
    ]


def _check_evidence_contract(
    unit: PlanUnit,
    contract: EvidenceContract,
    evidence_facts: list[EvidenceFact],
    *,
    cited_evidence_ids: list[str] | tuple[str, ...] | None,
    used_artifacts: list[str] | tuple[str, ...],
    abstained: bool,
    payload: dict[str, object],
) -> list[PlanSpecValidationIssue]:
    if _missing_evidence_abstention_allowed(
        unit,
        payload,
        abstained=abstained,
        allowed_abstention_kinds=[
            "unable_to_answer",
            "clarification",
            "human_handoff",
            "no_reply",
        ],
        contract=contract,
    ):
        return []

    cited_ids = _cited_ids_for_unit(
        unit,
        tuple(cited_evidence_ids or _cited_evidence_ids(payload)),
    )
    if contract.citation_required and not cited_ids:
        return [
            PlanSpecValidationIssue(
                code="citation_required",
                message="EvidenceContract requires citations or evidence references",
            )
        ]
    if not cited_ids and not _contract_requires_evidence(contract) and not used_artifacts:
        return []

    candidate_facts = (
        _facts_by_ids(evidence_facts, cited_ids)
        if cited_ids
        else _relevant_facts(contract, evidence_facts)
    )
    issues: list[PlanSpecValidationIssue] = []
    issues.extend(_check_required_evidence(contract, candidate_facts))
    issues.extend(_check_sources(contract, candidate_facts))
    issues.extend(
        _check_artifact_types(
            unit,
            contract,
            candidate_facts,
            used_artifacts=used_artifacts,
        )
    )
    issues.extend(_check_scope_match(unit, contract, candidate_facts))
    issues.extend(_check_provenance(contract, candidate_facts))
    return issues


def _contract_requires_evidence(contract: EvidenceContract) -> bool:
    return bool(
        contract.required_evidence_types
        or contract.required_fact_types
        or contract.any_of_fact_types
        or contract.minimum_evidence_count > 0
        or contract.min_facts > 0
        or contract.allowed_source_types
        or contract.allowed_artifact_types
        or contract.required_scope_match
    )


def _check_required_evidence(
    contract: EvidenceContract,
    facts: list[EvidenceFact],
) -> list[PlanSpecValidationIssue]:
    truthy_fact_types = {str(fact.fact_type) for fact in facts if bool(fact.value)}
    required_types = contract.required_evidence_types or contract.required_fact_types
    missing_required = [
        fact_type for fact_type in required_types if fact_type not in truthy_fact_types
    ]
    any_of_missing = bool(contract.any_of_fact_types) and not any(
        fact_type in truthy_fact_types for fact_type in contract.any_of_fact_types
    )
    minimum = contract.minimum_evidence_count or contract.min_facts
    truthy_count = sum(1 for fact in facts if bool(fact.value))
    issues: list[PlanSpecValidationIssue] = []
    if missing_required:
        issues.append(
            PlanSpecValidationIssue(
                code="required_evidence_missing",
                message="required evidence types are missing",
                metadata={"missing_evidence_types": missing_required},
            )
        )
    if any_of_missing:
        issues.append(
            PlanSpecValidationIssue(
                code="required_evidence_missing",
                message="none of the allowed any-of evidence types are present",
                metadata={"missing_any_of_evidence_types": contract.any_of_fact_types},
            )
        )
    if minimum > 0 and truthy_count < minimum:
        issues.append(
            PlanSpecValidationIssue(
                code="evidence_count_below_minimum",
                message="minimum evidence count was not satisfied",
                metadata={"minimum": minimum, "actual": truthy_count},
            )
        )
    return issues


def _check_sources(
    contract: EvidenceContract,
    facts: list[EvidenceFact],
) -> list[PlanSpecValidationIssue]:
    issues: list[PlanSpecValidationIssue] = []
    allowed = set(contract.allowed_source_types)
    disallowed = set(contract.disallowed_source_types) | set(
        contract.forbidden_source_types
    )
    for fact in facts:
        source_metadata = source_metadata_for_fact(fact)
        if not contract.allow_history and is_history_fact(fact):
            issues.append(
                PlanSpecValidationIssue(
                    code="history_evidence_not_allowed",
                    message="EvidenceContract does not allow history evidence",
                    metadata={"source_id": fact.source_id, "source_type": fact.source_type},
                )
            )
            continue
        if (
            contract.allow_history
            and is_history_fact(fact)
            and contract.provenance_required
            and source_provenance_missing(fact)
        ):
            issues.append(
                PlanSpecValidationIssue(
                    code="evidence_provenance_missing",
                    message="history evidence lacks sufficient provenance",
                    metadata={"source_id": fact.source_id, "source_type": fact.source_type},
                )
            )
            continue
        if (
            source_metadata is not None
            and not source_metadata.evidence_allowed_by_default
            and not contract.allow_history
        ):
            issues.append(
                PlanSpecValidationIssue(
                    code="history_evidence_not_allowed",
                    message="source metadata marks this context item as non-evidence",
                    metadata={
                        "source_id": fact.source_id,
                        "source_type": fact.source_type,
                        "source_metadata_type": source_metadata.source_type,
                    },
                )
            )
            continue
        if fact.source_type in disallowed:
            issues.append(
                PlanSpecValidationIssue(
                    code="evidence_source_disallowed",
                    message="evidence source type is disallowed",
                    metadata={
                        "source_id": fact.source_id,
                        "source_type": fact.source_type,
                    },
                )
            )
            continue
        if allowed and fact.source_type not in allowed:
            issues.append(
                PlanSpecValidationIssue(
                    code="evidence_source_disallowed",
                    message="evidence source type is outside allowed sources",
                    metadata={
                        "source_id": fact.source_id,
                        "source_type": fact.source_type,
                        "allowed_source_types": sorted(allowed),
                    },
                )
            )
    return issues


def _check_artifact_types(
    unit: PlanUnit,
    contract: EvidenceContract,
    facts: list[EvidenceFact],
    *,
    used_artifacts: list[str] | tuple[str, ...],
) -> list[PlanSpecValidationIssue]:
    allowed = set(contract.allowed_artifact_types or unit.allowed_artifacts)
    if not allowed:
        return []
    artifact_types = {str(item) for item in used_artifacts if str(item)}
    artifact_types.update(
        evidence_artifact_type(fact)
        for fact in facts
        if evidence_artifact_type(fact)
    )
    disallowed = sorted(artifact_types - allowed)
    if not disallowed:
        return []
    return [
        PlanSpecValidationIssue(
            code="evidence_artifact_disallowed",
            message="evidence artifact type is outside allowed artifacts",
            metadata={
                "disallowed_artifact_types": disallowed,
                "allowed_artifact_types": sorted(allowed),
            },
        )
    ]


def _check_scope_match(
    unit: PlanUnit,
    contract: EvidenceContract,
    facts: list[EvidenceFact],
) -> list[PlanSpecValidationIssue]:
    fields = set(contract.required_scope_match)
    if not fields:
        return []
    issues: list[PlanSpecValidationIssue] = []
    for fact in facts:
        mismatched = [
            field
            for field in sorted(fields)
            if _scope_field_mismatch(field, unit, fact)
        ]
        if mismatched:
            issues.append(
                PlanSpecValidationIssue(
                    code="evidence_scope_mismatch",
                    message="evidence scope does not satisfy EvidenceContract",
                    metadata={
                        "source_id": fact.source_id,
                        "fact_type": fact.fact_type,
                        "mismatched_fields": mismatched,
                        "plan_scope": unit.domain_scope.model_dump(
                            mode="json",
                            exclude_none=True,
                        ),
                        "evidence_scope": fact.scope.to_prompt_dict(),
                    },
                )
            )
    return issues


def _check_provenance(
    contract: EvidenceContract,
    facts: list[EvidenceFact],
) -> list[PlanSpecValidationIssue]:
    if not contract.provenance_required:
        return []
    issues: list[PlanSpecValidationIssue] = []
    for fact in facts:
        source_metadata = source_metadata_for_fact(fact)
        provenance = (
            str(source_metadata.provenance or "").strip()
            if source_metadata is not None
            else ""
        )
        if not provenance:
            provenance = str(fact.scope.provenance or "").strip()
        if provenance and provenance != "unknown":
            continue
        if str(fact.source_id or "").strip():
            continue
        issues.append(
            PlanSpecValidationIssue(
                code="evidence_provenance_missing",
                message="evidence lacks provenance or source id",
                metadata={"fact_type": fact.fact_type, "source_type": fact.source_type},
            )
        )
    return issues


def _missing_evidence_abstention_allowed(
    unit: PlanUnit,
    payload: dict[str, object],
    *,
    abstained: bool,
    allowed_abstention_kinds: list[str],
    contract: EvidenceContract,
) -> bool:
    if unit.answerability_policy not in {"answer", "send", "abstain", "clarify", "handoff"}:
        return False
    if contract.fallback_policy not in {"abstain", "clarify", "handoff"}:
        return False
    if not abstained:
        return False
    reply_kind_value = reply_kind(payload)
    if allowed_abstention_kinds and reply_kind_value not in set(allowed_abstention_kinds):
        return False
    return True


def _scope_field_mismatch(field: str, unit: PlanUnit, fact: EvidenceFact) -> bool:
    scope = unit.domain_scope
    source_metadata = source_metadata_for_fact(fact)
    if field == "channel_id":
        expected = scope.channel_id
        actual = (
            source_metadata.channel_id
            if source_metadata is not None and source_metadata.channel_id
            else fact.scope.channel_id
        )
        return bool(
            expected
            and expected != "unknown"
            and (actual in {"", "unknown"} or actual != expected)
        )
    if field == "channel_kind":
        expected = scope.channel_kind
        actual = str(fact.metadata.get("channel_kind") or fact.metadata.get("channel_type") or "")
        return bool(expected and expected != "unknown" and actual and actual != expected)
    if field == "material_pack_option":
        expected = scope.material_pack_option
        actual = str(fact.metadata.get("material_pack_option") or "")
        if not expected or not actual:
            return False
        return actual != expected
    if field == "product_id":
        expected_ids = set(scope.product_ids)
        if not expected_ids:
            return False
        actual_ids = set(
            source_metadata.product_ids
            if source_metadata is not None and source_metadata.product_ids
            else fact.scope.product_ids
        )
        return not expected_ids.intersection(actual_ids)
    if field == "product_ids":
        expected_ids = set(scope.product_ids)
        if not expected_ids:
            return False
        actual_ids = set(
            source_metadata.product_ids
            if source_metadata is not None and source_metadata.product_ids
            else fact.scope.product_ids
        )
        return not expected_ids <= actual_ids
    if field == "time_range":
        expected = scope.time_range
        actual = (
            source_metadata.time_range
            if source_metadata is not None and source_metadata.time_range is not None
            else fact.scope.time_range
        )
        if expected is None or actual is None:
            return expected is not None and is_history_fact(fact)
        for attr in ("period", "start", "end"):
            expected_value = getattr(expected, attr)
            actual_value = getattr(actual, attr)
            if expected_value and not actual_value and is_history_fact(fact):
                return True
            if expected_value and actual_value and expected_value != actual_value:
                return True
        return False
    if field == "artifact_type":
        return bool(
            unit.required_artifacts
            and evidence_artifact_type(fact) not in set(unit.required_artifacts)
        )
    return False


def _available_artifact_types(evidence_facts: list[EvidenceFact]) -> tuple[str, ...]:
    output: list[str] = []
    for fact in evidence_facts:
        if not fact.value:
            continue
        artifact_type = evidence_artifact_type(fact)
        if artifact_type and artifact_type != "unknown" and artifact_type not in output:
            output.append(artifact_type)
    return tuple(output)


def _relevant_facts(
    contract: EvidenceContract,
    evidence_facts: list[EvidenceFact],
) -> list[EvidenceFact]:
    fact_types = set(contract.required_evidence_types or contract.required_fact_types)
    fact_types.update(contract.any_of_fact_types)
    if fact_types:
        return [fact for fact in evidence_facts if str(fact.fact_type) in fact_types]
    if contract.allowed_source_types:
        return [
            fact for fact in evidence_facts if fact.source_type in contract.allowed_source_types
        ]
    return list(evidence_facts)


def _facts_by_ids(
    evidence_facts: list[EvidenceFact],
    evidence_ids: list[str] | tuple[str, ...],
) -> list[EvidenceFact]:
    wanted = set(evidence_ids)
    return [
        fact
        for fact in evidence_facts
        if fact.source_id in wanted or evidence_id(fact) in wanted
    ]


def _cited_ids_for_unit(
    unit: PlanUnit,
    cited_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if unit.answerability_policy == "answer":
        return cited_ids
    return ()


def _cited_evidence_ids(payload: dict[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("evidence_refs", "evidence_ids", "citations", "provenance"):
        values.extend(string_values(payload.get(key)))
    reply = payload.get("reply")
    if isinstance(reply, Mapping):
        for key in ("evidence_refs", "evidence_ids", "citations", "provenance"):
            values.extend(string_values(reply.get(key)))
    return tuple(dict.fromkeys(values))
