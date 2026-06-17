from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from market_support_crewai_agent.runtime.domain.capabilities.adapters import (
    manifests_for_manifest_ids,
)
from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    CapabilityManifest,
    CapabilityRegistry,
    VerifierPrimitive,
)
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    evidence_artifact_type,
    infer_abstained,
    is_missing,
    lookup_path,
    payload_dict,
    schema_errors,
)

CapabilityContractSeverity = Literal["error", "fatal"]


@dataclass(frozen=True)
class CapabilityContractIssue:
    capability_id: str
    check: VerifierPrimitive
    message: str
    severity: CapabilityContractSeverity = "error"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityContractValidationResult:
    valid: bool
    issues: tuple[CapabilityContractIssue, ...] = ()


def verify_capability_manifest(
    manifest: CapabilityManifest,
    *,
    output_payload: object,
    runtime_inputs: Mapping[str, object] | object | None,
    evidence_facts: list[EvidenceFact],
    used_artifacts: list[str] | tuple[str, ...] = (),
    used_sources: list[str] | tuple[str, ...] = (),
    abstained: bool | None = None,
) -> CapabilityContractValidationResult:
    payload = payload_dict(output_payload)
    inferred_abstained = infer_abstained(payload) if abstained is None else abstained
    issues: list[CapabilityContractIssue] = []
    for check in manifest.verifier_checks:
        if check == "output_schema":
            issues.extend(_check_output_schema(manifest, payload))
        elif check == "required_runtime_input_present":
            issues.extend(_check_required_runtime_inputs(manifest, runtime_inputs))
        elif check == "required_evidence_present":
            issues.extend(_check_required_evidence(manifest, evidence_facts))
        elif check == "evidence_artifact_type_allowed":
            issues.extend(
                _check_evidence_artifacts(manifest, evidence_facts, used_artifacts)
            )
        elif check == "forbidden_source_not_used":
            issues.extend(
                _check_forbidden_sources(manifest, evidence_facts, used_sources)
            )
        elif check == "abstention_correctness":
            issues.extend(
                _check_abstention(
                    manifest,
                    evidence_facts,
                    abstained=bool(inferred_abstained),
                )
            )
    return CapabilityContractValidationResult(
        valid=not issues,
        issues=tuple(issues),
    )


def verify_capability_contracts(
    manifest_ids: list[str] | tuple[str, ...],
    *,
    output_payload: object,
    runtime_inputs: Mapping[str, object] | object | None,
    evidence_facts: list[EvidenceFact],
    registry: CapabilityRegistry | None = None,
    used_artifacts: list[str] | tuple[str, ...] = (),
    used_sources: list[str] | tuple[str, ...] = (),
    abstained: bool | None = None,
) -> CapabilityContractValidationResult:
    issues: list[CapabilityContractIssue] = []
    for manifest in manifests_for_manifest_ids(manifest_ids, registry):
        result = verify_capability_manifest(
            manifest,
            output_payload=output_payload,
            runtime_inputs=runtime_inputs,
            evidence_facts=evidence_facts,
            used_artifacts=used_artifacts,
            used_sources=used_sources,
            abstained=abstained,
        )
        issues.extend(result.issues)
    return CapabilityContractValidationResult(
        valid=not issues,
        issues=tuple(issues),
    )


def _check_output_schema(
    manifest: CapabilityManifest,
    payload: dict[str, object],
) -> list[CapabilityContractIssue]:
    errors = schema_errors(manifest.output_schema, payload)
    return [
        CapabilityContractIssue(
            capability_id=manifest.id,
            check="output_schema",
            message=error,
            severity="fatal",
        )
        for error in errors
    ]


def _check_required_runtime_inputs(
    manifest: CapabilityManifest,
    runtime_inputs: Mapping[str, object] | object | None,
) -> list[CapabilityContractIssue]:
    missing = [
        input_name
        for input_name in manifest.required_inputs
        if is_missing(lookup_path(runtime_inputs, input_name))
    ]
    if not missing:
        return []
    return [
        CapabilityContractIssue(
            capability_id=manifest.id,
            check="required_runtime_input_present",
            message="required runtime inputs are missing",
            severity="fatal",
            metadata={"missing_inputs": missing},
        )
    ]


def _check_required_evidence(
    manifest: CapabilityManifest,
    evidence_facts: list[EvidenceFact],
) -> list[CapabilityContractIssue]:
    missing = _missing_evidence(manifest, evidence_facts)
    if not missing:
        return []
    return [
        CapabilityContractIssue(
            capability_id=manifest.id,
            check="required_evidence_present",
            message="required evidence is missing",
            metadata=missing,
        )
    ]


def _check_evidence_artifacts(
    manifest: CapabilityManifest,
    evidence_facts: list[EvidenceFact],
    used_artifacts: list[str] | tuple[str, ...],
) -> list[CapabilityContractIssue]:
    allowed = set(manifest.allowed_artifacts)
    if not allowed:
        return []

    artifact_types = set(str(item) for item in used_artifacts if str(item))
    artifact_types.update(
        artifact_type
        for fact in _relevant_facts(manifest, evidence_facts)
            if (artifact_type := evidence_artifact_type(fact))
    )
    disallowed = sorted(artifact_types - allowed)
    if not disallowed:
        return []
    return [
        CapabilityContractIssue(
            capability_id=manifest.id,
            check="evidence_artifact_type_allowed",
            message="evidence artifact type is not allowed for capability",
            metadata={"disallowed_artifact_types": disallowed},
        )
    ]


def _check_forbidden_sources(
    manifest: CapabilityManifest,
    evidence_facts: list[EvidenceFact],
    used_sources: list[str] | tuple[str, ...],
) -> list[CapabilityContractIssue]:
    forbidden = set(manifest.evidence_contract.forbidden_source_types)
    if not forbidden:
        return []
    source_types = set(str(item) for item in used_sources if str(item))
    source_types.update(
        fact.source_type
        for fact in _relevant_facts(manifest, evidence_facts)
    )
    used_forbidden = sorted(source_types & forbidden)
    if not used_forbidden:
        return []
    return [
        CapabilityContractIssue(
            capability_id=manifest.id,
            check="forbidden_source_not_used",
            message="forbidden evidence source was used",
            metadata={"forbidden_source_types": used_forbidden},
        )
    ]


def _check_abstention(
    manifest: CapabilityManifest,
    evidence_facts: list[EvidenceFact],
    *,
    abstained: bool,
) -> list[CapabilityContractIssue]:
    if not manifest.abstention_policy.requires_abstention_when_evidence_missing:
        return []
    missing = _missing_evidence(manifest, evidence_facts)
    if not missing or abstained:
        return []
    return [
        CapabilityContractIssue(
            capability_id=manifest.id,
            check="abstention_correctness",
            message="capability must abstain when required evidence is missing",
            metadata=missing,
        )
    ]


def _missing_evidence(
    manifest: CapabilityManifest,
    evidence_facts: list[EvidenceFact],
) -> dict[str, object]:
    truthy_fact_types = {
        str(fact.fact_type)
        for fact in evidence_facts
        if bool(fact.value)
    }
    required = [
        fact_type
        for fact_type in manifest.evidence_contract.required_fact_types
        if fact_type not in truthy_fact_types
    ]
    any_of = list(manifest.evidence_contract.any_of_fact_types)
    any_of_missing = bool(any_of) and not any(
        fact_type in truthy_fact_types for fact_type in any_of
    )
    fact_count = sum(
        1
        for fact in _relevant_facts(manifest, evidence_facts)
        if bool(fact.value)
    )
    below_min = (
        manifest.evidence_contract.min_facts > 0
        and fact_count < manifest.evidence_contract.min_facts
    )
    missing: dict[str, object] = {}
    if required:
        missing["missing_fact_types"] = required
    if any_of_missing:
        missing["missing_any_of_fact_types"] = any_of
    if below_min:
        missing["min_facts"] = manifest.evidence_contract.min_facts
        missing["actual_facts"] = fact_count
    return missing


def _relevant_facts(
    manifest: CapabilityManifest,
    evidence_facts: list[EvidenceFact],
) -> list[EvidenceFact]:
    contract = manifest.evidence_contract
    fact_types = set(contract.required_fact_types) | set(contract.any_of_fact_types)
    if fact_types:
        return [fact for fact in evidence_facts if str(fact.fact_type) in fact_types]
    if contract.allowed_source_types:
        return [
            fact
            for fact in evidence_facts
            if fact.source_type in contract.allowed_source_types
        ]
    return list(evidence_facts)
