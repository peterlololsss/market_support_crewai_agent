from __future__ import annotations

import hashlib
import json
from typing import Final, TypedDict

from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityAbstentionPolicyV2,
    CapabilityManifestV2,
    HistoryConstraintsV1,
    ManifestRefV1,
    StaleDataPolicyV1,
)
from market_support_crewai_agent.runtime.policy.capabilities.evidence_vocabulary import (
    EvidenceArtifactTypeV2,
    EvidenceFactTypeV2,
    EvidenceScopeMatchFieldV2,
    FallbackPolicyV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    CapabilityTypeV1,
    CapabilityViewOutputSchemaV1,
    ModelEvidenceContractV1,
)

_ARTIFACT_TYPES: Final[tuple[EvidenceArtifactTypeV2, ...]] = (
    "material_pack",
    "weekly_report",
    "monthly_report",
    "document_context",
    "adapter_context",
    "history",
    "user_upload",
    "unknown",
)
_CAPABILITY_TYPE_BY_MANIFEST: Final[dict[str, CapabilityTypeV1]] = {
    "material_pack.send": "send",
    "weekly_report.send": "send",
    "monthly_report.send": "send",
    "sales.handoff": "handoff",
    "general.clarification": "clarification",
    "general.abstention": "abstention",
    "general.refusal": "refusal",
    "general.smalltalk": "smalltalk",
    "general.no_reply": "no_reply",
    "weekly_report.product_list": "answer",
    "monthly_report.product_list": "answer",
    "answer_internal_company_knowledge": "answer",
    "general.handoff": "handoff",
}
_RUNTIME_CAPABILITY_BY_MANIFEST: Final[dict[str, str]] = {
    "material_pack.send": "material_pack",
    "weekly_report.send": "weekly_report",
    "monthly_report.send": "monthly_report",
    "sales.handoff": "sales_mention",
    "answer_internal_company_knowledge": "document_context",
}
_CANONICAL_REPLY_OUTPUT_SCHEMA: Final[CapabilityViewOutputSchemaV1] = {
    "type": "object",
    "required": ["reply", "actions"],
    "properties": {
        "reply": {"type": "object"},
        "actions": {"type": "array"},
    },
}


class _FlatEvidenceFields(TypedDict):
    manifest_ref: ManifestRefV1
    capability_type: CapabilityTypeV1
    internal_company_knowledge_required: bool
    required_artifacts: tuple[EvidenceArtifactTypeV2, ...]
    allowed_artifacts: tuple[EvidenceArtifactTypeV2, ...]
    forbidden_artifacts: tuple[EvidenceArtifactTypeV2, ...]
    required_fact_types: tuple[EvidenceFactTypeV2, ...]
    any_of_fact_types: tuple[EvidenceFactTypeV2, ...]
    allowed_fact_types: tuple[EvidenceFactTypeV2, ...]
    forbidden_fact_types: tuple[EvidenceFactTypeV2, ...]
    required_artifact_types: tuple[EvidenceArtifactTypeV2, ...]
    allowed_artifact_types: tuple[EvidenceArtifactTypeV2, ...]
    required_scope_match: tuple[EvidenceScopeMatchFieldV2, ...]
    min_facts: int
    allow_history: bool
    history_constraints: HistoryConstraintsV1
    fallback_policy: FallbackPolicyV2
    citation_required: bool
    provenance_required: bool
    citation_requirements: tuple[str, ...]
    stale_data_policy: StaleDataPolicyV1
    abstention_policy: CapabilityAbstentionPolicyV2


class _PlannerRoleFields(TypedDict):
    manifest_ref: ManifestRefV1
    display_name: str
    description: str
    capability_type: CapabilityTypeV1
    runtime_capability: str | None
    domain_entities: tuple[str, ...]
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    required_artifacts: tuple[EvidenceArtifactTypeV2, ...]
    allowed_artifacts: tuple[EvidenceArtifactTypeV2, ...]
    forbidden_artifacts: tuple[EvidenceArtifactTypeV2, ...]


def _model_evidence_contract(
    manifest: CapabilityManifestV2,
) -> ModelEvidenceContractV1:
    contract = manifest.evidence_contract
    return ModelEvidenceContractV1(
        required_fact_types=contract.required_fact_types,
        any_of_fact_types=contract.any_of_fact_types,
        allowed_fact_types=contract.allowed_fact_types,
        forbidden_fact_types=contract.forbidden_fact_types,
        required_artifact_types=contract.required_artifact_types,
        allowed_artifact_types=contract.allowed_artifact_types,
        required_scope_match=contract.required_scope_match,
        min_facts=contract.min_facts,
        allow_history=contract.allow_history,
        history_constraints=contract.history_constraints,
        fallback_policy=contract.fallback_policy,
        citation_required=contract.citation_required,
        provenance_required=contract.provenance_required,
        citation_requirements=contract.citation_requirements,
        stale_data_policy=contract.stale_data_policy,
    )


def _planner_role_fields(
    ref: ManifestRefV1,
    manifest: CapabilityManifestV2,
) -> _PlannerRoleFields:
    contract = manifest.evidence_contract
    allowed_artifacts = contract.allowed_artifact_types
    return {
        "manifest_ref": ref,
        "display_name": manifest.manifest_id,
        "description": manifest.selection_contract.summary,
        "capability_type": _CAPABILITY_TYPE_BY_MANIFEST[manifest.manifest_id],
        "runtime_capability": _RUNTIME_CAPABILITY_BY_MANIFEST.get(manifest.manifest_id),
        "domain_entities": ("artifact",) if allowed_artifacts else (),
        "required_inputs": (),
        "optional_inputs": (),
        "required_artifacts": contract.required_artifact_types,
        "allowed_artifacts": allowed_artifacts,
        "forbidden_artifacts": _forbidden_artifacts(allowed_artifacts),
    }


def _flat_evidence_fields(
    ref: ManifestRefV1,
    manifest: CapabilityManifestV2,
) -> _FlatEvidenceFields:
    contract = manifest.evidence_contract
    allowed_artifacts = contract.allowed_artifact_types
    return {
        "manifest_ref": ref,
        "capability_type": _CAPABILITY_TYPE_BY_MANIFEST[manifest.manifest_id],
        "internal_company_knowledge_required": (
            manifest.manifest_id == "answer_internal_company_knowledge"
        ),
        "required_artifacts": contract.required_artifact_types,
        "allowed_artifacts": allowed_artifacts,
        "forbidden_artifacts": _forbidden_artifacts(allowed_artifacts),
        "required_fact_types": contract.required_fact_types,
        "any_of_fact_types": contract.any_of_fact_types,
        "allowed_fact_types": contract.allowed_fact_types,
        "forbidden_fact_types": contract.forbidden_fact_types,
        "required_artifact_types": contract.required_artifact_types,
        "allowed_artifact_types": contract.allowed_artifact_types,
        "required_scope_match": contract.required_scope_match,
        "min_facts": contract.min_facts,
        "allow_history": contract.allow_history,
        "history_constraints": contract.history_constraints,
        "fallback_policy": contract.fallback_policy,
        "citation_required": contract.citation_required,
        "provenance_required": contract.provenance_required,
        "citation_requirements": contract.citation_requirements,
        "stale_data_policy": contract.stale_data_policy,
        "abstention_policy": manifest.abstention_policy,
    }


def _forbidden_artifacts(
    allowed_artifacts: tuple[EvidenceArtifactTypeV2, ...],
) -> tuple[EvidenceArtifactTypeV2, ...]:
    return tuple(
        artifact for artifact in _ARTIFACT_TYPES if artifact not in allowed_artifacts
    )


def _canonical_output_schema() -> CapabilityViewOutputSchemaV1:
    return _CANONICAL_REPLY_OUTPUT_SCHEMA


def _output_schema_hash() -> str:
    encoded = json.dumps(
        _CANONICAL_REPLY_OUTPUT_SCHEMA,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"osh1:{hashlib.sha256(b'output-schema.v1\\0' + encoded).hexdigest()}"
