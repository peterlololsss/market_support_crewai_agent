from __future__ import annotations

from typing import Final, Literal

from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestV2,
    CapabilityName,
    VerifierPrimitiveV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.evidence_vocabulary import (
    EvidenceArtifactTypeV2,
)

ManifestCapabilityKindV2 = Literal[
    "action",
    "answer",
    "summary",
    "handoff",
    "control",
]

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
_CANONICAL_REPLY_OUTPUT_SCHEMA: Final[
    dict[str, str | list[str] | dict[str, dict[str, str]]]
] = {
    "type": "object",
    "required": ["reply", "actions"],
    "properties": {
        "reply": {"type": "object"},
        "actions": {"type": "array"},
    },
}


def runtime_capability_for_manifest(
    manifest: CapabilityManifestV2,
) -> CapabilityName | None:
    match manifest.manifest_id:
        case "material_pack.send":
            return "material_pack"
        case "weekly_report.send" | "weekly_report.product_list":
            return "weekly_report"
        case "monthly_report.send" | "monthly_report.product_list":
            return "monthly_report"
        case "sales.handoff":
            return "sales_mention"
        case "answer_internal_company_knowledge":
            return "document_context"
        case (
            "general.clarification"
            | "general.abstention"
            | "general.refusal"
            | "general.smalltalk"
            | "general.no_reply"
            | "general.handoff"
        ):
            return None
        case unreachable:
            raise AssertionError(f"unreachable manifest id: {unreachable}")


def capability_kind_for_manifest(
    manifest: CapabilityManifestV2,
) -> ManifestCapabilityKindV2:
    match manifest.manifest_id:
        case "material_pack.send" | "weekly_report.send" | "monthly_report.send":
            return "action"
        case "answer_internal_company_knowledge":
            return "answer"
        case "weekly_report.product_list" | "monthly_report.product_list":
            return "summary"
        case "sales.handoff" | "general.handoff":
            return "handoff"
        case (
            "general.clarification"
            | "general.abstention"
            | "general.refusal"
            | "general.smalltalk"
            | "general.no_reply"
        ):
            return "control"
        case unreachable:
            raise AssertionError(f"unreachable manifest id: {unreachable}")


def adapter_tools_for_manifest(manifest: CapabilityManifestV2) -> tuple[str, ...]:
    match manifest.manifest_id:
        case "material_pack.send":
            return ("adapter_resolve.material_pack",)
        case "weekly_report.send":
            return ("adapter_resolve.weekly_report",)
        case "monthly_report.send":
            return ("adapter_resolve.monthly_report",)
        case "sales.handoff":
            return ("adapter_resolve.sales_mention",)
        case (
            "general.clarification"
            | "general.abstention"
            | "general.refusal"
            | "general.smalltalk"
            | "general.no_reply"
            | "answer_internal_company_knowledge"
            | "weekly_report.product_list"
            | "monthly_report.product_list"
            | "general.handoff"
        ):
            return ()
        case unreachable:
            raise AssertionError(f"unreachable manifest id: {unreachable}")


def required_inputs_for_manifest(manifest: CapabilityManifestV2) -> tuple[str, ...]:
    del manifest
    return ()


def required_artifacts_for_manifest(
    manifest: CapabilityManifestV2,
) -> tuple[EvidenceArtifactTypeV2, ...]:
    return manifest.evidence_contract.required_artifact_types


def allowed_artifacts_for_manifest(
    manifest: CapabilityManifestV2,
) -> tuple[EvidenceArtifactTypeV2, ...]:
    return manifest.evidence_contract.allowed_artifact_types


def forbidden_artifacts_for_manifest(
    manifest: CapabilityManifestV2,
) -> tuple[EvidenceArtifactTypeV2, ...]:
    allowed = set(allowed_artifacts_for_manifest(manifest))
    return tuple(artifact for artifact in _ARTIFACT_TYPES if artifact not in allowed)


def verifier_primitives_for_manifest(
    manifest: CapabilityManifestV2,
) -> tuple[VerifierPrimitiveV2, ...]:
    return manifest.verifier_primitives


def output_schema_for_manifest(
    manifest: CapabilityManifestV2,
) -> dict[str, str | list[str] | dict[str, dict[str, str]]]:
    del manifest
    return _CANONICAL_REPLY_OUTPUT_SCHEMA
