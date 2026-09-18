from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field, JsonValue, model_validator

from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityAbstentionPolicyV2,
    CapabilityComposerConstraintV2,
    CapabilitySelectionContractV2,
    HistoryConstraintsV1,
    ManifestRefV1,
    StaleDataPolicyV1,
    VerifierPrimitiveV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.evidence_vocabulary import (
    EvidenceArtifactTypeV2,
    EvidenceFactTypeV2,
    EvidenceScopeMatchFieldV2,
    FallbackPolicyV2,
)
from market_support_crewai_agent.schemas.base import StrictModel

CapabilityViewStageV1 = Literal[
    "planner_intent",
    "knowledge_composer",
    "smalltalk_composer",
    "alignment_verifier",
]
ProhibitedComposerOutputV1 = Literal["actions", "mentions", "media"]
CapabilityTypeV1 = Literal[
    "send",
    "handoff",
    "clarification",
    "abstention",
    "refusal",
    "smalltalk",
    "no_reply",
    "answer",
]
CapabilityViewOutputSchemaV1: TypeAlias = dict[str, JsonValue]
CapabilityViewBindingErrorCode = Literal[
    "planner_capability_view_refs_mismatch",
    "composer_capability_view_refs_mismatch",
    "verifier_capability_view_refs_mismatch",
    "view_hash_mismatch",
]


@dataclass(frozen=True, slots=True)
class CapabilityViewBindingError(ValueError):
    code: CapabilityViewBindingErrorCode

    def __str__(self) -> str:
        return self.code


def capability_view_hash(
    *,
    contract_version: str,
    stage: str,
    manifest_refs: tuple[ManifestRefV1, ...],
    items: tuple[_FrozenCapabilityViewModel, ...],
) -> str:
    payload = {
        "contract_version": contract_version,
        "stage": stage,
        "manifest_refs": [ref.model_dump(mode="json") for ref in manifest_refs],
        "items": [item.model_dump(mode="json") for item in items],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"cvh1:{hashlib.sha256(b'capability-view-set.v1\0' + encoded).hexdigest()}"


def _validate_view_hash(
    *,
    contract_version: str,
    stage: str,
    manifest_refs: tuple[ManifestRefV1, ...],
    items: tuple[_FrozenCapabilityViewModel, ...],
    visible_items_hash: str,
) -> None:
    expected = capability_view_hash(
        contract_version=contract_version,
        stage=stage,
        manifest_refs=manifest_refs,
        items=items,
    )
    if visible_items_hash != expected:
        raise CapabilityViewBindingError("view_hash_mismatch")


class _FrozenCapabilityViewModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelEvidenceContractV1(_FrozenCapabilityViewModel):
    required_fact_types: tuple[EvidenceFactTypeV2, ...]
    any_of_fact_types: tuple[EvidenceFactTypeV2, ...]
    allowed_fact_types: tuple[EvidenceFactTypeV2, ...]
    forbidden_fact_types: tuple[EvidenceFactTypeV2, ...]
    required_artifact_types: tuple[EvidenceArtifactTypeV2, ...]
    allowed_artifact_types: tuple[EvidenceArtifactTypeV2, ...]
    required_scope_match: tuple[EvidenceScopeMatchFieldV2, ...]
    min_facts: int = Field(ge=0, le=32)
    allow_history: bool
    history_constraints: HistoryConstraintsV1
    fallback_policy: FallbackPolicyV2
    citation_required: bool
    provenance_required: bool
    citation_requirements: tuple[str, ...]
    stale_data_policy: StaleDataPolicyV1


class PlannerCapabilityViewV1(_FrozenCapabilityViewModel):
    contract_version: Literal["planner-capability-view.v1"] = (
        "planner-capability-view.v1"
    )
    manifest_ref: ManifestRefV1
    display_name: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=512)
    capability_type: CapabilityTypeV1
    runtime_capability: str | None = Field(default=None, max_length=64)
    domain_entities: tuple[str, ...] = Field(max_length=16)
    required_inputs: tuple[str, ...] = Field(max_length=16)
    optional_inputs: tuple[str, ...] = Field(max_length=16)
    required_artifacts: tuple[EvidenceArtifactTypeV2, ...] = Field(max_length=16)
    allowed_artifacts: tuple[EvidenceArtifactTypeV2, ...] = Field(max_length=16)
    forbidden_artifacts: tuple[EvidenceArtifactTypeV2, ...] = Field(max_length=16)
    selection_contract: CapabilitySelectionContractV2
    internal_company_knowledge_required: bool
    evidence_contract: ModelEvidenceContractV1
    abstention_policy: CapabilityAbstentionPolicyV2


class _FlatEvidenceCapabilityViewV1(_FrozenCapabilityViewModel):
    manifest_ref: ManifestRefV1
    capability_type: CapabilityTypeV1
    internal_company_knowledge_required: bool
    required_artifacts: tuple[EvidenceArtifactTypeV2, ...] = Field(max_length=16)
    allowed_artifacts: tuple[EvidenceArtifactTypeV2, ...] = Field(max_length=16)
    forbidden_artifacts: tuple[EvidenceArtifactTypeV2, ...] = Field(max_length=16)
    required_fact_types: tuple[EvidenceFactTypeV2, ...] = Field(max_length=16)
    any_of_fact_types: tuple[EvidenceFactTypeV2, ...] = Field(max_length=16)
    allowed_fact_types: tuple[EvidenceFactTypeV2, ...] = Field(max_length=16)
    forbidden_fact_types: tuple[EvidenceFactTypeV2, ...] = Field(max_length=16)
    required_artifact_types: tuple[EvidenceArtifactTypeV2, ...] = Field(max_length=16)
    allowed_artifact_types: tuple[EvidenceArtifactTypeV2, ...] = Field(max_length=16)
    required_scope_match: tuple[EvidenceScopeMatchFieldV2, ...] = Field(max_length=16)
    min_facts: int = Field(ge=0, le=32)
    allow_history: bool
    history_constraints: HistoryConstraintsV1
    fallback_policy: FallbackPolicyV2
    citation_required: bool
    provenance_required: bool
    citation_requirements: tuple[str, ...] = Field(max_length=16)
    stale_data_policy: StaleDataPolicyV1
    abstention_policy: CapabilityAbstentionPolicyV2


class ComposerCapabilityViewV1(_FlatEvidenceCapabilityViewV1):
    contract_version: Literal["composer-capability-view.v1"] = (
        "composer-capability-view.v1"
    )
    selection_summary: str = Field(min_length=1, max_length=512)
    composer_constraints: tuple[CapabilityComposerConstraintV2, ...] = Field(
        max_length=16
    )
    allowed_reply_kinds: tuple[
        Literal["answer", "clarification", "unable_to_answer"], ...
    ] = ("answer", "clarification", "unable_to_answer")
    max_reply_chars: Literal[4000] = 4000
    prohibited_output_types: tuple[ProhibitedComposerOutputV1, ...] = (
        "actions",
        "mentions",
        "media",
    )


class VerifierPostconditionV1(_FrozenCapabilityViewModel):
    primitive: VerifierPrimitiveV2
    required: Literal[True] = True


class VerifierCapabilityViewV1(_FlatEvidenceCapabilityViewV1):
    contract_version: Literal["verifier-capability-view.v1"] = (
        "verifier-capability-view.v1"
    )
    required_inputs: tuple[str, ...] = Field(max_length=16)
    output_schema: CapabilityViewOutputSchemaV1
    output_schema_hash: str = Field(pattern=r"^osh1:[0-9a-f]{64}$")
    verifier_primitives: tuple[VerifierPrimitiveV2, ...]
    postconditions: tuple[VerifierPostconditionV1, ...]
    prohibited_output_types: tuple[ProhibitedComposerOutputV1, ...] = (
        "actions",
        "mentions",
        "media",
    )


class PlannerCapabilityViewSetV1(_FrozenCapabilityViewModel):
    contract_version: Literal["planner-capability-view-set.v1"] = (
        "planner-capability-view-set.v1"
    )
    stage: Literal["planner_intent"] = "planner_intent"
    manifest_refs: tuple[ManifestRefV1, ...] = Field(max_length=32)
    visible_items_hash: str = Field(pattern=r"^cvh1:[0-9a-f]{64}$")
    items: tuple[PlannerCapabilityViewV1, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def _validate_refs(self) -> PlannerCapabilityViewSetV1:
        if self.manifest_refs != tuple(item.manifest_ref for item in self.items):
            raise CapabilityViewBindingError("planner_capability_view_refs_mismatch")
        _validate_view_hash(
            contract_version=self.contract_version,
            stage=self.stage,
            manifest_refs=self.manifest_refs,
            items=self.items,
            visible_items_hash=self.visible_items_hash,
        )
        return self


class ComposerCapabilityViewSetV1(_FrozenCapabilityViewModel):
    contract_version: Literal["composer-capability-view-set.v1"] = (
        "composer-capability-view-set.v1"
    )
    stage: Literal["knowledge_composer", "smalltalk_composer"]
    manifest_refs: tuple[ManifestRefV1, ...] = Field(min_length=1, max_length=4)
    visible_items_hash: str = Field(pattern=r"^cvh1:[0-9a-f]{64}$")
    items: tuple[ComposerCapabilityViewV1, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _validate_refs(self) -> ComposerCapabilityViewSetV1:
        if self.manifest_refs != tuple(item.manifest_ref for item in self.items):
            raise CapabilityViewBindingError("composer_capability_view_refs_mismatch")
        _validate_view_hash(
            contract_version=self.contract_version,
            stage=self.stage,
            manifest_refs=self.manifest_refs,
            items=self.items,
            visible_items_hash=self.visible_items_hash,
        )
        return self


class VerifierCapabilityViewSetV1(_FrozenCapabilityViewModel):
    contract_version: Literal["verifier-capability-view-set.v1"] = (
        "verifier-capability-view-set.v1"
    )
    stage: Literal["alignment_verifier"] = "alignment_verifier"
    manifest_refs: tuple[ManifestRefV1, ...] = Field(min_length=1, max_length=4)
    visible_items_hash: str = Field(pattern=r"^cvh1:[0-9a-f]{64}$")
    items: tuple[VerifierCapabilityViewV1, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def _validate_refs(self) -> VerifierCapabilityViewSetV1:
        if self.manifest_refs != tuple(item.manifest_ref for item in self.items):
            raise CapabilityViewBindingError("verifier_capability_view_refs_mismatch")
        _validate_view_hash(
            contract_version=self.contract_version,
            stage=self.stage,
            manifest_refs=self.manifest_refs,
            items=self.items,
            visible_items_hash=self.visible_items_hash,
        )
        return self


CapabilityViewSetV1 = (
    PlannerCapabilityViewSetV1
    | ComposerCapabilityViewSetV1
    | VerifierCapabilityViewSetV1
)
