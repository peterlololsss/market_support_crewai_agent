from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

# Absolute submodule import: a package-level import of the vocabulary would cycle
# through capabilities/__init__.py, which imports this module.
import market_support_crewai_agent.runtime.policy.capabilities.evidence_vocabulary as vocabulary
from market_support_crewai_agent.schemas.base import StrictModel

CapabilityName = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "sales_mention",
    "document_context",
]
ArtifactKind = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "multi_action",
    "knowledge_answer",
    "human_support",
    "refusal",
    "unclear",
    "smalltalk",
]
ResponseMode = Literal[
    "action",
    "clarification",
    "handoff",
    "refusal",
    "unable",
    "knowledge_answer",
    "smalltalk",
    "no_reply",
]
ResolvableBusinessStateField = Literal[
    "material_pack",
    "weekly_report",
    "monthly_report",
    "sales_mention",
]

CapabilityManifestIdV2 = Literal[
    "material_pack.send",
    "weekly_report.send",
    "monthly_report.send",
    "sales.handoff",
    "general.clarification",
    "general.abstention",
    "general.refusal",
    "general.smalltalk",
    "general.no_reply",
    "answer_internal_company_knowledge",
    "weekly_report.product_list",
    "monthly_report.product_list",
    "general.handoff",
]
ManifestVersionV2 = Literal["2026-07-18.1"]
VerifierPrimitiveV2 = Literal[
    "output_schema",
    "required_evidence_present",
    "evidence_artifact_type_allowed",
    "required_runtime_input_present",
    "forbidden_source_not_used",
    "abstention_correctness",
]
SelectionEffectV1 = Literal["select"]
OutboundIntentV1 = Literal["required", "forbidden"]
ReplyKindV1 = Literal[
    "answer",
    "clarification",
    "human_handoff",
    "unable_to_answer",
    "no_reply",
]


class ManifestRefV1(StrictModel):
    manifest_id: CapabilityManifestIdV2
    manifest_version: ManifestVersionV2


class CapabilitySelectionRuleV2(StrictModel):
    rule_id: str = Field(min_length=1, max_length=64)
    condition: str = Field(min_length=1, max_length=512)
    effect: SelectionEffectV1


class CapabilitySelectionContractV2(StrictModel):
    summary: str = Field(min_length=1, max_length=512)
    outbound_intent: OutboundIntentV1
    rules: tuple[CapabilitySelectionRuleV2, ...] = Field(min_length=1, max_length=8)


class CapabilityComposerConstraintV2(StrictModel):
    constraint_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=512)


class HistoryConstraintsV1(StrictModel):
    max_turns: int | None = Field(default=None, ge=0, le=20)
    max_age_seconds: int | None = Field(default=None, ge=0, le=2_592_000)
    allowed_roles: tuple[vocabulary.HistoryRoleV1, ...] = Field(
        default=(), max_length=2
    )

    @model_validator(mode="after")
    def validate_roles(self) -> HistoryConstraintsV1:
        if len(set(self.allowed_roles)) != len(self.allowed_roles):
            raise ValueError("duplicate_history_role")
        return self


class StaleDataPolicyV1(StrictModel):
    max_age_seconds: int | None = Field(default=None, ge=0, le=31_536_000)
    on_stale: vocabulary.StaleDataActionV1 = "allow"
    require_observed_at: bool = False


class CapabilityEvidenceContractV2(StrictModel):
    required_fact_types: tuple[vocabulary.EvidenceFactTypeV2, ...] = Field(
        default=(), max_length=16
    )
    any_of_fact_types: tuple[vocabulary.EvidenceFactTypeV2, ...] = Field(
        default=(), max_length=16
    )
    allowed_fact_types: tuple[vocabulary.EvidenceFactTypeV2, ...] = Field(
        default=(), max_length=16
    )
    forbidden_fact_types: tuple[vocabulary.EvidenceFactTypeV2, ...] = Field(
        default=(), max_length=16
    )
    allowed_source_types: tuple[vocabulary.EvidenceSourceTypeV2, ...] = Field(
        default=(), max_length=16
    )
    forbidden_source_types: tuple[vocabulary.EvidenceSourceTypeV2, ...] = Field(
        default=(), max_length=16
    )
    required_artifact_types: tuple[vocabulary.EvidenceArtifactTypeV2, ...] = Field(
        default=(), max_length=16
    )
    allowed_artifact_types: tuple[vocabulary.EvidenceArtifactTypeV2, ...] = Field(
        default=(), max_length=16
    )
    required_scope_match: tuple[vocabulary.EvidenceScopeMatchFieldV2, ...] = Field(
        default=(), max_length=16
    )
    min_facts: int = Field(default=0, ge=0, le=32)
    allow_history: bool = False
    history_constraints: HistoryConstraintsV1 = Field(
        default_factory=HistoryConstraintsV1
    )
    fallback_policy: vocabulary.FallbackPolicyV2 = "abstain"
    citation_required: bool = False
    provenance_required: bool = True
    citation_requirements: tuple[str, ...] = Field(default=(), max_length=16)
    stale_data_policy: StaleDataPolicyV1 = Field(default_factory=StaleDataPolicyV1)
    notes: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def validate_fact_sets(self) -> CapabilityEvidenceContractV2:
        fact_sets = (
            self.required_fact_types,
            self.any_of_fact_types,
            self.allowed_fact_types,
            self.forbidden_fact_types,
        )
        if any(len(set(values)) != len(values) for values in fact_sets):
            raise ValueError("duplicate_evidence_fact_type")
        if set(self.required_fact_types) & set(self.any_of_fact_types):
            raise ValueError("required and any-of evidence facts must not overlap")
        if not set(self.required_fact_types) <= set(self.allowed_fact_types):
            raise ValueError("required evidence facts must be allowed")
        if not set(self.any_of_fact_types) <= set(self.allowed_fact_types):
            raise ValueError("any-of evidence facts must be allowed")
        if set(self.allowed_fact_types) & set(self.forbidden_fact_types):
            raise ValueError("allowed and forbidden evidence facts must not overlap")
        if set(self.allowed_source_types) & set(self.forbidden_source_types):
            raise ValueError("allowed and forbidden evidence sources must not overlap")
        if len(set(self.allowed_source_types)) != len(self.allowed_source_types):
            raise ValueError("duplicate_allowed_evidence_source")
        if len(set(self.forbidden_source_types)) != len(self.forbidden_source_types):
            raise ValueError("duplicate_forbidden_evidence_source")
        if len(set(self.required_artifact_types)) != len(self.required_artifact_types):
            raise ValueError("duplicate_required_evidence_artifact")
        if len(set(self.allowed_artifact_types)) != len(self.allowed_artifact_types):
            raise ValueError("duplicate_allowed_evidence_artifact")
        if not set(self.required_artifact_types) <= set(self.allowed_artifact_types):
            raise ValueError("required evidence artifacts must be allowed")
        if len(set(self.required_scope_match)) != len(self.required_scope_match):
            raise ValueError("duplicate_required_scope_match")
        if len(set(self.citation_requirements)) != len(self.citation_requirements):
            raise ValueError("duplicate_citation_requirement")
        if any(not item or len(item) > 240 for item in self.citation_requirements):
            raise ValueError("invalid_citation_requirement")
        if (
            not self.allow_history
            and self.history_constraints != HistoryConstraintsV1()
        ):
            raise ValueError("history_constraints_require_history_admission")
        if self.allow_history and (
            not self.history_constraints.allowed_roles
            or (
                self.history_constraints.max_turns is None
                and self.history_constraints.max_age_seconds is None
            )
        ):
            raise ValueError("history_admission_requires_bounded_constraints")
        if self.citation_required and not self.citation_requirements:
            raise ValueError("citation_required_without_requirements")
        has_fact_contract = bool(
            self.required_fact_types
            or self.any_of_fact_types
            or self.allowed_fact_types
            or self.forbidden_fact_types
        )
        if has_fact_contract and not self.allowed_fact_types:
            raise ValueError("evidence_contract_requires_allowed_fact_types")
        if not has_fact_contract and (
            self.required_fact_types
            or self.any_of_fact_types
            or self.allowed_fact_types
            or self.forbidden_fact_types
        ):
            raise ValueError("capability_free_fact_contract_must_be_empty")
        return self


class CapabilityAbstentionPolicyV2(StrictModel):
    requires_abstention_when_evidence_missing: bool
    abstention_reply_kinds: tuple[ReplyKindV1, ...]
    guidance: str = Field(max_length=512)


class CapabilityManifestV2(StrictModel):
    manifest_id: CapabilityManifestIdV2
    manifest_version: ManifestVersionV2
    source_legacy_version: Literal["2026-06-16.1"] | None
    selection_contract: CapabilitySelectionContractV2
    composer_constraints: tuple[CapabilityComposerConstraintV2, ...]
    evidence_contract: CapabilityEvidenceContractV2
    abstention_policy: CapabilityAbstentionPolicyV2
    verifier_primitives: tuple[VerifierPrimitiveV2, ...] = Field(min_length=1)
    expected_content_hash: str = Field(pattern=r"^cmh1:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_internal_knowledge_evidence(self) -> CapabilityManifestV2:
        if self.manifest_id != "answer_internal_company_knowledge":
            return self
        if self.evidence_contract.allowed_fact_types != ("document_context",):
            raise ValueError("internal knowledge requires normalized document context")
        if set(self.evidence_contract.allowed_source_types) != {
            "document_mcp",
            "approved_static_knowledge",
        }:
            raise ValueError("internal knowledge requires unified sources")
        return self


VerifierPrimitive = VerifierPrimitiveV2
EvidenceContract = CapabilityEvidenceContractV2
AbstentionPolicy = CapabilityAbstentionPolicyV2
CapabilityManifest = CapabilityManifestV2
