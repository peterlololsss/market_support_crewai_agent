from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, assert_never

from pydantic import ConfigDict, Field, JsonValue

from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestV2,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.capabilities.projection_fields import (
    _canonical_output_schema,
    _flat_evidence_fields,
    _model_evidence_contract,
    _output_schema_hash,
    _planner_role_fields,
)
from market_support_crewai_agent.runtime.policy.capabilities.registry import (
    CAPABILITY_MANIFEST_REGISTRY,
    CapabilityRegistryV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    CapabilityViewSetV1,
    CapabilityViewStageV1,
    ComposerCapabilityViewSetV1,
    ComposerCapabilityViewV1,
    PlannerCapabilityViewSetV1,
    PlannerCapabilityViewV1,
    VerifierCapabilityViewSetV1,
    VerifierCapabilityViewV1,
    VerifierPostconditionV1,
    capability_view_hash,
)
from market_support_crewai_agent.schemas.base import StrictModel

CapabilityViewProjectionErrorCode = Literal[
    "duplicate_manifest_ref",
    "manifest_ref_unknown",
    "manifest_ref_version_mismatch",
    "planner_selected_manifest_refs_forbidden",
    "selected_manifest_refs_required",
    "selected_manifest_ref_not_eligible",
]

CapabilityViewByteBudgetStage = Literal[
    "planner_intent",
    "knowledge_composer",
    "smalltalk_composer",
    "alignment_verifier",
]

CAPABILITY_VIEW_MAX_SERIALIZED_BYTES: Final[
    dict[CapabilityViewByteBudgetStage, int]
] = {
    "planner_intent": 65_536,
    "knowledge_composer": 65_536,
    "smalltalk_composer": 65_536,
    "alignment_verifier": 65_536,
}


@dataclass(frozen=True, slots=True)
class CapabilityViewProjectionError(ValueError):
    code: CapabilityViewProjectionErrorCode
    manifest_ref: ManifestRefV1 | None = None

    def __str__(self) -> str:
        if self.manifest_ref is None:
            return self.code
        return (
            f"{self.code}:"
            f"{self.manifest_ref.manifest_id}@{self.manifest_ref.manifest_version}"
        )


@dataclass(frozen=True, slots=True)
class CapabilityViewByteBudgetError(ValueError):
    stage: CapabilityViewByteBudgetStage
    actual_bytes: int
    max_bytes: int

    def __str__(self) -> str:
        return (
            "capability_view_byte_budget_exceeded:"
            f"{self.stage}:{self.actual_bytes}>{self.max_bytes}"
        )


class CapabilityViewAuthorityV1(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["capability-view-authority.v1"] = (
        "capability-view-authority.v1"
    )
    eligible_manifest_refs: tuple[ManifestRefV1, ...] = Field(max_length=32)
    selected_manifest_refs: tuple[ManifestRefV1, ...] = Field(max_length=4)


def project_capability_view_set(
    *,
    stage: CapabilityViewStageV1,
    authority: CapabilityViewAuthorityV1,
    registry: CapabilityRegistryV2 = CAPABILITY_MANIFEST_REGISTRY,
) -> CapabilityViewSetV1:
    eligible = _resolve_refs(authority.eligible_manifest_refs, registry)
    selected = _resolve_refs(authority.selected_manifest_refs, registry)
    match stage:
        case "planner_intent":
            if selected:
                raise CapabilityViewProjectionError(
                    "planner_selected_manifest_refs_forbidden"
                )
            items = tuple(
                PlannerCapabilityViewV1(
                    **_planner_role_fields(ref, manifest),
                    selection_contract=manifest.selection_contract,
                    internal_company_knowledge_required=(
                        manifest.manifest_id == "answer_internal_company_knowledge"
                    ),
                    evidence_contract=_model_evidence_contract(manifest),
                    abstention_policy=manifest.abstention_policy,
                )
                for ref, manifest in eligible
            )
            view = PlannerCapabilityViewSetV1(
                manifest_refs=authority.eligible_manifest_refs,
                visible_items_hash=capability_view_hash(
                    contract_version="planner-capability-view-set.v1",
                    stage="planner_intent",
                    manifest_refs=authority.eligible_manifest_refs,
                    items=items,
                ),
                items=items,
            )
            validate_capability_view_budget(view)
            return view
        case "knowledge_composer" | "smalltalk_composer":
            selected_pairs = _authorized_selected_refs(eligible, selected)
            items = tuple(
                ComposerCapabilityViewV1(
                    **_flat_evidence_fields(ref, manifest),
                    selection_summary=manifest.selection_contract.summary,
                    composer_constraints=manifest.composer_constraints,
                )
                for ref, manifest in selected_pairs
            )
            view = ComposerCapabilityViewSetV1(
                stage=stage,
                manifest_refs=authority.selected_manifest_refs,
                visible_items_hash=capability_view_hash(
                    contract_version="composer-capability-view-set.v1",
                    stage=stage,
                    manifest_refs=authority.selected_manifest_refs,
                    items=items,
                ),
                items=items,
            )
            validate_capability_view_budget(view)
            return view
        case "alignment_verifier":
            selected_pairs = _authorized_selected_refs(eligible, selected)
            items = tuple(
                VerifierCapabilityViewV1(
                    **_flat_evidence_fields(ref, manifest),
                    required_inputs=(),
                    output_schema=_canonical_output_schema(),
                    output_schema_hash=_output_schema_hash(),
                    verifier_primitives=manifest.verifier_primitives,
                    postconditions=tuple(
                        VerifierPostconditionV1(primitive=primitive)
                        for primitive in manifest.verifier_primitives
                    ),
                )
                for ref, manifest in selected_pairs
            )
            view = VerifierCapabilityViewSetV1(
                manifest_refs=authority.selected_manifest_refs,
                visible_items_hash=capability_view_hash(
                    contract_version="verifier-capability-view-set.v1",
                    stage="alignment_verifier",
                    manifest_refs=authority.selected_manifest_refs,
                    items=items,
                ),
                items=items,
            )
            validate_capability_view_budget(view)
            return view
        case unreachable:
            assert_never(unreachable)


def _resolve_refs(
    refs: tuple[ManifestRefV1, ...],
    registry: CapabilityRegistryV2,
) -> tuple[tuple[ManifestRefV1, CapabilityManifestV2], ...]:
    resolved: list[tuple[ManifestRefV1, CapabilityManifestV2]] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.manifest_id in seen:
            raise CapabilityViewProjectionError("duplicate_manifest_ref", ref)
        manifest = registry.find(ref.manifest_id)
        if manifest is None:
            raise CapabilityViewProjectionError("manifest_ref_unknown", ref)
        if manifest.manifest_version != ref.manifest_version:
            raise CapabilityViewProjectionError("manifest_ref_version_mismatch", ref)
        seen.add(ref.manifest_id)
        resolved.append((ref, manifest))
    return tuple(resolved)


def _authorized_selected_refs(
    eligible: tuple[tuple[ManifestRefV1, CapabilityManifestV2], ...],
    selected: tuple[tuple[ManifestRefV1, CapabilityManifestV2], ...],
) -> tuple[tuple[ManifestRefV1, CapabilityManifestV2], ...]:
    if not selected:
        raise CapabilityViewProjectionError("selected_manifest_refs_required")
    eligible_refs = frozenset(
        (ref.manifest_id, ref.manifest_version) for ref, _manifest in eligible
    )
    for ref, _manifest in selected:
        if (ref.manifest_id, ref.manifest_version) not in eligible_refs:
            raise CapabilityViewProjectionError(
                "selected_manifest_ref_not_eligible", ref
            )
    return selected


def capability_view_serialized_bytes(
    view: CapabilityViewSetV1,
) -> bytes:
    return view.model_dump_json(exclude_none=False).encode("utf-8")


def validate_capability_view_budget(
    view: CapabilityViewSetV1,
    *,
    max_bytes: int | None = None,
) -> None:
    stage = view.stage
    limit = (
        CAPABILITY_VIEW_MAX_SERIALIZED_BYTES[stage] if max_bytes is None else max_bytes
    )
    actual = len(capability_view_serialized_bytes(view))
    if actual > limit:
        raise CapabilityViewByteBudgetError(
            stage=stage,
            actual_bytes=actual,
            max_bytes=limit,
        )
