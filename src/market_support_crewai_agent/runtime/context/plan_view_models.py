from __future__ import annotations

from typing import Annotated, Literal, TypeAlias, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    ArtifactKind,
    CapabilityName,
    ManifestRefV1,
    ResponseMode,
)
from market_support_crewai_agent.runtime.policy.compliance import (
    ComplianceReasonCode,
)
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import OutboundActionType


class _FrozenPlanView(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanTimeRangeViewV1(_FrozenPlanView):
    period: str | None = Field(default=None, max_length=80)
    start: str | None = Field(default=None, max_length=80)
    end: str | None = Field(default=None, max_length=80)
    label: str | None = Field(default=None, max_length=80)

    @field_validator("period", "start", "end", "label")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 for character in value):
            raise ContextViewInvariantError("plan_time_range_control_character")
        return value


class ActionIntentViewV1(_FrozenPlanView):
    action_type: OutboundActionType
    capability_id: CapabilityName
    material_pack_option: str | None = Field(default=None, max_length=80)


class DistributionPlanScopeViewV1(_FrozenPlanView):
    kind: Literal["distribution"] = "distribution"
    channel_type: Literal["bank", "non_bank", "unknown"]
    dist_channel_name: str = Field(min_length=1, max_length=120)
    material_pack_option: str | None = Field(default=None, max_length=80)
    time_range: PlanTimeRangeViewV1 | None = None
    product_count: int = Field(ge=0, le=10_000)

    @field_validator("dist_channel_name", "material_pack_option")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 for character in value):
            raise ContextViewInvariantError("plan_scope_control_character")
        return value


class UnscopedPlanScopeViewV1(_FrozenPlanView):
    kind: Literal["unscoped"] = "unscoped"


PlanScopeViewV1: TypeAlias = Annotated[
    Union[DistributionPlanScopeViewV1, UnscopedPlanScopeViewV1],
    Field(discriminator="kind"),
]


class ValidatedPlanUnitViewV1(_FrozenPlanView):
    contract_version: Literal["validated-plan-unit-view.v1"] = (
        "validated-plan-unit-view.v1"
    )
    unit_id: str = Field(min_length=1, max_length=120)
    manifest_ref: ManifestRefV1
    answerability: Literal[
        "answer",
        "send",
        "clarify",
        "abstain",
        "refuse",
        "handoff",
        "smalltalk",
        "no_reply",
    ]
    scope: PlanScopeViewV1
    evidence_query: str | None = Field(default=None, max_length=200)
    action_intents: tuple[ActionIntentViewV1, ...] = Field(default=(), max_length=3)
    answer_capability_ids: tuple[CapabilityName, ...] = Field(default=(), max_length=4)
    ambiguity_slots: tuple[str, ...] = Field(default=(), max_length=8)
    risk_flags: tuple[Literal["weekly_report_rationale_required"], ...] = Field(
        default=(), max_length=1
    )

    @model_validator(mode="after")
    def validate_canonical_fields(self) -> ValidatedPlanUnitViewV1:
        if tuple(sorted(set(self.answer_capability_ids))) != self.answer_capability_ids:
            raise ContextViewInvariantError(
                "validated_plan_answer_capabilities_not_canonical"
            )
        if len(set(self.ambiguity_slots)) != len(self.ambiguity_slots):
            raise ContextViewInvariantError("validated_plan_ambiguity_slots_not_unique")
        if any(
            not slot
            or len(slot) > 120
            or any(ord(character) < 32 for character in slot)
            for slot in self.ambiguity_slots
        ):
            raise ContextViewInvariantError("validated_plan_ambiguity_slot_invalid")
        if self.risk_flags not in (
            (),
            ("weekly_report_rationale_required",),
        ):
            raise ContextViewInvariantError("validated_plan_risk_flags_not_canonical")
        return self


class ValidatedPlanViewV1(_FrozenPlanView):
    contract_version: Literal["validated-plan-view.v1"] = "validated-plan-view.v1"
    execution_plan_id: str = Field(pattern=r"^epl1:[0-9a-f]{64}$")
    user_need: str = Field(min_length=1, max_length=500)
    response_mode: ResponseMode
    artifact_kind: ArtifactKind
    units: tuple[ValidatedPlanUnitViewV1, ...] = Field(min_length=1, max_length=4)
    selected_manifest_refs: tuple[ManifestRefV1, ...] = Field(
        min_length=1, max_length=4
    )
    action_intents: tuple[ActionIntentViewV1, ...] = Field(default=(), max_length=3)
    is_compliant: bool | None = None
    compliance_reason_code: ComplianceReasonCode
    compliance_reason: str = Field(default="", max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_aggregates(self) -> ValidatedPlanViewV1:
        refs: list[ManifestRefV1] = []
        seen: set[str] = set()
        for unit in self.units:
            if unit.manifest_ref.manifest_id not in seen:
                seen.add(unit.manifest_ref.manifest_id)
                refs.append(unit.manifest_ref)
        if self.selected_manifest_refs != tuple(refs):
            raise ContextViewInvariantError("validated_plan_selected_refs_mismatch")
        expected_actions = tuple(
            action for unit in self.units for action in unit.action_intents
        )
        if self.action_intents != expected_actions:
            raise ContextViewInvariantError("validated_plan_action_intents_mismatch")
        return self
