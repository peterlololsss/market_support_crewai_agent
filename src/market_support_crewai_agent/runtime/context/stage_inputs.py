from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.runtime.context.models import (
    BusinessScopeViewV1,
    CandidateReplyViewV1,
    ComposerRetryOverlayV1,
    CurrentMessageViewV1,
    EffectiveOutputCeilingsViewV1,
    EffectivePolicyViewV1,
    GuardrailDecisionViewV1,
    HistoryTurnViewV1,
    IntentGateViewV1,
    MaterialPackOptionSummaryViewV1,
    PendingClarificationViewV1,
    PlannerRetryOverlayV1,
    PreflightFactViewV1,
    RecallPlannerViewV1,
    RecentExecutedActionSummaryViewV1,
    ResponseDirectiveViewV1,
    RuntimeClockViewV1,
    ScenePresentationViewV1,
    UnitGroundingViewV1,
    ValidatedPlanViewV1,
)
from market_support_crewai_agent.runtime.context.stage_input_validation import (
    StageSceneV1,
    validate_composer_binding,
    validate_grounding_binding,
    validate_plan_directive,
    validate_preflight_order,
    validate_recent_actions_order,
    validate_stage_scene,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    ComposerCapabilityViewSetV1,
    PlannerCapabilityViewSetV1,
    VerifierCapabilityViewSetV1,
)
from market_support_crewai_agent.schemas.base import StrictModel


class _FrozenStageInput(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlannerPromptInputV1(_FrozenStageInput):
    contract_version: Literal["planner-prompt-input.v1"] = "planner-prompt-input.v1"
    scene: StageSceneV1
    message: CurrentMessageViewV1
    history: tuple[HistoryTurnViewV1, ...] = Field(default=(), max_length=12)
    runtime_clock: RuntimeClockViewV1
    pending_clarification: PendingClarificationViewV1 | None = None
    recent_executed_actions: tuple[RecentExecutedActionSummaryViewV1, ...] = Field(
        default=(), max_length=20
    )
    material_pack_options: MaterialPackOptionSummaryViewV1
    presentation: ScenePresentationViewV1
    business_scope: BusinessScopeViewV1
    effective_policy: EffectivePolicyViewV1
    intent_gate: IntentGateViewV1
    eligible_capabilities: PlannerCapabilityViewSetV1
    recall: RecallPlannerViewV1
    retry_overlay: PlannerRetryOverlayV1 | None = None

    @model_validator(mode="after")
    def validate_role_bindings(self) -> PlannerPromptInputV1:
        validate_stage_scene(self.scene, self.presentation)
        if self.effective_policy.scene != self.scene:
            raise ContextViewInvariantError("stage_input_policy_scene_mismatch")
        validate_recent_actions_order(self.recent_executed_actions)
        if (
            self.eligible_capabilities.manifest_refs
            != self.effective_policy.eligible_capabilities
        ):
            raise ContextViewInvariantError("stage_input_capability_policy_mismatch")
        if self.recall.mode != self.effective_policy.recall_mode:
            raise ContextViewInvariantError("stage_input_recall_mode_mismatch")
        if (
            self.material_pack_options.total_count
            != self.intent_gate.material_pack_option_count
        ):
            raise ContextViewInvariantError(
                "stage_input_material_option_count_mismatch"
            )
        return self


class KnowledgeComposerPromptInputV1(_FrozenStageInput):
    contract_version: Literal["knowledge-composer-input.v1"] = (
        "knowledge-composer-input.v1"
    )
    scene: StageSceneV1
    message: CurrentMessageViewV1
    history: tuple[HistoryTurnViewV1, ...] = Field(default=(), max_length=12)
    runtime_clock: RuntimeClockViewV1
    presentation: ScenePresentationViewV1
    validated_plan: ValidatedPlanViewV1
    directive: ResponseDirectiveViewV1
    output_ceilings: EffectiveOutputCeilingsViewV1
    selected_capabilities: ComposerCapabilityViewSetV1
    preflight: tuple[PreflightFactViewV1, ...] = Field(default=(), max_length=8)
    unit_groundings: tuple[UnitGroundingViewV1, ...] = Field(min_length=1, max_length=4)
    retry_overlay: ComposerRetryOverlayV1 | None = None

    @model_validator(mode="after")
    def validate_role_bindings(self) -> KnowledgeComposerPromptInputV1:
        validate_stage_scene(self.scene, self.presentation)
        validate_composer_binding(
            self.directive,
            self.output_ceilings,
            self.scene,
            "knowledge_composer",
        )
        if self.selected_capabilities.stage != "knowledge_composer":
            raise ContextViewInvariantError("stage_input_capability_stage_mismatch")
        validate_plan_directive(self.validated_plan, self.directive)
        validate_grounding_binding(
            self.validated_plan,
            self.selected_capabilities.manifest_refs,
            self.unit_groundings,
        )
        validate_preflight_order(self.preflight)
        return self


class SmalltalkComposerPromptInputV1(_FrozenStageInput):
    contract_version: Literal["smalltalk-composer-input.v1"] = (
        "smalltalk-composer-input.v1"
    )
    scene: StageSceneV1
    message: CurrentMessageViewV1
    history: tuple[HistoryTurnViewV1, ...] = Field(default=(), max_length=12)
    presentation: ScenePresentationViewV1
    directive: ResponseDirectiveViewV1
    output_ceilings: EffectiveOutputCeilingsViewV1
    selected_capabilities: ComposerCapabilityViewSetV1
    guardrails: tuple[GuardrailDecisionViewV1, ...] = Field(default=(), max_length=32)
    retry_overlay: ComposerRetryOverlayV1 | None = None

    @model_validator(mode="after")
    def validate_role_bindings(self) -> SmalltalkComposerPromptInputV1:
        validate_stage_scene(self.scene, self.presentation)
        validate_composer_binding(
            self.directive,
            self.output_ceilings,
            self.scene,
            "smalltalk_composer",
        )
        if self.selected_capabilities.stage != "smalltalk_composer":
            raise ContextViewInvariantError("stage_input_capability_stage_mismatch")
        return self


class AlignmentVerifierPromptInputV1(_FrozenStageInput):
    contract_version: Literal["alignment-verifier-input.v1"] = (
        "alignment-verifier-input.v1"
    )
    scene: StageSceneV1
    message: CurrentMessageViewV1
    history: tuple[HistoryTurnViewV1, ...] = Field(default=(), max_length=12)
    presentation: ScenePresentationViewV1
    validated_plan: ValidatedPlanViewV1
    directive: ResponseDirectiveViewV1
    selected_capabilities: VerifierCapabilityViewSetV1
    unit_groundings: tuple[UnitGroundingViewV1, ...] = Field(min_length=1, max_length=4)
    candidate: CandidateReplyViewV1
    attempt: int = Field(ge=0, le=2)

    @model_validator(mode="after")
    def validate_role_bindings(self) -> AlignmentVerifierPromptInputV1:
        validate_stage_scene(self.scene, self.presentation)
        validate_plan_directive(self.validated_plan, self.directive)
        validate_grounding_binding(
            self.validated_plan,
            self.selected_capabilities.manifest_refs,
            self.unit_groundings,
        )
        if self.scene == "direct" and self.candidate.mentions:
            raise ContextViewInvariantError(
                "stage_input_direct_candidate_mentions_forbidden"
            )
        return self


SanitizedAlignmentVerifierInputV1: TypeAlias = AlignmentVerifierPromptInputV1


@dataclass(frozen=True, slots=True)
class PlannerPromptInputSourceV1:
    scene: StageSceneV1
    message: CurrentMessageViewV1
    history: tuple[HistoryTurnViewV1, ...]
    runtime_clock: RuntimeClockViewV1
    pending_clarification: PendingClarificationViewV1 | None
    recent_executed_actions: tuple[RecentExecutedActionSummaryViewV1, ...]
    material_pack_options: MaterialPackOptionSummaryViewV1
    presentation: ScenePresentationViewV1
    business_scope: BusinessScopeViewV1
    effective_policy: EffectivePolicyViewV1
    intent_gate: IntentGateViewV1
    eligible_capabilities: PlannerCapabilityViewSetV1
    recall: RecallPlannerViewV1
    retry_overlay: PlannerRetryOverlayV1 | None


@dataclass(frozen=True, slots=True)
class KnowledgeComposerPromptInputSourceV1:
    scene: StageSceneV1
    message: CurrentMessageViewV1
    history: tuple[HistoryTurnViewV1, ...]
    runtime_clock: RuntimeClockViewV1
    presentation: ScenePresentationViewV1
    validated_plan: ValidatedPlanViewV1
    directive: ResponseDirectiveViewV1
    output_ceilings: EffectiveOutputCeilingsViewV1
    selected_capabilities: ComposerCapabilityViewSetV1
    preflight: tuple[PreflightFactViewV1, ...]
    unit_groundings: tuple[UnitGroundingViewV1, ...]
    retry_overlay: ComposerRetryOverlayV1 | None


@dataclass(frozen=True, slots=True)
class SmalltalkComposerPromptInputSourceV1:
    scene: StageSceneV1
    message: CurrentMessageViewV1
    history: tuple[HistoryTurnViewV1, ...]
    presentation: ScenePresentationViewV1
    directive: ResponseDirectiveViewV1
    output_ceilings: EffectiveOutputCeilingsViewV1
    selected_capabilities: ComposerCapabilityViewSetV1
    guardrails: tuple[GuardrailDecisionViewV1, ...]
    retry_overlay: ComposerRetryOverlayV1 | None


@dataclass(frozen=True, slots=True)
class AlignmentVerifierPromptInputSourceV1:
    scene: StageSceneV1
    message: CurrentMessageViewV1
    history: tuple[HistoryTurnViewV1, ...]
    presentation: ScenePresentationViewV1
    validated_plan: ValidatedPlanViewV1
    directive: ResponseDirectiveViewV1
    selected_capabilities: VerifierCapabilityViewSetV1
    unit_groundings: tuple[UnitGroundingViewV1, ...]
    candidate: CandidateReplyViewV1
    attempt: int


def build_planner_prompt_input_v1(
    source: PlannerPromptInputSourceV1,
) -> PlannerPromptInputV1:
    return PlannerPromptInputV1.model_validate(source, from_attributes=True)


def build_knowledge_composer_prompt_input_v1(
    source: KnowledgeComposerPromptInputSourceV1,
) -> KnowledgeComposerPromptInputV1:
    return KnowledgeComposerPromptInputV1.model_validate(source, from_attributes=True)


def build_smalltalk_composer_prompt_input_v1(
    source: SmalltalkComposerPromptInputSourceV1,
) -> SmalltalkComposerPromptInputV1:
    return SmalltalkComposerPromptInputV1.model_validate(source, from_attributes=True)


def build_alignment_verifier_prompt_input_v1(
    source: AlignmentVerifierPromptInputSourceV1,
) -> AlignmentVerifierPromptInputV1:
    return AlignmentVerifierPromptInputV1.model_validate(source, from_attributes=True)
