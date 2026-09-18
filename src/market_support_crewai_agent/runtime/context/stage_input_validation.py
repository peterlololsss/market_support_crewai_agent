from __future__ import annotations

from typing import Final, Literal, TypeAlias

from market_support_crewai_agent.runtime.context.models import (
    EffectiveOutputCeilingsViewV1,
    PreflightFactViewV1,
    RecentExecutedActionSummaryViewV1,
    ResponseDirectiveViewV1,
    ScenePresentationViewV1,
    UnitGroundingViewV1,
    ValidatedPlanViewV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1


StageSceneV1: TypeAlias = Literal["group", "direct"]
_PREFLIGHT_ORDER: Final = (
    "material_pack",
    "weekly_report",
    "monthly_report",
    "sales_mention",
)


def validate_stage_scene(
    scene: StageSceneV1,
    presentation: ScenePresentationViewV1,
) -> None:
    if presentation.scene != scene:
        raise ContextViewInvariantError("stage_input_scene_mismatch")


def validate_recent_actions_order(
    actions: tuple[RecentExecutedActionSummaryViewV1, ...],
) -> None:
    known_ages = tuple(
        action.age_seconds for action in actions if action.age_seconds is not None
    )
    if known_ages != tuple(sorted(known_ages)):
        raise ContextViewInvariantError("stage_input_recent_actions_order_mismatch")


def validate_composer_binding(
    directive: ResponseDirectiveViewV1,
    ceilings: EffectiveOutputCeilingsViewV1,
    scene: StageSceneV1,
    expected_stage: Literal["knowledge_composer", "smalltalk_composer"],
) -> None:
    if not directive.requires_composer or directive.composer_stage != expected_stage:
        raise ContextViewInvariantError("stage_input_composer_directive_mismatch")
    if directive.mentions_requested and not ceilings.mentions_allowed:
        raise ContextViewInvariantError("stage_input_mentions_ceiling_mismatch")
    if scene == "direct" and (ceilings.mentions_allowed or ceilings.max_mentions):
        raise ContextViewInvariantError("stage_input_direct_mentions_forbidden")


def validate_plan_directive(
    plan: ValidatedPlanViewV1,
    directive: ResponseDirectiveViewV1,
) -> None:
    if directive.action_intent_count != len(plan.action_intents):
        raise ContextViewInvariantError("stage_input_directive_action_count_mismatch")


def validate_grounding_binding(
    plan: ValidatedPlanViewV1,
    selected_manifest_refs: tuple[ManifestRefV1, ...],
    groundings: tuple[UnitGroundingViewV1, ...],
) -> None:
    if selected_manifest_refs != plan.selected_manifest_refs:
        raise ContextViewInvariantError("stage_input_capability_plan_mismatch")
    if len(groundings) != len(plan.units):
        raise ContextViewInvariantError("stage_input_grounding_count_mismatch")
    for unit, grounding in zip(plan.units, groundings, strict=True):
        if grounding.unit_id != unit.unit_id:
            raise ContextViewInvariantError("stage_input_grounding_unit_mismatch")
        if grounding.manifest_ref != unit.manifest_ref:
            raise ContextViewInvariantError("stage_input_grounding_manifest_mismatch")
        if grounding.scope != unit.scope:
            raise ContextViewInvariantError("stage_input_grounding_scope_mismatch")
        if grounding.evidence_query != unit.evidence_query:
            raise ContextViewInvariantError("stage_input_grounding_query_mismatch")
        if grounding.action_intents != unit.action_intents:
            raise ContextViewInvariantError("stage_input_grounding_intents_mismatch")
        if grounding.answerability != unit.answerability:
            raise ContextViewInvariantError(
                "stage_input_grounding_answerability_mismatch"
            )


def validate_preflight_order(
    preflight: tuple[PreflightFactViewV1, ...],
) -> None:
    ranks = tuple(_PREFLIGHT_ORDER.index(item.resolve_type) for item in preflight)
    if ranks != tuple(sorted(ranks)):
        raise ContextViewInvariantError("stage_input_preflight_order_mismatch")
