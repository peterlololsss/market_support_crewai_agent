from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from market_support_crewai_agent.runtime.context.business_view_projection import (
    project_guardrail_decision_view_v1,
)
from market_support_crewai_agent.runtime.context.common_view_models import (
    RelativeYearsViewV1,
)
from market_support_crewai_agent.runtime.context.grounding_projection_context import (
    GroundingProjectionContextV1,
)
from market_support_crewai_agent.runtime.context.models import (
    ComposerRetryOverlayV1,
    CurrentMessageViewV1,
    EffectiveOutputCeilingsViewV1,
    HistoryTurnViewV1,
    ResponseDirectiveViewV1,
    RuntimeClockViewV1,
    ScenePresentationViewV1,
)
from market_support_crewai_agent.runtime.context.projection import (
    project_unit_grounding_views_v1,
    project_validated_plan_view_v1,
)
from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputSourceV1,
    KnowledgeComposerPromptInputV1,
    SmalltalkComposerPromptInputSourceV1,
    SmalltalkComposerPromptInputV1,
    build_knowledge_composer_prompt_input_v1,
    build_smalltalk_composer_prompt_input_v1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    ExecutionUnitGroundingV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    RegisteredMediaBindingV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.capabilities.prompt_projection import (
    CapabilityViewAuthorityV1,
    project_capability_view_set,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    ComposerCapabilityViewSetV1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.rendering.composer_preflight_projection import (
    project_preflight_fact_views_v1,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.conversation import (
    DirectPresentationV1,
    GroupPresentationV1,
)

ComposerPromptInputV1 = KnowledgeComposerPromptInputV1 | SmalltalkComposerPromptInputV1
_SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class ComposerInvocationV1:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    scope_authority: BusinessScopeAuthorityV1
    plan: ExecutionPlanV2
    preflight: AdapterPreflightSnapshot
    groundings: tuple[ExecutionUnitGroundingV1, ...]
    media_bindings: tuple[RegisteredMediaBindingV1, ...]
    locator_safety: LocatorSafetyClassifierV1
    now: datetime
    history: tuple[HistoryTurnViewV1, ...] = ()
    retry_overlay: ComposerRetryOverlayV1 | None = None


def build_composer_prompt_input_v1(
    source: ComposerInvocationV1,
) -> ComposerPromptInputV1:
    _validate_invocation_authority(source)
    match source.plan.response_mode:  # noqa: MATCH_OK - reject valid non-composer modes.
        case "knowledge_answer":
            return _knowledge_input(source)
        case "smalltalk":
            return _smalltalk_input(source)
        case _:
            raise ContextViewInvariantError(
                "composer_requires_knowledge_or_smalltalk_response_mode"
            )


def _knowledge_input(source: ComposerInvocationV1) -> KnowledgeComposerPromptInputV1:
    return build_knowledge_composer_prompt_input_v1(
        KnowledgeComposerPromptInputSourceV1(
            scene=source.request.identity.scene,
            message=CurrentMessageViewV1(text=source.request.message),
            history=source.history,
            runtime_clock=_runtime_clock_view(source.now),
            presentation=_presentation_view(source.request),
            validated_plan=project_validated_plan_view_v1(
                source.plan,
                source.scope_authority,
            ),
            directive=_directive_view(source.plan, "knowledge_composer"),
            output_ceilings=_output_ceilings(),
            selected_capabilities=_selected_capability_views(
                source,
                "knowledge_composer",
            ),
            preflight=project_preflight_fact_views_v1(source.preflight),
            unit_groundings=project_unit_grounding_views_v1(
                source.plan,
                source.groundings,
                GroundingProjectionContextV1(
                    business_scope_authority=source.scope_authority,
                    locator_safety=source.locator_safety,
                    evaluation_epoch_seconds=int(source.now.timestamp()),
                ),
            ),
            retry_overlay=source.retry_overlay,
        )
    )


def _smalltalk_input(source: ComposerInvocationV1) -> SmalltalkComposerPromptInputV1:
    return build_smalltalk_composer_prompt_input_v1(
        SmalltalkComposerPromptInputSourceV1(
            scene=source.request.identity.scene,
            message=CurrentMessageViewV1(text=source.request.message),
            history=source.history,
            presentation=_presentation_view(source.request),
            directive=_directive_view(source.plan, "smalltalk_composer"),
            output_ceilings=_output_ceilings(),
            selected_capabilities=_selected_capability_views(
                source,
                "smalltalk_composer",
            ),
            guardrails=tuple(
                project_guardrail_decision_view_v1(decision)
                for decision in source.plan.guardrail_decisions
            ),
            retry_overlay=source.retry_overlay,
        )
    )


def _validate_invocation_authority(source: ComposerInvocationV1) -> None:
    if source.request.identity.scene != source.policy.scene:
        raise ContextViewInvariantError("composer_request_policy_scene_mismatch")
    if source.request.business_scope != source.scope_authority.scope:
        raise ContextViewInvariantError("composer_business_scope_authority_mismatch")
    if source.policy.business_scope_hash != source.scope_authority.business_scope_hash:
        raise ContextViewInvariantError("composer_policy_scope_hash_mismatch")
    if source.now.tzinfo is None or source.now.utcoffset() is None:
        raise ContextViewInvariantError("composer_runtime_clock_timezone_required")


def _selected_capability_views(
    source: ComposerInvocationV1,
    stage: Literal["knowledge_composer", "smalltalk_composer"],
) -> ComposerCapabilityViewSetV1:
    view = project_capability_view_set(
        stage=stage,
        authority=CapabilityViewAuthorityV1(
            eligible_manifest_refs=source.policy.eligible_capabilities,
            selected_manifest_refs=source.plan.selected_manifest_refs,
        ),
    )
    match view:  # noqa: MATCH_OK - reject non-composer role projections.
        case ComposerCapabilityViewSetV1():
            return view
        case _:
            raise ContextViewInvariantError("composer_capability_view_role_mismatch")


def _presentation_view(request: KernelReplyRequestV1) -> ScenePresentationViewV1:
    match request.presentation:
        case GroupPresentationV1():
            return ScenePresentationViewV1(
                scene="group",
                audience="group",
                conversation_name=request.presentation.conversation_name,
                principal_name=request.presentation.principal_name,
                redacted=False,
            )
        case DirectPresentationV1():
            return ScenePresentationViewV1(
                scene="direct",
                audience="individual",
                principal_name=request.presentation.principal_name,
                redacted=request.presentation.principal_name is None,
            )


def _runtime_clock_view(now: datetime) -> RuntimeClockViewV1:
    local = now.astimezone(_SHANGHAI)
    year = local.year
    return RuntimeClockViewV1(
        current_date=local.date().isoformat(),
        current_datetime=local.isoformat(timespec="seconds"),
        weekday=local.isoweekday(),
        relative_years=RelativeYearsViewV1(
            current=year,
            last=year - 1,
            two_years_ago=year - 2,
        ),
    )


def _directive_view(
    plan: ExecutionPlanV2,
    stage: Literal["knowledge_composer", "smalltalk_composer"],
) -> ResponseDirectiveViewV1:
    return ResponseDirectiveViewV1(
        mode=plan.response_mode,
        reply_kind="answer",
        reason_code=(
            "grounded_answer" if stage == "knowledge_composer" else "smalltalk_reply"
        ),
        requires_composer=True,
        composer_stage=stage,
        mentions_requested=False,
        action_intent_count=len(plan.action_intents),
    )


def _output_ceilings() -> EffectiveOutputCeilingsViewV1:
    return EffectiveOutputCeilingsViewV1(
        allowed_reply_kinds=("answer", "clarification", "unable_to_answer"),
        mentions_allowed=False,
        max_mentions=0,
        max_reply_chars=4_000,
    )
