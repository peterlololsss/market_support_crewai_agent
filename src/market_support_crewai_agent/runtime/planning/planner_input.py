from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, assert_never
from zoneinfo import ZoneInfo

from market_support_crewai_agent.runtime.context.common_view_models import (
    RelativeYearsViewV1,
)
from market_support_crewai_agent.runtime.context.models import (
    CurrentMessageViewV1,
    HistoryTurnViewV1,
    IntentGateViewV1,
    MaterialPackOptionSummaryViewV1,
    RecentExecutedActionSummaryViewV1,
    RuntimeClockViewV1,
    ScenePresentationViewV1,
)
from market_support_crewai_agent.runtime.context.policy_view_models import (
    DistributionBusinessScopeViewV1,
    EffectivePolicyViewV1,
    UnscopedBusinessScopeViewV1,
)
from market_support_crewai_agent.runtime.context.recall_view_models import (
    RecallPlannerShortcutSummaryViewV1,
    RecallPlannerViewV1,
)
from market_support_crewai_agent.runtime.context.retry_view_models import (
    PlannerRetryOverlayV1,
)
from market_support_crewai_agent.runtime.context.stage_inputs import (
    PlannerPromptInputSourceV1,
    PlannerPromptInputV1,
    build_planner_prompt_input_v1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.capabilities.prompt_projection import (
    CapabilityViewAuthorityV1,
    project_capability_view_set,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    ComposerCapabilityViewSetV1,
    PlannerCapabilityViewSetV1,
    VerifierCapabilityViewSetV1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.schemas.conversation import (
    DirectPresentationV1,
    DistributionScopeV1,
    GroupPresentationV1,
    UnscopedScopeV1,
)

_SHANGHAI: Final = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class PlannerRuntimeInputSourceV1:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    scope_authority: BusinessScopeAuthorityV1
    intent_gate: IntentGateResult
    history: Sequence[ConversationMessage]
    action_history: Sequence[RecentExecutedActionSummaryViewV1]
    recall_state: RecallTurnStateV1
    now: datetime
    retry_overlay: PlannerRetryOverlayV1 | None = None


def build_runtime_planner_prompt_input_v1(
    source: PlannerRuntimeInputSourceV1,
) -> PlannerPromptInputV1:
    if source.request.identity.scene != source.policy.scene:
        raise ContextViewInvariantError("planner_request_policy_scene_mismatch")
    if source.request.business_scope != source.scope_authority.scope:
        raise ContextViewInvariantError("planner_business_scope_authority_mismatch")
    if source.policy.business_scope_hash != source.scope_authority.business_scope_hash:
        raise ContextViewInvariantError("planner_policy_scope_hash_mismatch")
    eligible_capabilities = project_capability_view_set(
        stage="planner_intent",
        authority=CapabilityViewAuthorityV1(
            eligible_manifest_refs=source.policy.eligible_capabilities,
            selected_manifest_refs=(),
        ),
    )
    match eligible_capabilities:
        case PlannerCapabilityViewSetV1():
            pass
        case ComposerCapabilityViewSetV1() | VerifierCapabilityViewSetV1():
            raise ContextViewInvariantError(
                "planner_capability_projection_stage_mismatch"
            )
        case unreachable:
            assert_never(unreachable)
    return build_planner_prompt_input_v1(
        PlannerPromptInputSourceV1(
            scene=source.request.identity.scene,
            message=CurrentMessageViewV1(text=source.request.message),
            history=_history_views(source.history, source.now),
            runtime_clock=_runtime_clock_view(source.now),
            pending_clarification=None,
            recent_executed_actions=tuple(source.action_history[:20]),
            material_pack_options=MaterialPackOptionSummaryViewV1(
                total_count=len(source.policy.material_pack_options),
                exact_requested_option=None,
                exact_match=None,
                bounded_page_available=bool(source.policy.material_pack_options),
            ),
            presentation=_presentation_view(source.request),
            business_scope=_business_scope_view(
                source.request,
                source.scope_authority,
            ),
            effective_policy=_effective_policy_view(source.policy),
            intent_gate=IntentGateViewV1(
                artifact_hint=source.intent_gate.artifact_hint,
                outbound_action_hint=source.intent_gate.outbound_action_hint,
                material_pack_option_count=(
                    source.intent_gate.material_pack_option_count
                ),
                compliance_hint=source.intent_gate.compliance_hint,
                confidence=source.intent_gate.confidence,
            ),
            eligible_capabilities=eligible_capabilities,
            recall=_recall_view(source.recall_state),
            retry_overlay=source.retry_overlay,
        )
    )


def _runtime_clock_view(now: datetime) -> RuntimeClockViewV1:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ContextViewInvariantError("planner_runtime_clock_timezone_required")
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


def _history_views(
    history: Sequence[ConversationMessage],
    now: datetime,
) -> tuple[HistoryTurnViewV1, ...]:
    local_now = now.astimezone(_SHANGHAI)
    views: list[HistoryTurnViewV1] = []
    for message in history[-12:]:
        age_seconds = None
        if message.created_at.tzinfo is not None:
            age_seconds = min(
                max(0, int((local_now - message.created_at).total_seconds())),
                2_592_000,
            )
        views.append(
            HistoryTurnViewV1(
                role=message.role,
                text=message.content[:1_200],
                age_seconds=age_seconds,
            )
        )
    return tuple(views)


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
        case unreachable:
            assert_never(unreachable)


def _business_scope_view(
    request: KernelReplyRequestV1,
    authority: BusinessScopeAuthorityV1,
) -> DistributionBusinessScopeViewV1 | UnscopedBusinessScopeViewV1:
    match request.business_scope:
        case DistributionScopeV1():
            return DistributionBusinessScopeViewV1(
                business_scope_ref=authority.business_scope_ref,
                channel_type=request.business_scope.channel_type,
                dist_channel_name=request.business_scope.dist_channel_name,
                artifact_types=tuple(
                    sorted(
                        {
                            artifact.type
                            for artifact in request.business_scope.available_artifacts
                        }
                    )
                ),
            )
        case UnscopedScopeV1():
            return UnscopedBusinessScopeViewV1()
        case unreachable:
            assert_never(unreachable)


def _effective_policy_view(policy: PolicyManifestV2) -> EffectivePolicyViewV1:
    return EffectivePolicyViewV1(
        policy_id=policy.policy_id,
        scene=policy.scene,
        eligible_capabilities=policy.eligible_capabilities,
        read_capabilities=policy.allowed_read_capabilities,
        internal_company_knowledge_enabled=policy.internal_company_knowledge_enabled,
        outbound_actions=policy.allowed_outbound_actions,
        mention_types=policy.allowed_mention_types,
        adapter_resolves=policy.allowed_adapter_resolves,
        allowed_reply_modes=policy.allowed_reply_modes,
        recall_mode=policy.recall_mode,
        evidence_call_limit=policy.evidence_call_limit,
        actions_allowed=policy.actions_allowed,
        mentions_allowed=policy.mentions_allowed,
    )


def _recall_view(state: RecallTurnStateV1) -> RecallPlannerViewV1:
    summary = state.outcome.shortcut_summary
    projected_summary = (
        None
        if summary is None
        else RecallPlannerShortcutSummaryViewV1(
            candidate_id=summary.candidate_id,
            canonical_id=summary.canonical_id,
            selected_manifest_ref=summary.selected_manifest_ref,
            confidence=summary.confidence,
            reply_text_hash=summary.reply_text_hash,
            evidence_text_hash=summary.evidence_text_hash,
        )
    )
    return RecallPlannerViewV1(
        mode=state.outcome.mode,
        decision=state.outcome.decision,
        candidates=state.outcome.candidates,
        branch_outcomes=state.outcome.branch_outcomes,
        shortcut_summary=projected_summary,
        trace_hash=state.outcome.trace_hash,
    )
