from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import assert_never

from market_support_crewai_agent.runtime.context.grounding_projection_context import (
    GroundingProjectionContextV1,
)
from market_support_crewai_agent.runtime.context.models import (
    CandidateActionViewV1,
    CandidateMentionViewV1,
    CandidateReplyViewV1,
    CurrentMessageViewV1,
    HistoryTurnViewV1,
    ResponseDirectiveViewV1,
    ScenePresentationViewV1,
)
from market_support_crewai_agent.runtime.context.projection import (
    project_unit_grounding_views_v1,
    project_validated_plan_view_v1,
)
from market_support_crewai_agent.runtime.context.stage_inputs import (
    AlignmentVerifierPromptInputSourceV1,
    SanitizedAlignmentVerifierInputV1,
    build_alignment_verifier_prompt_input_v1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.capabilities.prompt_projection import (
    CapabilityViewAuthorityV1,
    project_capability_view_set,
)
from market_support_crewai_agent.runtime.policy.capabilities.views import (
    VerifierCapabilityViewSetV1,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationMessage,
)
from market_support_crewai_agent.runtime.validation.alignment_transport import (
    InternalAlignmentVerifierSourceV1,
    verify_reply_alignment,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.conversation import (
    DirectPresentationV1,
    GroupPresentationV1,
)
from market_support_crewai_agent.schemas.reply import OutboundAction, ReplyResponse


@dataclass(frozen=True, slots=True)
class AlignmentVerifierInvocationV1:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    scope_authority: BusinessScopeAuthorityV1
    plan: ExecutionPlanV2
    evidence: CanonicalEvidenceExecutionResultV1
    response: ReplyResponse
    reason_code: str
    history: Sequence[ConversationMessage]
    attempt: int
    now: datetime
    locator_safety: LocatorSafetyClassifierV1


def build_runtime_alignment_verifier_input_v1(
    source: AlignmentVerifierInvocationV1,
) -> SanitizedAlignmentVerifierInputV1:
    if source.request.identity.scene != source.policy.scene:
        raise ContextViewInvariantError("verifier_request_policy_scene_mismatch")
    if source.request.business_scope != source.scope_authority.scope:
        raise ContextViewInvariantError("verifier_business_scope_authority_mismatch")
    if source.policy.business_scope_hash != source.scope_authority.business_scope_hash:
        raise ContextViewInvariantError("verifier_policy_scope_hash_mismatch")
    selected = project_capability_view_set(
        stage="alignment_verifier",
        authority=CapabilityViewAuthorityV1(
            eligible_manifest_refs=source.policy.eligible_capabilities,
            selected_manifest_refs=source.plan.selected_manifest_refs,
        ),
    )
    if not isinstance(selected, VerifierCapabilityViewSetV1):
        raise ContextViewInvariantError("verifier_capability_view_role_mismatch")
    return build_alignment_verifier_prompt_input_v1(
        AlignmentVerifierPromptInputSourceV1(
            scene=source.request.identity.scene,
            message=CurrentMessageViewV1(text=source.request.message),
            history=project_alignment_history_v1(source.history, source.now),
            presentation=_presentation_view(source.request),
            validated_plan=project_validated_plan_view_v1(
                source.plan,
                source.scope_authority,
            ),
            directive=_directive_view(source),
            selected_capabilities=selected,
            unit_groundings=project_unit_grounding_views_v1(
                source.plan,
                source.evidence.groundings,
                GroundingProjectionContextV1(
                    business_scope_authority=source.scope_authority,
                    locator_safety=source.locator_safety,
                    evaluation_epoch_seconds=int(source.now.timestamp()),
                ),
            ),
            candidate=_candidate_view(source.response),
            attempt=source.attempt,
        )
    )


def project_alignment_history_v1(
    history: Sequence[ConversationMessage],
    now: datetime,
) -> tuple[HistoryTurnViewV1, ...]:
    return tuple(
        HistoryTurnViewV1(
            role=message.role,
            text=message.content[:1_200],
            age_seconds=(
                None
                if message.created_at.tzinfo is None
                else min(
                    max(0, int((now - message.created_at).total_seconds())),
                    2_592_000,
                )
            ),
        )
        for message in history[-12:]
    )


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
        case _:
            assert_never(request.presentation)


def _directive_view(
    source: AlignmentVerifierInvocationV1,
) -> ResponseDirectiveViewV1:
    match source.plan.response_mode:
        case "knowledge_answer":
            composer_stage = "knowledge_composer"
        case "smalltalk":
            composer_stage = "smalltalk_composer"
        case "action" | "clarification" | "handoff" | "refusal" | "unable" | "no_reply":
            composer_stage = None
        case _:
            assert_never(source.plan.response_mode)
    return ResponseDirectiveViewV1(
        mode=source.plan.response_mode,
        reply_kind=source.response.reply.kind,
        reason_code=source.reason_code,
        requires_composer=composer_stage is not None,
        composer_stage=composer_stage,
        mentions_requested=bool(source.response.reply.mentions),
        action_intent_count=len(source.plan.action_intents),
    )


def _candidate_view(response: ReplyResponse) -> CandidateReplyViewV1:
    return CandidateReplyViewV1(
        reply_kind=response.reply.kind,
        text=response.reply.text,
        mentions=tuple(
            CandidateMentionViewV1(reason=mention.reason)
            for mention in response.reply.mentions
        ),
        actions=tuple(_candidate_action(action) for action in response.actions),
    )


def _candidate_action(action: OutboundAction) -> CandidateActionViewV1:
    match action.type:
        case "send_material_pack":
            return CandidateActionViewV1(
                type=action.type,
                resolve_type=action.resolve_type,
                resolve_ref_available=bool(action.resolve_ref),
                material_pack_option=action.material_pack_option,
            )
        case "send_weekly_report" | "send_monthly_report":
            return CandidateActionViewV1(
                type=action.type,
                resolve_type=action.resolve_type,
                resolve_ref_available=bool(action.resolve_ref),
                period=action.period,
                report_date=action.report_date,
            )
        case _:
            assert_never(action.type)


__all__ = [
    "AlignmentVerifierInvocationV1",
    "InternalAlignmentVerifierSourceV1",
    "build_runtime_alignment_verifier_input_v1",
    "project_alignment_history_v1",
    "verify_reply_alignment",
]
