from __future__ import annotations

# noqa: SIZE_OK - one canonical V2 candidate construction and rendering state machine.
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, Literal, Protocol
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter, ValidationError

from market_support_crewai_agent.runtime.context.models import HistoryTurnViewV1
from market_support_crewai_agent.runtime.context.retry_view_models import (
    ComposerRetryOverlayV1,
)
from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    UnitBusinessFactsV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
    ExecutionUnitGroundingV1,
    ground_execution_plan_v2,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.planning.validation import (
    validate_execution_plan_v2,
)
from market_support_crewai_agent.runtime.policy.compliance import (
    refusal_text_for_reason,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.recall.turn_state import RecallTurnStateV1
from market_support_crewai_agent.runtime.rendering.response_ids import (
    ensure_response_ids,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerInvocationV1,
    V2Composer,
    V2ComposerOutputRejected,
    V2ComposerUnavailable,
    compose_v2_reply,
)
from market_support_crewai_agent.runtime.state.audit_types import ReasonCodeV1
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    image_marker_filenames,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from market_support_crewai_agent.schemas.reply import (
    OutboundAction,
    PrimaryReply,
    ReplyMention,
    ReplyResponse,
    SendMaterialPackAction,
    SendMonthlyReportAction,
    SendWeeklyReportAction,
)
from market_support_crewai_agent.schemas.type_ids import OutboundActionType

V2ReplyKind = Literal[
    "answer", "clarification", "human_handoff", "unable_to_answer", "no_reply"
]
_PRODUCT_LIST_MANIFEST_IDS: Final = frozenset(
    {"weekly_report.product_list", "monthly_report.product_list"}
)
_REASON_CODE_ADAPTER: Final[TypeAdapter[ReasonCodeV1]] = TypeAdapter(ReasonCodeV1)


class CandidatePlanRuntimeV1(Protocol):
    evidence_executor: EvidenceExecutor
    document_cache_config: DocumentMcpCacheConfigV1 | None
    locator_safety: LocatorSafetyClassifierV1
    v2_composer: V2Composer | None


def _reason_code(value: str) -> ReasonCodeV1:
    try:
        return _REASON_CODE_ADAPTER.validate_python(value)
    except ValidationError:
        raise AgentRuntimeError("unregistered_v2_attempt_reason_code")


@dataclass(frozen=True, slots=True)
class V2ReplyValidationResult:
    valid: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class V2AttemptResult:
    plan: ExecutionPlanV2
    evidence: CanonicalEvidenceExecutionResultV1
    response: ReplyResponse
    reply_validation: V2ReplyValidationResult
    reason_code: ReasonCodeV1
    recall_state: RecallTurnStateV1 | None = None


@dataclass(frozen=True, slots=True)
class V2RenderSourceV1:
    request: KernelReplyRequestV1
    policy: PolicyManifestV2
    scope_authority: BusinessScopeAuthorityV1
    plan: ExecutionPlanV2
    evidence: CanonicalEvidenceExecutionResultV1
    history: tuple[HistoryTurnViewV1, ...] = ()
    retry_overlay: ComposerRetryOverlayV1 | None = None


def reply_validation_error_summary(validation: V2ReplyValidationResult) -> str:
    return "; ".join(validation.issues) or "unknown"


async def build_candidate_from_plan_v2(
    runtime: CandidatePlanRuntimeV1,
    *,
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
    plan: ExecutionPlanV2,
    state_key_ref: str | None = None,
    recall_state: RecallTurnStateV1 | None = None,
    alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
) -> V2AttemptResult:
    validation = validate_execution_plan_v2(plan, policy)
    if not validation.valid:
        raise AgentRuntimeError("execution_plan_v2_invalid")
    evidence = await runtime.evidence_executor.execute_v2(
        request,
        plan,
        policy,
        scope_authority=scope_authority,
        state_key_ref=state_key_ref,
        document_cache_config=runtime.document_cache_config,
        alignment_refetch_request=alignment_refetch_request,
    )
    deterministic_plan = _material_pack_action_resolution_plan(
        plan,
        evidence.preflight,
        policy,
        scope_authority,
    )
    if deterministic_plan is not None:
        plan = deterministic_plan
        evidence = replace(
            evidence,
            groundings=ground_execution_plan_v2(
                plan,
                policy,
                evidence.canonical_facts,
                evidence.resolve_bindings,
            ),
        )
    response, reason_code = await _render_v2_response(
        runtime,
        V2RenderSourceV1(
            request=request,
            policy=policy,
            scope_authority=scope_authority,
            plan=plan,
            evidence=evidence,
        ),
    )
    response = ensure_response_ids(response)
    validation_result = _validate_v2_reply(request, policy, plan, response)
    return V2AttemptResult(
        plan=plan,
        evidence=evidence,
        response=response,
        reply_validation=validation_result,
        reason_code=reason_code,
        recall_state=recall_state,
    )


async def _render_v2_response(
    runtime: CandidatePlanRuntimeV1,
    source: V2RenderSourceV1,
) -> tuple[ReplyResponse, ReasonCodeV1]:
    plan = source.plan
    groundings = source.evidence.groundings
    mode = plan.response_mode
    if mode == "refusal":
        return (
            _reply(
                "unable_to_answer",
                refusal_text_for_reason(plan.compliance.reason_code),
            ),
            plan.compliance.reason_code,
        )
    if mode == "clarification":
        return _reply("clarification", _clarification_text(plan)), "ambiguous_request"
    if mode == "handoff":
        return _handoff_reply(source.request, groundings)
    if mode == "no_reply":
        return _reply("no_reply", ""), "no_reply"
    if mode == "action":
        return _action_reply(plan, groundings)
    if mode == "knowledge_answer" or mode == "smalltalk":
        return await _compose_v2_response(runtime, source)
    if mode == "unable":
        if source.request.identity.scene == "group" and plan.origin == "input_policy":
            for decision in plan.guardrail_decisions:
                if decision.phase != "input" or decision.outcome != "block":
                    continue
                handoff_text = decision.metadata.get(
                    HANDOFF_UNAVAILABLE_TEXT_METADATA_KEY
                )
                if isinstance(handoff_text, str) and handoff_text.strip():
                    return (
                        _reply("unable_to_answer", handoff_text),
                        _reason_code(decision.reason_code),
                    )
        return _reply("unable_to_answer", _unable_text()), "insufficient_evidence"
    raise AgentRuntimeError("unsupported_v2_response_mode")


async def _compose_v2_response(
    runtime: CandidatePlanRuntimeV1,
    source: V2RenderSourceV1,
) -> tuple[ReplyResponse, ReasonCodeV1]:
    plan = source.plan
    groundings = source.evidence.groundings
    if _product_list_evidence_is_missing(plan, groundings):
        return _reply(
            "unable_to_answer", _unable_text()
        ), "product_list_evidence_missing"
    try:
        composer_output = await compose_v2_reply(
            runtime,
            ComposerInvocationV1(
                request=source.request,
                policy=source.policy,
                scope_authority=source.scope_authority,
                plan=plan,
                preflight=source.evidence.preflight,
                groundings=groundings,
                media_bindings=source.evidence.media_bindings,
                locator_safety=runtime.locator_safety,
                now=datetime.now(ZoneInfo("Asia/Shanghai")),
                history=source.history,
                retry_overlay=source.retry_overlay,
            ),
        )
    except V2ComposerUnavailable:
        return _reply("unable_to_answer", _unable_text()), "composer_not_available"
    except V2ComposerOutputRejected:
        return _reply("unable_to_answer", _unable_text()), "composer_output_rejected"
    if plan.response_mode == "knowledge_answer":
        return composer_output.to_reply_response(), "knowledge_answer_composer"
    if plan.response_mode == "smalltalk":
        return composer_output.to_reply_response(), "smalltalk_composer"
    raise AgentRuntimeError("composer_response_mode_not_supported")


async def recompose_candidate_v2(
    runtime: CandidatePlanRuntimeV1,
    *,
    candidate: V2AttemptResult,
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
    history: tuple[HistoryTurnViewV1, ...],
    retry_overlay: ComposerRetryOverlayV1,
) -> V2AttemptResult:
    if candidate.plan.response_mode not in {"knowledge_answer", "smalltalk"}:
        raise AgentRuntimeError("alignment_recompose_mode_not_supported")
    response, reason_code = await _render_v2_response(
        runtime,
        V2RenderSourceV1(
            request=request,
            policy=policy,
            scope_authority=scope_authority,
            plan=candidate.plan,
            evidence=candidate.evidence,
            history=history,
            retry_overlay=retry_overlay,
        ),
    )
    response = ensure_response_ids(response)
    return replace(
        candidate,
        response=response,
        reply_validation=_validate_v2_reply(
            request,
            policy,
            candidate.plan,
            response,
        ),
        reason_code=reason_code,
    )


def _product_list_evidence_is_missing(
    plan: ExecutionPlanV2,
    groundings: tuple[ExecutionUnitGroundingV1, ...],
) -> bool:
    product_list_unit_ids = {
        unit.unit_id
        for unit in plan.units
        if unit.manifest_ref.manifest_id in _PRODUCT_LIST_MANIFEST_IDS
    }
    if not product_list_unit_ids:
        return False
    evidence_by_unit = {
        grounding.unit_id: bool(grounding.allowed_evidence) for grounding in groundings
    }
    return any(
        not evidence_by_unit.get(unit_id, False) for unit_id in product_list_unit_ids
    )


def _reply(kind: V2ReplyKind, text: str) -> ReplyResponse:
    return ReplyResponse(
        reply=PrimaryReply(kind=kind, text=text, mentions=[]), actions=[]
    )


def _clarification_text(plan: ExecutionPlanV2) -> str:
    slots = tuple(slot for unit in plan.units for slot in unit.ambiguity_slots)
    if "material_pack_option" in slots:
        return "老师，麻烦确认一下需要哪一类材料，我再继续处理。"
    return "老师，麻烦补充一下具体需求，我再继续处理。"


def _handoff_reply(
    request: KernelReplyRequestV1,
    groundings: tuple[ExecutionUnitGroundingV1, ...],
) -> tuple[ReplyResponse, ReasonCodeV1]:
    if request.identity.scene == "direct":
        return _reply(
            "human_handoff", "如需人工协助，请联系您的客户经理或人工客服继续处理。"
        ), "direct_human_handoff"
    sales_available = any(
        grounding.business_facts.sales_mention.availability == "available"
        for grounding in groundings
    )
    if sales_available:
        return (
            ReplyResponse(
                reply=PrimaryReply(
                    kind="human_handoff",
                    text="这个问题我帮你请销售/支持同事确认。",
                    mentions=[ReplyMention(type="sales", reason="handoff_requested")],
                ),
                actions=[],
            ),
            "handoff_requested",
        )
    return _reply("unable_to_answer", _unable_text()), "sales_mention_unavailable"


def _action_reply(
    plan: ExecutionPlanV2,
    groundings: tuple[ExecutionUnitGroundingV1, ...],
) -> tuple[ReplyResponse, ReasonCodeV1]:
    actions: list[OutboundAction] = []
    for unit, grounding in zip(plan.units, groundings, strict=True):
        if grounding.business_facts.user_permission != "allowed":
            continue
        for intent in unit.action_intents:
            action = _action_from_grounding(
                intent.action_type,
                intent.material_pack_option,
                grounding.business_facts,
            )
            if action is not None:
                actions.append(action)
    if not actions:
        return _reply("unable_to_answer", _unable_text()), "action_evidence_missing"
    return ReplyResponse(
        reply=PrimaryReply(kind="answer", text="", mentions=[]), actions=actions
    ), "action_ready"


def _material_pack_action_resolution_plan(
    plan: ExecutionPlanV2,
    preflight: AdapterPreflightSnapshot,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
) -> ExecutionPlanV2 | None:
    if (
        len(plan.action_intents) != 1
        or plan.action_intents[0].action_type != "send_material_pack"
    ):
        return None
    material_result = next(
        (
            item.result
            for item in preflight.items
            if item.resolve_type == "material_pack"
        ),
        None,
    )
    if material_result is None:
        return None
    status = material_result.status
    if status == "resolved":
        return None
    if status == "ambiguous":
        manifest_id = "general.clarification"
        answerability = "clarify"
        ambiguity_slots = ("material_pack_option",)
    else:
        sales_result = next(
            (
                item.result
                for item in preflight.items
                if item.resolve_type == "sales_mention"
            ),
            None,
        )
        sales_available = sales_result is not None and sales_result.status == "resolved"
        manifest_id = "sales.handoff" if sales_available else "general.abstention"
        answerability = "handoff" if sales_available else "abstain"
        ambiguity_slots = ()
    return finalize_execution_plan_v2(
        DeterministicPlanOriginInputV1(
            user_need=plan.user_need,
            units=(
                DeterministicPlanUnitV1(
                    unit_id="material-pack-resolution",
                    manifest_id=manifest_id,
                    answerability_policy=answerability,
                    ambiguity_slots=ambiguity_slots,
                ),
            ),
            compliance_reason_code=plan.compliance.reason_code,
            confidence=1.0,
        ),
        policy,
        scope_authority,
        origin="deterministic",
    )


def _action_from_grounding(
    action_type: OutboundActionType,
    material_pack_option: str | None,
    facts: UnitBusinessFactsV1,
) -> OutboundAction | None:
    if action_type == "send_material_pack":
        state = facts.material_pack
        if state.availability != "available" or state.resolve_ref is None:
            return None
        return SendMaterialPackAction(
            type="send_material_pack",
            resolve_type="material_pack",
            resolve_ref=state.resolve_ref,
            material_pack_option=material_pack_option or state.material_pack_option,
        )
    if action_type == "send_weekly_report":
        state = facts.weekly_report
        if (
            state.availability != "available"
            or state.resolve_ref is None
            or state.period is None
            or state.report_date is None
        ):
            return None
        return SendWeeklyReportAction(
            type="send_weekly_report",
            resolve_type="weekly_report",
            resolve_ref=state.resolve_ref,
            period=state.period,
            report_date=state.report_date.isoformat(),
        )
    if action_type == "send_monthly_report":
        state = facts.monthly_report
        if (
            state.availability != "available"
            or state.resolve_ref is None
            or state.period is None
            or state.report_date is None
        ):
            return None
        return SendMonthlyReportAction(
            type="send_monthly_report",
            resolve_type="monthly_report",
            resolve_ref=state.resolve_ref,
            period=state.period,
            report_date=state.report_date.isoformat(),
        )


def _validate_v2_reply(
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    plan: ExecutionPlanV2,
    response: ReplyResponse,
) -> V2ReplyValidationResult:
    issues: list[str] = []
    if request.identity.scene == "direct" and (
        response.actions or response.reply.mentions
    ):
        issues.append("direct_reply_has_actions_or_mentions")
    if request.identity.scene == "direct" and image_marker_filenames(
        response.reply.text
    ):
        issues.append("direct_reply_has_image_marker")
    if plan.response_mode == "action" and not response.actions:
        issues.append("action_plan_without_validated_action")
    if response.actions and not policy.actions_allowed:
        issues.append("reply_actions_outside_policy")
    if response.reply.mentions and not policy.mentions_allowed:
        issues.append("reply_mentions_outside_policy")
    return V2ReplyValidationResult(valid=not issues, issues=tuple(issues))


def _unable_text() -> str:
    return "老师，这个信息我这边暂时无法确认，先不回答避免信息不准确。"
