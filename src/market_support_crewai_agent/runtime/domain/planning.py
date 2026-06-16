from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, model_validator

from market_support_crewai_agent.runtime.domain.capabilities import (
    ArtifactKind,
    CapabilityName,
    ResponseMode,
    capability_by_action_type,
    capability_by_name,
)
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.compliance_policy import ComplianceReasonCode
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.send_scope_guard import (
    detect_send_scope_conflict,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveType,
    ReplyRequest,
    SideEffectActionType,
    StrictModel,
)

ActionIntent = Literal["send", "answer", "handoff", "refuse", "none"]
WorkItemIntent = Literal["send", "answer", "handoff"]
StrategyMentionSource = Literal["canonical_context", "message", "history", "unknown"]
IntentReportScope = Literal["channel_all", "strategy", "ambiguous", "none"]
ActionReportScope = Literal["channel_all", "strategy", "none"]
AmbiguitySlot = Literal["artifact", "strategy", "report_scope", "request_meaning"]
PlanValidationSeverity = Literal["error", "fatal"]
PlanValidationCode = Literal[
    "intent_frame_invalid_contract",
    "response_mode_not_allowed",
    "capability_not_allowed",
    "adapter_resolve_not_allowed",
    "too_many_evidence_calls",
    "action_not_allowed",
    "action_capability_mismatch",
    "action_missing_required_resolve",
    "report_action_selector_missing",
    "report_action_strategy_selector_missing_strategy",
    "report_action_channel_all_selector_has_strategy",
    "non_compliant_plan_has_actions",
    "non_compliant_plan_not_refusal",
    "unknown_compliance_has_actions",
    "ambiguous_plan_has_actions",
    "ambiguous_plan_not_clarification",
    "knowledge_answer_missing_capability",
]
_KNOWLEDGE_REPORT_CAPABILITIES = ("weekly_report", "monthly_report")


class ComplianceDecision(StrictModel):
    is_compliant: bool | None = Field(
        default=None,
        description=(
            "True when the request is compliant, false when it must be refused, "
            "null when not enough context."
        ),
    )
    reason_code: ComplianceReasonCode = "unknown"
    reason: str = Field(default="", max_length=400)


class StrategyMention(StrictModel):
    raw_text: str = Field(min_length=1)
    canonical_name: str | None = None
    source: StrategyMentionSource = "unknown"


class IntentWorkItem(StrictModel):
    intent: WorkItemIntent
    capability: CapabilityName
    evidence_query: str | None = Field(default=None, max_length=200)
    selected_strategy: str | None = None
    report_scope: IntentReportScope = "none"


class IntentFrame(StrictModel):
    contract_version: Literal["intent-frame"] = "intent-frame"
    user_need: str = Field(min_length=1, max_length=500)
    artifact_kind: ArtifactKind
    action_intent: ActionIntent
    compliance: ComplianceDecision
    evidence_query: str | None = Field(default=None, max_length=200)
    strategy_mentions: list[StrategyMention] = Field(default_factory=list, max_length=8)
    selected_strategy: str | None = None
    report_scope: IntentReportScope = "none"
    ambiguity_slots: list[AmbiguitySlot] = Field(default_factory=list)
    requested_capabilities: list[CapabilityName] = Field(default_factory=list, max_length=8)
    work_items: list[IntentWorkItem] = Field(default_factory=list, max_length=6)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_non_compliant_shape(self):
        if self.compliance.is_compliant is False and self.action_intent != "refuse":
            raise ValueError(
                "non-compliant intent frames must use action_intent=refuse"
            )
        return self


class AdapterResolveSpec(StrictModel):
    resolve_type: AdapterResolveType
    strategy: str | None = None


class ActionIntentSpec(StrictModel):
    action_type: SideEffectActionType
    capability: CapabilityName
    report_scope: ActionReportScope = "none"
    strategy: str | None = None


@dataclass(frozen=True)
class SendActionTarget:
    capability_name: CapabilityName
    selected_strategy: str | None = None
    report_scope: IntentReportScope = "none"


class ExecutionPlan(StrictModel):
    contract_version: Literal["execution-plan"] = "execution-plan"
    user_need: str = Field(min_length=1, max_length=500)
    artifact_kind: ArtifactKind
    response_mode: ResponseMode
    compliance: ComplianceDecision
    evidence_query: str | None = Field(default=None, max_length=200)
    capabilities: list[CapabilityName] = Field(default_factory=list, max_length=8)
    answer_capabilities: list[CapabilityName] = Field(default_factory=list, max_length=4)
    adapter_resolves: list[AdapterResolveSpec] = Field(default_factory=list, max_length=8)
    action_intents: list[ActionIntentSpec] = Field(default_factory=list, max_length=4)
    selected_strategy: str | None = None
    ambiguity_slots: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass(frozen=True)
class PlanValidationIssue:
    code: PlanValidationCode
    message: str
    severity: PlanValidationSeverity = "error"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    issues: tuple[PlanValidationIssue, ...] = ()

    @property
    def fatal(self) -> bool:
        return any(issue.severity == "fatal" for issue in self.issues)


def compile_intent_frame(
    frame: IntentFrame,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    policy: PolicyManifest,
) -> ExecutionPlan:
    selected_strategy = _selected_strategy(frame, canonical_context)
    send_targets = _send_action_capabilities(frame)
    knowledge_capabilities = _knowledge_capabilities(frame, policy)
    knowledge_evidence_query = _knowledge_evidence_query(frame)
    if _send_scope_conflict(frame, request, canonical_context, send_targets):
        return _plan(
            frame,
            response_mode="unable",
            capabilities=[],
            answer_capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    frame = _normalize_send_defaults(frame, request, canonical_context, send_targets)
    selected_strategy = _selected_strategy(frame, canonical_context)
    send_targets = _send_action_capabilities(frame)
    knowledge_capabilities = _knowledge_capabilities(frame, policy)
    knowledge_evidence_query = _knowledge_evidence_query(frame)

    if frame.compliance.is_compliant is False:
        return _plan(
            frame,
            response_mode="refusal",
            capabilities=[],
            answer_capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    ambiguity_slots = _effective_ambiguity_slots(frame, send_targets)
    if ambiguity_slots or (
        frame.artifact_kind == "unclear"
        and not send_targets
        and not knowledge_capabilities
    ):
        return _plan(
            frame,
            response_mode="clarification",
            capabilities=_capabilities(frame.requested_capabilities),
            answer_capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
            ambiguity_slots=ambiguity_slots or ["request_meaning"],
        )

    if frame.compliance.is_compliant is not True and _has_send_intent(frame):
        return _plan(
            frame,
            response_mode="unable",
            capabilities=[],
            answer_capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    if _has_send_intent(frame) and send_targets:
        report_scope = _report_scope_for_send_targets(
            frame,
            send_targets,
            selected_strategy,
            canonical_context,
        )
        if report_scope == "ambiguous":
            return _plan(
                frame,
                response_mode="clarification",
                capabilities=_capabilities(frame.requested_capabilities, *send_targets),
                answer_capabilities=[],
                adapter_resolves=[],
                action_intents=[],
                selected_strategy=selected_strategy,
                ambiguity_slots=["strategy"],
            )
        if knowledge_capabilities:
            return _send_action_plan(
                frame,
                capability_names=send_targets,
                selected_strategy=selected_strategy,
                report_scope=report_scope,
                answer_capabilities=knowledge_capabilities,
                knowledge_evidence_query=knowledge_evidence_query,
            )
        return _send_action_plan(
            frame,
            capability_names=send_targets,
            selected_strategy=selected_strategy,
            report_scope=report_scope,
        )

    if frame.artifact_kind == "knowledge_answer" and frame.action_intent == "answer":
        if not knowledge_capabilities:
            return _plan(
                frame,
                response_mode="unable",
                capabilities=[],
                answer_capabilities=[],
                adapter_resolves=[],
                action_intents=[],
                selected_strategy=selected_strategy,
            )
        return _plan(
            frame,
            response_mode="knowledge_answer",
            capabilities=knowledge_capabilities,
            answer_capabilities=knowledge_capabilities,
            adapter_resolves=_knowledge_adapter_resolves(
                knowledge_capabilities,
                selected_strategy,
            ),
            action_intents=[],
            selected_strategy=selected_strategy,
            evidence_query=knowledge_evidence_query,
        )

    if frame.artifact_kind == "human_support" or frame.action_intent == "handoff":
        return _plan(
            frame,
            response_mode="handoff",
            capabilities=["sales_mention"],
            answer_capabilities=[],
            adapter_resolves=_adapter_resolves("sales_mention", None),
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    if frame.artifact_kind == "smalltalk" and frame.action_intent == "none":
        return _plan(
            frame,
            response_mode="smalltalk",
            capabilities=[],
            answer_capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    if frame.action_intent == "refuse" or frame.artifact_kind == "refusal":
        return _plan(
            frame,
            response_mode="refusal",
            capabilities=[],
            answer_capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    return _plan(
        frame,
        response_mode="unable",
        capabilities=[],
        answer_capabilities=[],
        adapter_resolves=[],
        action_intents=[],
        selected_strategy=selected_strategy,
    )


def validate_execution_plan(
    plan: ExecutionPlan,
    policy: PolicyManifest,
) -> PlanValidationResult:
    issues: list[PlanValidationIssue] = []

    if plan.response_mode not in policy.allowed_reply_modes:
        issues.append(
            PlanValidationIssue(
                code="response_mode_not_allowed",
                message=f"response mode {plan.response_mode} is not allowed",
                severity="fatal",
                metadata={"response_mode": plan.response_mode},
            )
        )

    for capability in plan.capabilities:
        if capability not in policy.allowed_capabilities:
            issues.append(
                PlanValidationIssue(
                    code="capability_not_allowed",
                    message=f"capability {capability} is not allowed",
                    severity="fatal",
                    metadata={"capability": capability},
                )
            )

    evidence_call_count = len(plan.adapter_resolves) + (
        1 if "document_context" in plan.capabilities else 0
    )
    if evidence_call_count > policy.evidence_call_limit:
        issues.append(
            PlanValidationIssue(
                code="too_many_evidence_calls",
                message="plan exceeds policy evidence call limit",
                metadata={
                    "limit": policy.evidence_call_limit,
                    "actual": evidence_call_count,
                },
            )
        )

    adapter_resolve_types = {item.resolve_type for item in plan.adapter_resolves}
    for resolve_spec in plan.adapter_resolves:
        if resolve_spec.resolve_type not in policy.allowed_adapter_resolves:
            issues.append(
                PlanValidationIssue(
                    code="adapter_resolve_not_allowed",
                    message=f"adapter resolve {resolve_spec.resolve_type} is not allowed",
                    severity="fatal",
                    metadata={"resolve_type": resolve_spec.resolve_type},
                )
            )

    for action in plan.action_intents:
        if action.action_type not in policy.allowed_side_effect_actions:
            issues.append(
                PlanValidationIssue(
                    code="action_not_allowed",
                    message=f"action {action.action_type} is not allowed",
                    severity="fatal",
                    metadata={"action_type": action.action_type},
                )
            )
            continue

        capability = capability_by_action_type(action.action_type)
        expected_capability = capability.name if capability is not None else None
        expected_resolve = capability.resolve_type if capability is not None else None
        if expected_capability != action.capability:
            issues.append(
                PlanValidationIssue(
                    code="action_capability_mismatch",
                    message="action capability does not match registry",
                    severity="fatal",
                    metadata={
                        "action_type": action.action_type,
                        "capability": action.capability,
                        "expected_capability": expected_capability,
                    },
                )
            )
            continue
        if expected_resolve is not None and expected_resolve not in adapter_resolve_types:
            issues.append(
                PlanValidationIssue(
                    code="action_missing_required_resolve",
                    message=f"{action.action_type} must require {expected_resolve}",
                    metadata={
                        "action_type": action.action_type,
                        "resolve_type": expected_resolve,
                    },
                )
            )
        if capability is not None and capability.is_report:
            issues.extend(_validate_report_action_selector(action))

    if plan.compliance.is_compliant is False:
        if plan.response_mode != "refusal":
            issues.append(
                PlanValidationIssue(
                    code="non_compliant_plan_not_refusal",
                    message="non-compliant plan must use refusal mode",
                    severity="fatal",
                )
            )
        if plan.capabilities or plan.adapter_resolves or plan.action_intents:
            issues.append(
                PlanValidationIssue(
                    code="non_compliant_plan_has_actions",
                    message="non-compliant plan must not propose capabilities, resolves, or actions",
                    severity="fatal",
                )
            )

    if plan.compliance.is_compliant is None and plan.action_intents:
        issues.append(
            PlanValidationIssue(
                code="unknown_compliance_has_actions",
                message="unknown compliance status cannot propose side-effect actions",
                severity="fatal",
            )
        )

    if plan.ambiguity_slots:
        if plan.response_mode != "clarification":
            issues.append(
                PlanValidationIssue(
                    code="ambiguous_plan_not_clarification",
                    message="plans with ambiguity slots must use clarification mode",
                    severity="fatal",
                    metadata={"ambiguity_slots": list(plan.ambiguity_slots)},
                )
            )
        if plan.action_intents:
            issues.append(
                PlanValidationIssue(
                    code="ambiguous_plan_has_actions",
                    message="ambiguous plans must not propose side-effect actions",
                    severity="fatal",
                    metadata={"ambiguity_slots": list(plan.ambiguity_slots)},
                )
            )

    if (
        plan.response_mode == "knowledge_answer"
        and "document_context" not in plan.capabilities
        and not plan.adapter_resolves
    ):
        issues.append(
            PlanValidationIssue(
                code="knowledge_answer_missing_capability",
                message="knowledge_answer mode requires document_context or report resolve evidence",
                severity="fatal",
            )
        )

    return PlanValidationResult(valid=not issues, issues=tuple(issues))


def invalid_intent_validation(
    reason: str = "planner output did not match IntentFrame contract",
) -> PlanValidationResult:
    return PlanValidationResult(
        valid=False,
        issues=(
            PlanValidationIssue(
                code="intent_frame_invalid_contract",
                message=reason,
                severity="fatal",
            ),
        ),
    )


def _send_action_plan(
    frame: IntentFrame,
    *,
    capability_names: list[CapabilityName],
    selected_strategy: str | None,
    report_scope: IntentReportScope,
    answer_capabilities: list[CapabilityName] | None = None,
    knowledge_evidence_query: str | None = None,
) -> ExecutionPlan:
    action_intents: list[ActionIntentSpec] = []
    adapter_resolves: list[AdapterResolveSpec] = []
    for target in _send_action_targets(
        frame,
        capability_names,
        selected_strategy,
        report_scope,
    ):
        capability = capability_by_name(target.capability_name)
        if capability is None or capability.side_effect_action_type is None:
            continue
        action_report_scope = _action_report_scope(
            target.capability_name,
            target.report_scope,
        )
        action_strategy = _action_strategy(
            target.capability_name,
            target.selected_strategy,
            action_report_scope,
        )
        action_intents.append(
            ActionIntentSpec(
                action_type=capability.side_effect_action_type,
                capability=target.capability_name,
                report_scope=action_report_scope,
                strategy=action_strategy,
            )
        )
        adapter_resolves.extend(_adapter_resolves(target.capability_name, action_strategy))

    answer_capabilities = _capabilities(answer_capabilities or [])
    answer_resolves = _knowledge_adapter_resolves(
        answer_capabilities,
        selected_strategy,
    )
    response_mode: ResponseMode = "action"

    return _plan(
        frame,
        response_mode=response_mode,
        capabilities=_capabilities(
            frame.requested_capabilities,
            *capability_names,
            *answer_capabilities,
        ),
        answer_capabilities=answer_capabilities,
        adapter_resolves=(
            adapter_resolves
            + answer_resolves
            + _adapter_resolves("sales_mention", None)
        ),
        action_intents=action_intents,
        selected_strategy=selected_strategy,
        evidence_query=knowledge_evidence_query,
    )


def _plan(
    frame: IntentFrame,
    *,
    response_mode: ResponseMode,
    capabilities: list[CapabilityName],
    answer_capabilities: list[CapabilityName],
    adapter_resolves: list[AdapterResolveSpec],
    action_intents: list[ActionIntentSpec],
    selected_strategy: str | None,
    ambiguity_slots: list[str] | None = None,
    evidence_query: str | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        user_need=frame.user_need,
        artifact_kind=frame.artifact_kind,
        response_mode=response_mode,
        compliance=frame.compliance,
        evidence_query=frame.evidence_query if evidence_query is None else evidence_query,
        capabilities=_capabilities(capabilities),
        answer_capabilities=_capabilities(answer_capabilities),
        adapter_resolves=_unique_adapter_resolves(adapter_resolves),
        action_intents=action_intents,
        selected_strategy=selected_strategy,
        ambiguity_slots=ambiguity_slots or [],
        confidence=frame.confidence,
    )


def _selected_strategy(
    frame: IntentFrame,
    canonical_context: CanonicalContext,
) -> str | None:
    return frame.selected_strategy or canonical_context.selected_strategy


def _send_scope_conflict(
    frame: IntentFrame,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    send_targets: list[CapabilityName],
) -> bool:
    if not _has_send_intent(frame):
        return False
    for capability_name in send_targets or _send_action_capabilities(frame):
        capability = capability_by_name(capability_name)
        if capability is None:
            continue
        if (
            detect_send_scope_conflict(
                request,
                canonical_context,
                capability.artifact_kind,
            )
            is not None
        ):
            return True
    return False


def _normalize_send_defaults(
    frame: IntentFrame,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    send_targets: list[CapabilityName],
) -> IntentFrame:
    if not _has_send_intent(frame) or not send_targets:
        return frame

    updates: dict[str, object] = {}
    ambiguity_slots = _effective_ambiguity_slots(frame, send_targets)

    if _has_report_send_target(send_targets):
        normalized_report = _normalize_report_send_defaults(
            frame.model_copy(update={"ambiguity_slots": ambiguity_slots}),
            request,
            canonical_context,
        )
        updates["ambiguity_slots"] = normalized_report.ambiguity_slots
        updates["report_scope"] = normalized_report.report_scope
        if normalized_report.selected_strategy:
            updates["selected_strategy"] = normalized_report.selected_strategy
    elif ambiguity_slots != frame.ambiguity_slots:
        updates["ambiguity_slots"] = ambiguity_slots

    if not updates:
        return frame
    return frame.model_copy(update=updates)


def _send_action_capabilities(frame: IntentFrame) -> list[CapabilityName]:
    work_item_capabilities = [
        item.capability
        for item in frame.work_items
        if item.intent == "send" and _is_side_effect_capability(item.capability)
    ]
    if work_item_capabilities:
        return _capabilities(work_item_capabilities)

    if frame.action_intent != "send":
        return []

    requested = [
        capability_name
        for capability_name in frame.requested_capabilities
        if _is_side_effect_capability(capability_name)
    ]
    if requested:
        return _capabilities(requested)

    for capability_name in ("material_pack", "weekly_report", "monthly_report"):
        capability = capability_by_name(capability_name)
        if capability is not None and capability.artifact_kind == frame.artifact_kind:
            return [capability_name]
    return []


def _has_send_intent(frame: IntentFrame) -> bool:
    return frame.action_intent == "send" or any(
        item.intent == "send" for item in frame.work_items
    )


def _is_side_effect_capability(capability_name: CapabilityName | str) -> bool:
    capability = capability_by_name(capability_name)
    return capability is not None and capability.side_effect_action_type is not None


def _effective_ambiguity_slots(
    frame: IntentFrame,
    send_targets: list[CapabilityName],
) -> list[str]:
    slots = list(frame.ambiguity_slots)
    if frame.work_items or (frame.action_intent == "send" and len(send_targets) > 1):
        slots = [slot for slot in slots if slot != "artifact"]
    return slots


def _send_action_targets(
    frame: IntentFrame,
    capability_names: list[CapabilityName],
    selected_strategy: str | None,
    report_scope: IntentReportScope,
) -> list[SendActionTarget]:
    item_targets = [
        SendActionTarget(
            capability_name=item.capability,
            selected_strategy=item.selected_strategy or selected_strategy,
            report_scope=(
                item.report_scope
                if item.report_scope != "none"
                else report_scope
            ),
        )
        for item in frame.work_items
        if item.intent == "send" and item.capability in capability_names
    ]
    if item_targets:
        return item_targets
    return [
        SendActionTarget(
            capability_name=capability_name,
            selected_strategy=selected_strategy,
            report_scope=report_scope,
        )
        for capability_name in capability_names
    ]


def _has_report_send_target(send_targets: list[CapabilityName]) -> bool:
    return any(
        (capability := capability_by_name(capability_name)) is not None
        and capability.is_report
        for capability_name in send_targets
    )


def _normalize_report_send_defaults(
    frame: IntentFrame,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
) -> IntentFrame:
    if frame.compliance.is_compliant is False:
        return frame
    if not _has_send_intent(frame):
        return frame
    if not frame.ambiguity_slots and frame.report_scope != "ambiguous":
        return frame
    if not _report_scope_can_default(request, frame, canonical_context):
        return frame

    selected_strategy = _selected_strategy(frame, canonical_context)
    updates: dict[str, object] = {}
    if frame.ambiguity_slots:
        updates["ambiguity_slots"] = [
            slot
            for slot in frame.ambiguity_slots
            if slot not in {"strategy", "report_scope"}
        ]
    if frame.report_scope == "ambiguous":
        updates["report_scope"] = "strategy" if selected_strategy else "channel_all"
    if selected_strategy:
        updates["selected_strategy"] = selected_strategy
    if not updates:
        return frame
    return frame.model_copy(update=updates)


def _report_scope_can_default(
    request: ReplyRequest,
    frame: IntentFrame,
    canonical_context: CanonicalContext,
) -> bool:
    if canonical_context.strategy_status == "ambiguous":
        return False
    selected_strategy = _selected_strategy(frame, canonical_context)
    if selected_strategy:
        return True
    return not _message_requests_unnamed_strategy_report(request.message)


def _message_requests_unnamed_strategy_report(message: str) -> bool:
    text = _user_visible_text(message)
    if not any(token in text for token in ("周报", "月报", "weekly", "monthly")):
        return False
    return any(
        token in text
        for token in (
            "某个策略",
            "具体策略",
            "哪个策略",
            "哪一个策略",
            "策略的",
            "这个策略",
        )
    )


def _user_visible_text(message: str) -> str:
    lines = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("[adapter_") and stripped.endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip().lower()


def _report_scope(
    frame: IntentFrame,
    selected_strategy: str | None,
    canonical_context: CanonicalContext,
) -> IntentReportScope:
    if frame.report_scope == "ambiguous" or canonical_context.strategy_status == "ambiguous":
        return "ambiguous"
    if frame.report_scope == "channel_all":
        return "channel_all"
    if frame.report_scope == "strategy":
        return "strategy" if selected_strategy else "ambiguous"
    if selected_strategy:
        return "strategy"
    return "channel_all"


def _report_scope_for_send_targets(
    frame: IntentFrame,
    send_targets: list[CapabilityName],
    selected_strategy: str | None,
    canonical_context: CanonicalContext,
) -> IntentReportScope:
    if not _has_report_send_target(send_targets):
        return "none"
    return _report_scope(frame, selected_strategy, canonical_context)


def _action_report_scope(
    capability_name: CapabilityName,
    report_scope: IntentReportScope,
) -> ActionReportScope:
    capability = capability_by_name(capability_name)
    if capability is None or not capability.is_report:
        return "none"
    if report_scope in {"channel_all", "strategy"}:
        return report_scope
    return "channel_all"


def _action_strategy(
    capability_name: CapabilityName,
    selected_strategy: str | None,
    report_scope: ActionReportScope,
) -> str | None:
    capability = capability_by_name(capability_name)
    if capability is None:
        return None
    if capability.is_report:
        return selected_strategy if report_scope == "strategy" else None
    return selected_strategy


def _adapter_resolves(
    capability_name: CapabilityName,
    strategy: str | None,
) -> list[AdapterResolveSpec]:
    capability = capability_by_name(capability_name)
    if capability is None or capability.resolve_type is None:
        return []
    return [
        AdapterResolveSpec(
            resolve_type=capability.resolve_type,
            strategy=strategy,
        )
    ]


def _knowledge_capabilities(
    frame: IntentFrame,
    policy: PolicyManifest,
) -> list[CapabilityName]:
    work_item_capabilities = [
        item.capability
        for item in frame.work_items
        if item.intent == "answer"
    ]
    if work_item_capabilities:
        return _allowed_knowledge_capabilities(work_item_capabilities, policy)

    if frame.artifact_kind != "knowledge_answer" or frame.action_intent != "answer":
        return []

    return _allowed_knowledge_capabilities(frame.requested_capabilities, policy)


def _allowed_knowledge_capabilities(
    capability_names: list[CapabilityName],
    policy: PolicyManifest,
) -> list[CapabilityName]:
    output: list[CapabilityName] = []
    for capability_name in capability_names:
        if capability_name == "document_context":
            if capability_name in policy.allowed_capabilities:
                output.append(capability_name)
            continue
        if capability_name in _KNOWLEDGE_REPORT_CAPABILITIES:
            if capability_name in policy.allowed_capabilities:
                output.append(capability_name)
    return _capabilities(output)


def _knowledge_evidence_query(frame: IntentFrame) -> str | None:
    for item in frame.work_items:
        if item.intent == "answer":
            return item.evidence_query
    return frame.evidence_query


def _knowledge_adapter_resolves(
    capabilities: list[CapabilityName],
    selected_strategy: str | None,
) -> list[AdapterResolveSpec]:
    resolves: list[AdapterResolveSpec] = []
    for capability_name in capabilities:
        if capability_name not in _KNOWLEDGE_REPORT_CAPABILITIES:
            continue
        resolves.extend(_adapter_resolves(capability_name, selected_strategy))
    return resolves


def _capabilities(
    values: list[CapabilityName] | tuple[CapabilityName, ...],
    *extra: CapabilityName,
) -> list[CapabilityName]:
    seen = set()
    output: list[CapabilityName] = []
    for value in [*values, *extra]:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _unique_adapter_resolves(
    values: list[AdapterResolveSpec],
) -> list[AdapterResolveSpec]:
    seen = set()
    output: list[AdapterResolveSpec] = []
    for value in values:
        key = (value.resolve_type, value.strategy)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _validate_report_action_selector(
    action: ActionIntentSpec,
) -> list[PlanValidationIssue]:
    if action.report_scope == "none":
        return [
            PlanValidationIssue(
                code="report_action_selector_missing",
                message=f"{action.action_type} must declare report_scope",
                metadata={"action_type": action.action_type},
            )
        ]
    if action.report_scope == "strategy" and not action.strategy:
        return [
            PlanValidationIssue(
                code="report_action_strategy_selector_missing_strategy",
                message=f"{action.action_type} with report_scope=strategy must include a strategy",
                metadata={"action_type": action.action_type},
            )
        ]
    if action.report_scope == "channel_all" and action.strategy:
        return [
            PlanValidationIssue(
                code="report_action_channel_all_selector_has_strategy",
                message=f"{action.action_type} with report_scope=channel_all must not include a strategy",
                metadata={
                    "action_type": action.action_type,
                    "strategy": action.strategy,
                },
            )
        ]
    return []
