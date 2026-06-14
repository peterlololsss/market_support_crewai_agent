from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, model_validator

from market_support_crewai_agent.runtime.capabilities import (
    ArtifactKind,
    CapabilityName,
    ResponseMode,
    capability_by_action_type,
    capability_by_name,
)
from market_support_crewai_agent.runtime.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.compliance_policy import ComplianceReasonCode
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.schemas import (
    AdapterResolveType,
    ReplyRequest,
    SideEffectActionType,
    StrictModel,
)

ActionIntent = Literal["send", "answer", "handoff", "refuse", "none"]
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


class IntentFrame(StrictModel):
    contract_version: Literal["intent-frame"] = "intent-frame"
    user_need: str = Field(min_length=1, max_length=500)
    artifact_kind: ArtifactKind
    action_intent: ActionIntent
    compliance: ComplianceDecision
    strategy_mentions: list[StrategyMention] = Field(default_factory=list, max_length=8)
    selected_strategy: str | None = None
    report_scope: IntentReportScope = "none"
    ambiguity_slots: list[AmbiguitySlot] = Field(default_factory=list)
    requested_capabilities: list[CapabilityName] = Field(default_factory=list, max_length=8)
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


class ExecutionPlan(StrictModel):
    contract_version: Literal["execution-plan"] = "execution-plan"
    user_need: str = Field(min_length=1, max_length=500)
    artifact_kind: ArtifactKind
    response_mode: ResponseMode
    compliance: ComplianceDecision
    capabilities: list[CapabilityName] = Field(default_factory=list, max_length=8)
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

    if frame.compliance.is_compliant is False:
        return _plan(
            frame,
            response_mode="refusal",
            capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    if frame.ambiguity_slots or frame.artifact_kind == "unclear":
        return _plan(
            frame,
            response_mode="clarification",
            capabilities=_capabilities(frame.requested_capabilities),
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
            ambiguity_slots=frame.ambiguity_slots or ["request_meaning"],
        )

    if frame.compliance.is_compliant is not True and frame.action_intent == "send":
        return _plan(
            frame,
            response_mode="unable",
            capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    if frame.artifact_kind == "material_pack" and frame.action_intent == "send":
        return _action_plan(
            frame,
            capability_name="material_pack",
            selected_strategy=selected_strategy,
            report_scope="none",
        )

    if frame.artifact_kind in {"weekly_report", "monthly_report"} and frame.action_intent == "send":
        report_scope = _report_scope(frame, selected_strategy, canonical_context)
        if report_scope == "ambiguous":
            return _plan(
                frame,
                response_mode="clarification",
                capabilities=_capabilities(frame.requested_capabilities, frame.artifact_kind),
                adapter_resolves=[],
                action_intents=[],
                selected_strategy=selected_strategy,
                ambiguity_slots=["strategy"],
            )
        return _action_plan(
            frame,
            capability_name=frame.artifact_kind,
            selected_strategy=selected_strategy if report_scope == "strategy" else None,
            report_scope=report_scope,
        )

    if frame.artifact_kind == "knowledge_answer" and frame.action_intent == "answer":
        if "document_context" not in policy.allowed_capabilities:
            return _plan(
                frame,
                response_mode="unable",
                capabilities=[],
                adapter_resolves=[],
                action_intents=[],
                selected_strategy=selected_strategy,
            )
        return _plan(
            frame,
            response_mode="knowledge_answer",
            capabilities=["document_context"],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    if frame.artifact_kind == "human_support" or frame.action_intent == "handoff":
        return _plan(
            frame,
            response_mode="handoff",
            capabilities=["sales_mention"],
            adapter_resolves=_adapter_resolves("sales_mention", None),
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    if frame.artifact_kind == "smalltalk" and frame.action_intent == "none":
        return _plan(
            frame,
            response_mode="no_reply",
            capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    if frame.action_intent == "refuse" or frame.artifact_kind == "refusal":
        return _plan(
            frame,
            response_mode="refusal",
            capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
        )

    return _plan(
        frame,
        response_mode="unable",
        capabilities=[],
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

    if plan.response_mode == "knowledge_answer" and "document_context" not in plan.capabilities:
        issues.append(
            PlanValidationIssue(
                code="knowledge_answer_missing_capability",
                message="knowledge_answer mode requires document_context capability",
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


def _action_plan(
    frame: IntentFrame,
    *,
    capability_name: CapabilityName,
    selected_strategy: str | None,
    report_scope: ActionReportScope,
) -> ExecutionPlan:
    capability = capability_by_name(capability_name)
    action_type = capability.side_effect_action_type if capability is not None else None
    action_intents = []
    if action_type is not None:
        action_intents.append(
            ActionIntentSpec(
                action_type=action_type,
                capability=capability_name,
                report_scope=report_scope,
                strategy=selected_strategy,
            )
        )
    return _plan(
        frame,
        response_mode="action",
        capabilities=_capabilities(frame.requested_capabilities, capability_name),
        adapter_resolves=(
            _adapter_resolves(capability_name, selected_strategy)
            + _adapter_resolves("sales_mention", None)
        ),
        action_intents=action_intents,
        selected_strategy=selected_strategy,
    )


def _plan(
    frame: IntentFrame,
    *,
    response_mode: ResponseMode,
    capabilities: list[CapabilityName],
    adapter_resolves: list[AdapterResolveSpec],
    action_intents: list[ActionIntentSpec],
    selected_strategy: str | None,
    ambiguity_slots: list[str] | None = None,
) -> ExecutionPlan:
    return ExecutionPlan(
        user_need=frame.user_need,
        artifact_kind=frame.artifact_kind,
        response_mode=response_mode,
        compliance=frame.compliance,
        capabilities=_capabilities(capabilities),
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
