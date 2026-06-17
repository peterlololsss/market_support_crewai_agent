from __future__ import annotations

from typing import get_args

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ArtifactKind,
    CapabilityName,
    ResponseMode,
    capability_by_name,
)
from market_support_crewai_agent.runtime.domain.compliance_policy import (
    ComplianceReasonCode,
)
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.plan_spec import (
    AnswerabilityPolicy,
    PlanSpec,
)
from market_support_crewai_agent.runtime.domain.planning.models import (
    ActionIntentSpec,
    ActionReportScope,
    AdapterResolveSpec,
    ComplianceDecision,
    ExecutionPlan,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.schemas import ReplyRequest


def compile_plan_spec(
    spec: PlanSpec,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    policy: PolicyManifest,
    domain_context: DomainContext | None = None,
) -> ExecutionPlan:
    del request, canonical_context, policy
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(spec.selected_capability_id)
    selected_strategy = spec.domain_scope.strategy_name or spec.domain_scope.strategy_id
    compliance_reason_code = _compliance_reason_code_for_plan_spec(spec)
    compliance = ComplianceDecision(
        is_compliant=False if spec.answerability_policy == "refuse" else True,
        reason_code=compliance_reason_code,
        reason="PlanSpec answerability policy",
    )

    if manifest is None:
        return ExecutionPlan(
            user_need=spec.user_intent_summary,
            artifact_kind="unclear",
            response_mode="unable",
            compliance=compliance,
            capabilities=[],
            answer_capabilities=[],
            adapter_resolves=[],
            action_intents=[],
            selected_strategy=selected_strategy,
            plan_spec=spec,
        )

    capability_name = str(manifest.runtime_capability or "")
    capability = capability_by_name(capability_name)
    answerability = spec.answerability_policy
    capabilities: list[CapabilityName] = []
    answer_capabilities: list[CapabilityName] = []
    adapter_resolves: list[AdapterResolveSpec] = []
    action_intents: list[ActionIntentSpec] = []
    response_mode: ResponseMode = _response_mode_for_plan_spec(answerability)
    artifact_kind: ArtifactKind = _artifact_kind_for_manifest(manifest, response_mode)

    if capability is not None:
        capabilities.append(capability.name)
        if answerability == "answer":
            answer_capabilities.append(capability.name)
        if answerability in {"answer", "send", "handoff"}:
            adapter_resolves.extend(
                _adapter_resolves_from_plan_spec(
                    spec,
                    capability.name,
                    selected_strategy,
                )
            )
        if answerability == "send" and capability.side_effect_action_type is not None:
            report_scope = _report_scope_from_plan_spec(capability.name, selected_strategy)
            action_intents.append(
                ActionIntentSpec(
                    action_type=capability.side_effect_action_type,
                    capability=capability.name,
                    report_scope=report_scope,
                    strategy=_action_strategy(
                        capability.name,
                        selected_strategy,
                        report_scope,
                    ),
                )
            )
            adapter_resolves.extend(_adapter_resolves("sales_mention", None))
        elif answerability == "handoff":
            adapter_resolves.extend(_adapter_resolves("sales_mention", None))

    if response_mode == "clarification":
        capabilities = []
        answer_capabilities = []
        adapter_resolves = []
        action_intents = []

    if domain_context is not None and selected_strategy:
        resolved = domain_context.strategy_by_name(selected_strategy)
        if resolved is not None:
            selected_strategy = resolved.name

    return ExecutionPlan(
        user_need=spec.user_intent_summary,
        artifact_kind=artifact_kind,
        response_mode=response_mode,
        compliance=compliance,
        evidence_query=_evidence_query_from_plan_spec(spec),
        capabilities=_unique_capabilities(capabilities),
        answer_capabilities=_unique_capabilities(answer_capabilities),
        adapter_resolves=_unique_adapter_resolves(adapter_resolves),
        action_intents=action_intents,
        selected_strategy=selected_strategy,
        plan_spec=spec,
    )


def _response_mode_for_plan_spec(answerability: AnswerabilityPolicy) -> ResponseMode:
    if answerability == "send":
        return "action"
    if answerability == "answer":
        return "knowledge_answer"
    if answerability == "clarify":
        return "clarification"
    if answerability == "refuse":
        return "refusal"
    if answerability == "handoff":
        return "handoff"
    if answerability == "smalltalk":
        return "smalltalk"
    if answerability == "no_reply":
        return "no_reply"
    return "unable"


def _compliance_reason_code_for_plan_spec(spec: PlanSpec) -> ComplianceReasonCode:
    if spec.answerability_policy != "refuse":
        return "compliant_product_request"
    allowed_codes = set(get_args(ComplianceReasonCode))
    for flag in spec.risk_flags:
        if flag in allowed_codes:
            return flag  # type: ignore[return-value]
    return "unknown"


def _artifact_kind_for_manifest(manifest, response_mode: ResponseMode) -> ArtifactKind:
    capability = capability_by_name(str(manifest.runtime_capability or ""))
    if capability is not None and response_mode == "action":
        return capability.artifact_kind
    if response_mode == "knowledge_answer":
        return "knowledge_answer"
    if response_mode == "handoff":
        return "human_support"
    if response_mode == "refusal":
        return "refusal"
    if response_mode == "smalltalk":
        return "smalltalk"
    if response_mode == "no_reply":
        return "smalltalk"
    return "unclear"


def _adapter_resolves_from_plan_spec(
    spec: PlanSpec,
    capability_name: CapabilityName,
    selected_strategy: str | None,
) -> list[AdapterResolveSpec]:
    tools = list(spec.required_tools)
    if not tools:
        manifest = CAPABILITY_MANIFEST_REGISTRY.find(spec.selected_capability_id)
        tools = list(manifest.required_tools if manifest is not None else ())
    resolves: list[AdapterResolveSpec] = []
    for tool in tools:
        if not tool.startswith("adapter_resolve."):
            continue
        resolve_type = tool.removeprefix("adapter_resolve.")
        capability = capability_by_name(capability_name)
        strategy = (
            selected_strategy
            if capability is not None
            and (
                capability.is_report
                or capability.requires_strategy_for_bank_material
            )
            else None
        )
        resolves.append(
            AdapterResolveSpec(
                resolve_type=resolve_type,  # type: ignore[arg-type]
                strategy=strategy,
            )
        )
    if not resolves:
        resolves.extend(_adapter_resolves(capability_name, selected_strategy))
    return resolves


def _report_scope_from_plan_spec(
    capability_name: CapabilityName,
    selected_strategy: str | None,
) -> ActionReportScope:
    capability = capability_by_name(capability_name)
    if capability is None or not capability.is_report:
        return "none"
    return "strategy" if selected_strategy else "channel_all"


def _evidence_query_from_plan_spec(spec: PlanSpec) -> str | None:
    for step in spec.steps:
        if step.evidence_query:
            return step.evidence_query
    return None


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


def _unique_capabilities(
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
