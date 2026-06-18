from __future__ import annotations

import hashlib
import json

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.plan_spec import (
    AnswerabilityPolicy,
    PlanDomainScope,
    PlanSpec,
    PlanStep,
    PlanTimeRange,
)
from market_support_crewai_agent.runtime.domain.planning.models import ExecutionPlan


def plan_spec_for_execution_plan(
    plan: ExecutionPlan,
    *,
    domain_context: DomainContext | None = None,
) -> PlanSpec:
    if plan.plan_spec is not None:
        return plan.plan_spec
    manifest_id = _manifest_id_for_execution_plan(plan)
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(manifest_id)
    answerability = _answerability_for_execution_plan(plan)
    domain_scope = _domain_scope_for_execution_plan(plan, domain_context)
    payload = {
        "selected_capability_id": manifest_id,
        "user_intent_summary": plan.user_need,
        "domain_scope": domain_scope.model_dump(mode="json", exclude_none=True),
        "answerability_policy": answerability,
        "response_mode": plan.response_mode,
        "artifact_kind": plan.artifact_kind,
        "capabilities": list(plan.capabilities),
        "answer_capabilities": list(plan.answer_capabilities),
        "adapter_resolves": [
            item.model_dump(mode="json", exclude_none=True)
            for item in plan.adapter_resolves
        ],
        "action_intents": [
            item.model_dump(mode="json", exclude_none=True)
            for item in plan.action_intents
        ],
        "evidence_query": plan.evidence_query,
    }
    plan_id = "plan:" + hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    step = PlanStep(
        step_id="step-1",
        description=_step_description_for_plan(plan),
        uses_artifacts=list(manifest.required_artifacts),
        required_artifacts=list(manifest.required_artifacts),
        allowed_artifacts=list(manifest.allowed_artifacts),
        forbidden_artifacts=list(manifest.forbidden_artifacts),
        required_tools=list(manifest.required_tools),
        evidence_query=plan.evidence_query,
    )
    return PlanSpec(
        plan_id=plan_id,
        selected_capability_id=manifest_id,
        user_intent_summary=plan.user_need,
        domain_scope=domain_scope,
        required_artifacts=list(manifest.required_artifacts),
        allowed_artifacts=list(manifest.allowed_artifacts),
        forbidden_artifacts=list(manifest.forbidden_artifacts),
        required_tools=list(manifest.required_tools),
        answerability_policy=answerability,
        output_schema_ref=f"{manifest.id}:output_schema",
        evidence_contract_ref=f"{manifest.id}:evidence_contract",
        evidence_contract=manifest.evidence_contract,
        steps=[step],
        acceptance_criteria=[
            "output conforms to selected capability output schema",
            "claims are grounded by evidence satisfying EvidenceContract",
        ],
        abstention_cases=[manifest.abstention_policy.guidance]
        if manifest.abstention_policy.guidance
        else [],
        risk_flags=[
            decision.reason_code
            for decision in plan.guardrail_decisions
            if decision.reason_code
        ],
    )


def _manifest_id_for_execution_plan(plan: ExecutionPlan) -> str:
    if plan.response_mode == "action" and plan.action_intents:
        action_type = plan.action_intents[0].action_type
        if action_type == "send_material_pack":
            return "material_pack.send"
        if action_type == "send_weekly_report":
            return "weekly_report.send"
        if action_type == "send_monthly_report":
            return "monthly_report.send"
    if plan.response_mode == "handoff":
        return "sales.handoff"
    if plan.response_mode == "refusal":
        return "general.refusal"
    if plan.response_mode == "clarification":
        return "general.clarification"
    if plan.response_mode == "smalltalk":
        return "general.smalltalk"
    if plan.response_mode == "no_reply":
        return "general.no_reply"
    if plan.response_mode == "unable":
        return "general.abstention"
    for capability_name in plan.answer_capabilities:
        if capability_name == "material_pack":
            return "material_pack.product_list"
        if capability_name == "weekly_report":
            return "weekly_report.product_performance"
        if capability_name == "monthly_report":
            return "monthly_report.product_performance"
        if capability_name == "document_context":
            return "channel.strategy_summary"
    return "general.abstention"


def _answerability_for_execution_plan(plan: ExecutionPlan) -> AnswerabilityPolicy:
    if plan.response_mode == "action":
        return "send"
    if plan.response_mode == "knowledge_answer":
        return "answer"
    if plan.response_mode == "clarification":
        return "clarify"
    if plan.response_mode == "refusal":
        return "refuse"
    if plan.response_mode == "handoff":
        return "handoff"
    if plan.response_mode == "smalltalk":
        return "smalltalk"
    if plan.response_mode == "no_reply":
        return "no_reply"
    return "abstain"


def _domain_scope_for_execution_plan(
    plan: ExecutionPlan,
    domain_context: DomainContext | None,
) -> PlanDomainScope:
    if domain_context is None:
        return PlanDomainScope(
            channel_id="unknown",
            channel_kind="unknown",
            material_pack_option=plan.material_pack_option,
        )
    requested_time_range = None
    if plan.requested_scope is not None and (
        plan.requested_scope.period
        or plan.requested_scope.time_range_start
        or plan.requested_scope.time_range_end
    ):
        requested_time_range = PlanTimeRange(
            period=plan.requested_scope.period,
            start=plan.requested_scope.time_range_start,
            end=plan.requested_scope.time_range_end,
        )
    return PlanDomainScope(
        channel_id=domain_context.channel.id,
        channel_kind=domain_context.channel.kind,
        material_pack_option=plan.material_pack_option,
        product_ids=[],
        time_range=requested_time_range,
    )


def _step_description_for_plan(plan: ExecutionPlan) -> str:
    if plan.response_mode == "action":
        return "Validate adapter evidence and emit typed outbound action proposal."
    if plan.response_mode == "knowledge_answer":
        return "Compose answer from evidence satisfying the selected capability contract."
    if plan.response_mode == "clarification":
        return "Ask for the missing scope needed before execution."
    if plan.response_mode == "handoff":
        return "Use adapter-resolved mention target for human handoff."
    if plan.response_mode == "refusal":
        return "Apply compliance refusal policy."
    return "Render bounded non-action reply."
