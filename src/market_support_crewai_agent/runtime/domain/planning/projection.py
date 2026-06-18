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
    manifest_ids = _manifest_ids_for_execution_plan(plan)
    domain_scope = _domain_scope_for_execution_plan(plan, domain_context)
    payload = {
        "selected_capability_ids": manifest_ids,
        "user_intent_summary": plan.user_need,
        "domain_scope": domain_scope.model_dump(mode="json", exclude_none=True),
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
    return PlanSpec(
        plan_id=plan_id,
        user_intent_summary=plan.user_need,
        plan_units=[
            _plan_unit_for_manifest(
                index,
                manifest_id,
                plan=plan,
                domain_scope=domain_scope,
            )
            for index, manifest_id in enumerate(manifest_ids, start=1)
        ],
        risk_flags=[
            decision.reason_code
            for decision in plan.guardrail_decisions
            if decision.reason_code
        ],
    )


def _plan_unit_for_manifest(
    index: int,
    manifest_id: str,
    *,
    plan: ExecutionPlan,
    domain_scope: PlanDomainScope,
):
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(manifest_id)
    answerability = _answerability_for_manifest(plan, manifest_id)
    step = PlanStep(
        step_id=f"step-{index}",
        description=_step_description_for_plan(plan),
        uses_artifacts=list(manifest.required_artifacts),
        required_artifacts=list(manifest.required_artifacts),
        allowed_artifacts=list(manifest.allowed_artifacts),
        forbidden_artifacts=list(manifest.forbidden_artifacts),
        required_tools=list(manifest.required_tools),
        evidence_query=plan.evidence_query,
    )
    return {
        "unit_id": f"unit-{index}",
        "selected_capability_id": manifest_id,
        "domain_scope": domain_scope.model_dump(mode="json", exclude_none=True),
        "required_artifacts": list(manifest.required_artifacts),
        "allowed_artifacts": list(manifest.allowed_artifacts),
        "forbidden_artifacts": list(manifest.forbidden_artifacts),
        "required_tools": list(manifest.required_tools),
        "answerability_policy": answerability,
        "output_schema_ref": f"{manifest.id}:output_schema",
        "evidence_contract_ref": f"{manifest.id}:evidence_contract",
        "evidence_contract": manifest.evidence_contract,
        "steps": [step.model_dump(mode="json", exclude_none=True)],
        "acceptance_criteria": [
            "output conforms to selected capability output schema",
            "claims are grounded by evidence satisfying EvidenceContract",
        ],
        "abstention_cases": [manifest.abstention_policy.guidance]
        if manifest.abstention_policy.guidance
        else [],
    }


def _manifest_ids_for_execution_plan(plan: ExecutionPlan) -> list[str]:
    manifest_ids: list[str] = []
    if plan.response_mode == "action":
        for action in plan.action_intents:
            action_type = action.action_type
            if action_type == "send_material_pack":
                manifest_ids.append("material_pack.send")
            elif action_type == "send_weekly_report":
                manifest_ids.append("weekly_report.send")
            elif action_type == "send_monthly_report":
                manifest_ids.append("monthly_report.send")
    for capability_name in plan.answer_capabilities:
        if capability_name == "material_pack":
            manifest_ids.append("material_pack.product_list")
        elif capability_name == "weekly_report":
            manifest_ids.append("weekly_report.product_performance")
        elif capability_name == "monthly_report":
            manifest_ids.append("monthly_report.product_performance")
        elif capability_name == "document_context":
            manifest_ids.append("channel.strategy_summary")
    if manifest_ids:
        return _unique_strings(manifest_ids)
    if plan.response_mode == "action" and plan.action_intents:
        action_type = plan.action_intents[0].action_type
        if action_type == "send_material_pack":
            return ["material_pack.send"]
        if action_type == "send_weekly_report":
            return ["weekly_report.send"]
        if action_type == "send_monthly_report":
            return ["monthly_report.send"]
    if plan.response_mode == "handoff":
        return ["sales.handoff"]
    if plan.response_mode == "refusal":
        return ["general.refusal"]
    if plan.response_mode == "clarification":
        return ["general.clarification"]
    if plan.response_mode == "smalltalk":
        return ["general.smalltalk"]
    if plan.response_mode == "no_reply":
        return ["general.no_reply"]
    if plan.response_mode == "unable":
        return ["general.abstention"]
    return ["general.abstention"]


def _answerability_for_manifest(
    plan: ExecutionPlan,
    manifest_id: str,
) -> AnswerabilityPolicy:
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(manifest_id)
    if plan.response_mode == "action" and manifest.capability_type == "action":
        return "send"
    if manifest.capability_type in {"answer", "summary"}:
        return "answer"
    return _answerability_for_execution_plan(plan)


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


def _unique_strings(values: list[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
