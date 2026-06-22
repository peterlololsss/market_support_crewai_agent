from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.domain.canonicalization import canonicalize_request
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    compile_plan_spec,
)
from market_support_crewai_agent.runtime.domain.policy import (
    PolicyManifest,
    compile_policy,
)
from market_support_crewai_agent.schemas import ReplyRequest


def make_plan_spec(
    request: ReplyRequest | None = None,
    *,
    selected_capability_id: str | None = None,
    answerability_policy: str | None = None,
    user_intent_summary: str | None = None,
    evidence_query: str | None = None,
    material_pack_option: str | None = None,
    plan_units: list[dict] | None = None,
    **intent_like,
) -> PlanSpec:
    request = request or make_request()
    base_payload = {
        **intent_like,
        "selected_capability_id": selected_capability_id,
        "answerability_policy": answerability_policy,
        "user_intent_summary": user_intent_summary,
        "evidence_query": evidence_query,
        "material_pack_option": material_pack_option,
    }
    unit_payloads = plan_units or [base_payload]
    units = [
        _plan_unit_payload(
            request,
            index=index,
            payload={**base_payload, **unit_payload},
        )
        for index, unit_payload in enumerate(unit_payloads, start=1)
    ]
    summary = (
        user_intent_summary
        or intent_like.get("user_need")
        or units[0]["steps"][0]["description"]
    )
    risk_flags = list(intent_like.get("risk_flags") or [])
    risk_flags.extend(intent_like.get("ambiguity_slots") or [])
    compliance = intent_like.get("compliance")
    if isinstance(compliance, dict) and compliance.get("reason_code"):
        risk_flags.append(compliance["reason_code"])
    return PlanSpec.model_validate(
        {
            "plan_id": intent_like.get("plan_id", "plan-test"),
            "user_intent_summary": summary,
            "plan_units": units,
            "risk_flags": risk_flags,
        }
    )


def _plan_unit_payload(
    request: ReplyRequest,
    *,
    index: int,
    payload: dict,
) -> dict:
    capability_id = payload.get("selected_capability_id") or _capability_id_from_payload(
        payload
    )
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(capability_id)
    answerability = payload.get("answerability_policy") or _answerability_from_capability_id(
        capability_id,
        payload,
    )
    selected_option = _material_pack_option_for_scope(
        manifest.runtime_capability,
        payload.get("material_pack_option"),
        payload,
    )
    step = {
        "step_id": f"step-{index}",
        "description": payload.get("user_intent_summary")
        or payload.get("user_need")
        or "handle current support request",
        "uses_artifacts": list(manifest.required_artifacts),
        "required_artifacts": list(manifest.required_artifacts),
        "allowed_artifacts": list(manifest.allowed_artifacts),
        "forbidden_artifacts": list(manifest.forbidden_artifacts),
        "required_tools": list(manifest.required_tools),
        "evidence_query": payload.get("evidence_query"),
    }
    risk_flags = list(payload.get("risk_flags") or [])
    risk_flags.extend(payload.get("ambiguity_slots") or [])
    compliance = payload.get("compliance")
    if isinstance(compliance, dict) and compliance.get("reason_code"):
        risk_flags.append(compliance["reason_code"])
    return {
        "unit_id": payload.get("unit_id") or f"unit-{index}",
        "selected_capability_id": capability_id,
        "domain_scope": {
            "channel_id": request.group_id or request.conversation_key,
            "channel_kind": request.channel_type,
            "material_pack_option": selected_option,
            "product_ids": [],
        },
        "required_artifacts": list(manifest.required_artifacts),
        "allowed_artifacts": list(manifest.allowed_artifacts),
        "forbidden_artifacts": list(manifest.forbidden_artifacts),
        "required_tools": list(manifest.required_tools),
        "answerability_policy": answerability,
        "output_schema_ref": f"{manifest.id}:output_schema",
        "evidence_contract_ref": f"{manifest.id}:evidence_contract",
        "evidence_contract": manifest.evidence_contract,
        "steps": [step],
        "acceptance_criteria": ["satisfy selected capability contract"],
        "abstention_cases": [manifest.abstention_policy.guidance]
        if manifest.abstention_policy.guidance
        else [],
        "risk_flags": risk_flags,
    }


def compile_test_plan(
    request: ReplyRequest | None = None,
    *,
    policy: PolicyManifest | None = None,
    domain_context: DomainContext | None = None,
    doc_mcp_enabled: bool = False,
    **intent_like,
) -> ExecutionPlan:
    request = request or make_request()
    policy = policy or compile_policy(request, doc_mcp_enabled=doc_mcp_enabled)
    return compile_plan_spec(
        make_plan_spec(request, **intent_like),
        request,
        canonicalize_request(request),
        policy,
        domain_context=domain_context,
    )


def make_request(**overrides) -> ReplyRequest:
    payload = {
        "context_id": "msg-1",
        "conversation_key": "wecom:group-1:sender-1",
        "group_id": "group-1",
        "sender_id": "sender-1",
        "message": "hello",
        "is_group": True,
        "group_name": "test group",
        "dist_channel_name": "test channel",
        "sender_nickname": "test user",
        "available_artifacts": [
            {"type": "material_pack", "options": ["指增"]},
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def _capability_id_from_payload(payload: dict) -> str:
    compliance = payload.get("compliance")
    if isinstance(compliance, dict) and compliance.get("is_compliant") is False:
        return "general.refusal"
    artifact_kind = payload.get("artifact_kind", "unclear")
    action_intent = payload.get("action_intent", "none")
    requested = list(payload.get("requested_capabilities") or [])
    if payload.get("ambiguity_slots"):
        return "general.clarification"
    if action_intent == "send":
        if artifact_kind == "material_pack":
            return "material_pack.send"
        if artifact_kind == "weekly_report":
            return "weekly_report.send"
        if artifact_kind == "monthly_report":
            return "monthly_report.send"
    if action_intent == "handoff" or artifact_kind == "human_support":
        return "sales.handoff"
    if action_intent == "refuse" or artifact_kind == "refusal":
        return "general.refusal"
    if action_intent == "answer" or artifact_kind == "knowledge_answer":
        if "document_context" in requested:
            return "channel.strategy_summary"
        if "weekly_report" in requested:
            return "weekly_report.product_performance"
        if "monthly_report" in requested:
            return "monthly_report.product_performance"
        if "material_pack" in requested:
            return "material_pack.product_list"
    if artifact_kind == "smalltalk":
        return "general.smalltalk"
    return "general.abstention"


def _answerability_from_capability_id(capability_id: str, payload: dict) -> str:
    if capability_id.endswith(".send"):
        return "send"
    if capability_id == "sales.handoff":
        return "handoff"
    if capability_id == "general.clarification":
        return "clarify"
    if capability_id == "general.refusal":
        return "refuse"
    if capability_id == "general.smalltalk":
        return "smalltalk"
    if capability_id == "general.no_reply":
        return "no_reply"
    if capability_id == "general.abstention":
        return "abstain"
    if payload.get("action_intent") == "send":
        return "send"
    return "answer"


def _material_pack_option_for_scope(
    runtime_capability: str | None,
    material_pack_option: str | None,
    payload: dict,
) -> str | None:
    if runtime_capability != "material_pack":
        return None
    option = material_pack_option or payload.get("material_pack_option")
    return str(option).strip() if option else None
