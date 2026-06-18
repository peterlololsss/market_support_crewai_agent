from __future__ import annotations

from market_support_crewai_agent.runtime.domain.ontology import ArtifactScope
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.schemas import (
    ReplyRequest,
    SendMaterialPackAction,
    SendMonthlyReportAction,
    SendWeeklyReportAction,
)
from tests.helpers.planning import compile_test_plan


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
        "available_materials": ["material", "weekly", "monthly"],
        "material_pack_options": ["指增"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


def make_plan(request: ReplyRequest | None = None, **overrides):
    request = request or make_request()
    payload = {
        "user_need": "send weekly report",
        "artifact_kind": "weekly_report",
        "action_intent": "send",
        "compliance": {
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal report request",
        },
        "confidence": 0.8,
    }
    payload.update(overrides)
    return compile_test_plan(request, doc_mcp_enabled=True, **payload)


def weekly_action(**overrides) -> SendWeeklyReportAction:
    payload = {
        "type": "send_weekly_report",
        "action_id": "act-1",
        "resolve_type": "weekly_report",
        "resolve_ref": "weekly:ref",
        "period": "20260529",
        "report_date": "2026-05-29",
    }
    payload.update(overrides)
    return SendWeeklyReportAction.model_validate(payload)


def monthly_action(**overrides) -> SendMonthlyReportAction:
    payload = {
        "type": "send_monthly_report",
        "action_id": "act-1",
        "resolve_type": "monthly_report",
        "resolve_ref": "monthly:ref",
        "period": "202605",
        "report_date": "2026-05-31",
    }
    payload.update(overrides)
    return SendMonthlyReportAction.model_validate(payload)


def material_action(**overrides) -> SendMaterialPackAction:
    payload = {
        "type": "send_material_pack",
        "action_id": "act-1",
        "resolve_type": "material_pack",
        "resolve_ref": "material:ref",
    }
    payload.update(overrides)
    return SendMaterialPackAction.model_validate(payload)


def make_directive(plan, **overrides) -> ResponseDirective:
    payload = {
        "mode": plan.response_mode,
        "reply_kind": "answer" if plan.response_mode == "action" else "unable_to_answer",
        "text": "",
        "action_intents": plan.action_intents if plan.response_mode == "action" else [],
    }
    payload.update(overrides)
    return ResponseDirective.model_validate(payload)


def resolved_fact(resolve_type: str, resolve_ref: str, **metadata) -> EvidenceFact:
    fact_type = {
        "material_pack": "material_pack_resolvable",
        "weekly_report": "weekly_report_resolvable",
        "monthly_report": "monthly_report_resolvable",
        "sales_mention": "sales_mention_resolvable",
    }[resolve_type]
    payload = {"status": "resolved", "resolve_ref": resolve_ref}
    payload.update(metadata)
    return EvidenceFact(
        fact_type=fact_type,
        value=True,
        resolve_type=resolve_type,
        metadata=payload,
    )


def material_product_fact(
    *products: str,
    material_pack_option: str | None = "指增",
    scope: ArtifactScope | None = None,
    source_type: str = "adapter_material_pack_content",
    artifact_type: str = "material_pack",
) -> EvidenceFact:
    metadata = {
        "status": "resolved",
        "products": [{"product_name": product} for product in products],
    }
    if material_pack_option:
        metadata["material_pack_option"] = material_pack_option
    return EvidenceFact(
        fact_type="material_pack_product_list",
        value=True,
        source_type=source_type,  # type: ignore[arg-type]
        source_id="material_pack",
        resolve_type="material_pack",
        metadata=metadata,
        artifact_type=artifact_type,  # type: ignore[arg-type]
        scope=scope or ArtifactScope(channel_id="unknown"),
    )


def material_answer_plan(request: ReplyRequest | None = None):
    request = request or make_request(
        message="材料包里有哪些产品",
        material_pack_options=["指增"],
    )
    return make_plan(
        request,
        user_need="answer material pack product list",
        artifact_kind="knowledge_answer",
        action_intent="answer",
        requested_capabilities=["material_pack"],
        compliance={
            "is_compliant": True,
            "reason_code": "compliant_product_request",
            "reason": "normal material question",
        },
    )
