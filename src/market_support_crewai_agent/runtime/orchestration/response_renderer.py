from __future__ import annotations

from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.capabilities import (
    capability_by_resolve_type,
    resolve_type_for_action,
)
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import ActionIntentSpec, ExecutionPlan
from market_support_crewai_agent.schemas import (
    OutboundAction,
    PrimaryReply,
    ReplyResponse,
    SendMaterialPackAction,
    SendMonthlyReportAction,
    SendWeeklyReportAction,
)


def render_directive(
    directive: ResponseDirective,
    plan: ExecutionPlan,
    business_facts: BusinessFacts,
    evidence_facts: list[EvidenceFact],
) -> ReplyResponse:
    del plan, evidence_facts
    if directive.requires_knowledge_composer:
        raise ValueError("knowledge composer directives must be rendered by composer LLM")

    return ReplyResponse(
        reply=PrimaryReply(
            kind=directive.reply_kind,
            text=directive.text,
            mentions=directive.mentions,
        ),
        actions=(
            [
                _render_action(action_intent, business_facts)
                for action_intent in directive.action_intents
            ]
            if directive.mode == "action"
            else []
        ),
    )


def _render_action(
    action_intent: ActionIntentSpec,
    business_facts: BusinessFacts,
) -> OutboundAction:
    resolve_type = resolve_type_for_action(action_intent.action_type)
    if resolve_type is None:
        raise ValueError(f"unsupported action intent: {action_intent.action_type}")

    resolve_state = business_facts.resolve_state(resolve_type)
    resolve_ref = resolve_state.resolve_ref or ""
    material_pack_option = (
        action_intent.material_pack_option or resolve_state.material_pack_option
    )
    capability = capability_by_resolve_type(resolve_type)

    if action_intent.action_type == "send_material_pack":
        return SendMaterialPackAction(
            type="send_material_pack",
            resolve_type="material_pack",
            resolve_ref=resolve_ref,
            material_pack_option=material_pack_option,
        )

    if capability is not None and capability.is_report:
        report_state = business_facts.report_state(resolve_type)
        period = report_state.period if report_state is not None else None
        report_date = report_state.report_date if report_state is not None else None
        if action_intent.action_type == "send_weekly_report":
            return SendWeeklyReportAction(
                type="send_weekly_report",
                resolve_type="weekly_report",
                resolve_ref=resolve_ref,
                period=period,
                report_date=report_date,
            )
        if action_intent.action_type == "send_monthly_report":
            return SendMonthlyReportAction(
                type="send_monthly_report",
                resolve_type="monthly_report",
                resolve_ref=resolve_ref,
                period=period,
                report_date=report_date,
            )

    raise ValueError(f"unsupported action intent: {action_intent.action_type}")
