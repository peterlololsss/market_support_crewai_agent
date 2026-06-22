from __future__ import annotations

import calendar
import datetime as dt
from typing import Literal

from pydantic import Field

from market_support_crewai_agent.runtime.domain.business_facts import (
    BusinessFacts,
    ReportState,
    ResolvableState,
)
from market_support_crewai_agent.runtime.domain.capabilities import (
    ResponseMode,
    capability_by_name,
    capability_by_resolve_type,
    resolve_type_for_action,
)
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.compliance_policy import refusal_text_for_reason
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import ActionIntentSpec, ExecutionPlan
from market_support_crewai_agent.runtime.domain.planning.clarification import (
    clarification_spec,
)
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.domain.sources.precedence import (
    evidence_facts_for_plan,
    plan_has_knowledge_evidence,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    abstention_response_text,
)
from market_support_crewai_agent.schemas import (
    ReplyKind,
    ReplyMention,
    ReplyRequest,
    StrictModel,
)


class ResponseDirective(StrictModel):
    contract_version: Literal["response-directive"] = "response-directive"
    mode: ResponseMode
    reply_kind: ReplyKind
    text: str = ""
    mentions: list[ReplyMention] = Field(default_factory=list)
    action_intents: list[ActionIntentSpec] = Field(default_factory=list)
    requires_knowledge_composer: bool = False
    composer_stage: Literal["knowledge_composer", "smalltalk_composer"] | None = None
    reason_code: str = ""


class DecisionEngine:
    def decide(
        self,
        plan: ExecutionPlan,
        business_facts: BusinessFacts,
        evidence_facts: list[EvidenceFact],
        request: ReplyRequest,
        policy: PolicyManifest,
        domain_context: DomainContext | None = None,
    ) -> ResponseDirective:
        del policy
        if plan.response_mode == "refusal" or plan.compliance.is_compliant is False:
            return _directive(
                mode="refusal",
                reply_kind="unable_to_answer",
                text=refusal_text_for_reason(plan.compliance.reason_code),
                reason_code=plan.compliance.reason_code,
            )

        if plan.response_mode == "clarification" or plan.ambiguity_slots:
            return _directive(
                mode="clarification",
                reply_kind="clarification",
                text=_clarification_text(plan, business_facts, request),
                requires_knowledge_composer=True,
                composer_stage="knowledge_composer",
                reason_code="ambiguous_request",
            )

        if plan.response_mode == "handoff":
            return _handoff_or_unable(
                business_facts,
                text="这个问题我帮你请销售/支持同事确认。",
                reason="intent requires human handoff",
                reason_code="handoff_requested",
            )

        if plan.response_mode == "knowledge_answer":
            report_period_text = _report_period_answer(plan, business_facts)
            if report_period_text:
                return _directive(
                    mode="knowledge_answer",
                    reply_kind="answer",
                    text=report_period_text,
                    reason_code="report_period_metadata_available",
                )
            material_products_text = _material_pack_product_answer(
                plan,
                evidence_facts,
                domain_context,
            )
            if material_products_text:
                return _directive(
                    mode="knowledge_answer",
                    reply_kind="answer",
                    text=material_products_text,
                    reason_code="material_pack_product_list_available",
                )
            if _has_knowledge_answer_evidence(plan, evidence_facts, domain_context):
                return _directive(
                    mode="knowledge_answer",
                    reply_kind="answer",
                    requires_knowledge_composer=True,
                    composer_stage="knowledge_composer",
                    reason_code="knowledge_evidence_available",
                )
            return _directive(
                mode="unable",
                reply_kind="unable_to_answer",
                text="老师，这个信息我这边暂时无法确认，先不展开避免信息不准确。",
                reason_code="document_context_missing",
            )

        if plan.response_mode == "smalltalk":
            return _directive(
                mode="smalltalk",
                reply_kind="answer",
                requires_knowledge_composer=True,
                composer_stage="smalltalk_composer",
                reason_code="smalltalk_requires_composer",
            )

        if plan.response_mode == "no_reply":
            return _directive(
                mode="no_reply",
                reply_kind="no_reply",
                reason_code="no_reply",
            )

        if plan.response_mode == "unable":
            scope_conflict_text = _send_scope_conflict_text(plan, request)
            if scope_conflict_text:
                return _directive(
                    mode="unable",
                    reply_kind="unable_to_answer",
                    text=scope_conflict_text,
                    reason_code="send_scope_conflict",
                )

        if plan.response_mode == "action":
            return _action_directive(
                plan,
                business_facts,
                evidence_facts,
                request,
                domain_context,
            )

        return _directive(
            mode="unable",
            reply_kind="unable_to_answer",
            text="老师，这个信息我这边暂时无法确认，先不展开避免信息不准确。",
            reason_code="insufficient_evidence",
        )


def _report_period_answer(
    plan: ExecutionPlan,
    business_facts: BusinessFacts,
) -> str:
    if plan.evidence_query:
        return ""
    answer_resolve_types = _answer_report_resolve_types(plan)
    if plan.answer_capabilities and not answer_resolve_types:
        return ""
    for resolve_type, label in (
        ("weekly_report", "周报"),
        ("monthly_report", "月报"),
    ):
        if answer_resolve_types and resolve_type not in answer_resolve_types:
            continue
        if resolve_type not in {item.resolve_type for item in plan.adapter_resolves}:
            continue
        report_state = business_facts.report_state(resolve_type)  # type: ignore[arg-type]
        if report_state is None or not report_state.period:
            continue
        period_start, period_end = _report_period_range(resolve_type, report_state)
        if period_start and period_end:
            if resolve_type == "weekly_report":
                report_date = report_state.report_date or period_end
                return f"这份{label}覆盖{period_start}至{period_end}（报告日：{report_date}）"
            return f"这份{label}覆盖{period_start}至{period_end}（期数：{report_state.period}）"
        if report_state.period_label:
            return f"这是{report_state.period_label}"
        if report_state.report_date:
            return f"这是{report_state.report_date}的{label}"
    return ""


def _answer_report_resolve_types(plan: ExecutionPlan) -> set[str]:
    return {
        str(capability.resolve_type)
        for capability_name in plan.answer_capabilities
        if (capability := capability_by_name(capability_name)) is not None
        and capability.resolve_type in {"weekly_report", "monthly_report"}
    }


def _report_period_range(resolve_type: str, report_state: ReportState) -> tuple[str, str]:
    if report_state.period_start and report_state.period_end:
        return report_state.period_start, report_state.period_end

    period = str(report_state.period or "").strip()
    if resolve_type == "weekly_report":
        report_date = report_state.report_date or _weekly_report_date(period)
        if not report_date:
            return "", ""
        parsed = dt.datetime.strptime(report_date, "%Y-%m-%d")
        period_start = (parsed - dt.timedelta(days=parsed.weekday())).strftime("%Y-%m-%d")
        return period_start, report_date

    if resolve_type == "monthly_report" and len(period) == 7 and period[4] == "-":
        year = int(period[:4])
        month = int(period[5:])
        return (
            f"{year:04d}-{month:02d}-01",
            f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}",
        )
    return "", ""


def _weekly_report_date(period: str) -> str:
    if len(period) == 8 and period.isdigit():
        return f"{period[:4]}-{period[4:6]}-{period[6:8]}"
    return ""


def _action_directive(
    plan: ExecutionPlan,
    business_facts: BusinessFacts,
    evidence_facts: list[EvidenceFact],
    request: ReplyRequest,
    domain_context: DomainContext | None = None,
) -> ResponseDirective:
    if not plan.action_intents:
        return _directive(
            mode="unable",
            reply_kind="unable_to_answer",
            text="当前请求没有可执行的发送动作。",
            reason_code="missing_action_intent",
        )

    for action_intent in plan.action_intents:
        resolve_type = resolve_type_for_action(action_intent.action_type)
        if resolve_type is None:
            return _directive(
                mode="unable",
                reply_kind="unable_to_answer",
                text="当前请求没有可执行的发送动作。",
                reason_code="unsupported_action_intent",
            )

        resolve_state = business_facts.resolve_state(resolve_type)
        if resolve_state.status == "ambiguous" and resolve_state.candidates:
            return _directive(
                mode="clarification",
                reply_kind="clarification",
                requires_knowledge_composer=True,
                composer_stage="knowledge_composer",
                reason_code="ambiguous_action_resolve",
            )
        if resolve_state.status != "available":
            if plan.answer_capabilities and _has_knowledge_answer_evidence(
                plan,
                evidence_facts,
                domain_context,
            ):
                return _directive(
                    mode="knowledge_answer",
                    reply_kind="answer",
                    requires_knowledge_composer=True,
                    composer_stage="knowledge_composer",
                    reason_code="action_unavailable_with_knowledge_evidence",
                )
            return _handoff_or_unable(
                business_facts,
                text="目前这个渠道下我没有看到可发送的对应内容，我帮你请销售/支持同事确认。",
                reason="requested outbound action is unavailable in adapter evidence",
                reason_code="action_resolve_unavailable",
                unable_text="目前这个渠道下我没有看到可发送的对应内容。",
            )
        if not resolve_state.resolve_ref:
            return _directive(
                mode="unable",
                reply_kind="unable_to_answer",
                text="当前可发送内容缺少 adapter resolve_ref。",
                reason_code="missing_resolve_ref",
            )

    if plan.answer_capabilities:
        report_period_text = _report_period_answer(plan, business_facts)
        if report_period_text:
            return _directive(
                mode="action",
                reply_kind="answer",
                text=report_period_text,
                action_intents=plan.action_intents,
                reason_code="action_ready_with_deterministic_answer",
            )
        material_products_text = _material_pack_product_answer(
            plan,
            evidence_facts,
            domain_context,
        )
        if material_products_text:
            return _directive(
                mode="action",
                reply_kind="answer",
                text=material_products_text,
                action_intents=plan.action_intents,
                reason_code="action_ready_with_material_pack_product_list",
            )
        if _has_knowledge_answer_evidence(plan, evidence_facts, domain_context):
            return _directive(
                mode="action",
                reply_kind="answer",
                action_intents=plan.action_intents,
                requires_knowledge_composer=True,
                composer_stage="knowledge_composer",
                reason_code="action_ready_with_knowledge_evidence",
            )
        return _directive(
            mode="action",
            reply_kind="unable_to_answer",
            text=abstention_response_text(),
            action_intents=plan.action_intents,
            reason_code="action_ready_answer_evidence_missing",
        )

    return _directive(
        mode="action",
        reply_kind="answer",
        text=_action_rationale_text(plan),
        action_intents=plan.action_intents,
        reason_code="action_ready",
    )


def _action_rationale_text(plan: ExecutionPlan) -> str:
    if _weekly_report_rationale_required(plan):
        return "这个问题需要看最新周报里的近期表现数据，我先把周报发你，具体以报告为准。"
    return ""


def _weekly_report_rationale_required(plan: ExecutionPlan) -> bool:
    if not any(intent.action_type == "send_weekly_report" for intent in plan.action_intents):
        return False
    flags = set(plan.plan_spec.risk_flags if plan.plan_spec is not None else [])
    return "weekly_report_rationale_required" in flags


def _handoff_or_unable(
    business_facts: BusinessFacts,
    *,
    text: str,
    reason: str,
    reason_code: str,
    unable_text: str | None = None,
) -> ResponseDirective:
    if business_facts.sales_mention.resolvable:
        return _directive(
            mode="handoff",
            reply_kind="human_handoff",
            text=text,
            mentions=[ReplyMention(type="sales", reason=reason)],
            reason_code=reason_code,
        )
    return _directive(
        mode="unable",
        reply_kind="unable_to_answer",
        text=unable_text or "当前渠道暂未配置可用负责人。",
        reason_code="sales_mention_unavailable",
    )


def _clarification_text(
    plan: ExecutionPlan,
    business_facts: BusinessFacts,
    request: ReplyRequest,
) -> str:
    spec = clarification_spec(plan.ambiguity_slots)
    if spec is None:
        return abstention_response_text()
    candidates = (
        _best_candidates(business_facts, request)
        if spec.slot == "material_pack_option"
        else _artifact_candidates(request)
    )
    if spec.candidate_prefix:
        return _candidate_text(spec.candidate_prefix, candidates)
    return spec.question_text


def _best_candidates(
    business_facts: BusinessFacts,
    request: ReplyRequest,
) -> tuple[str, ...]:
    if business_facts.material_pack.candidates:
        return business_facts.material_pack.candidates
    for artifact in request.available_artifacts:
        if artifact.type == "material_pack":
            return tuple(option for option in artifact.options if option.strip())
    return ()


def _artifact_candidates(request: ReplyRequest) -> tuple[str, ...]:
    labels = {
        "material_pack": "\u6750\u6599\u5305",
        "weekly_report": "\u5468\u62a5",
        "monthly_report": "\u6708\u62a5",
    }
    return tuple(
        label
        for artifact in request.available_artifacts
        if (label := labels.get(artifact.type))
    )


def _candidate_text(prefix: str, candidates: tuple[str, ...] | list[str]) -> str:
    clean = [str(candidate).strip() for candidate in candidates if str(candidate).strip()]
    if clean:
        return _sentence("{}：{}".format(prefix, "、".join(clean)))
    return _sentence(prefix)


def _directive(
    *,
    mode: ResponseMode,
    reply_kind: ReplyKind,
    text: str = "",
    mentions: list[ReplyMention] | None = None,
    action_intents: list[ActionIntentSpec] | None = None,
    requires_knowledge_composer: bool = False,
    composer_stage: Literal["knowledge_composer", "smalltalk_composer"] | None = None,
    reason_code: str = "",
) -> ResponseDirective:
    return ResponseDirective(
        mode=mode,
        reply_kind=reply_kind,
        text=_sentence(text) if text else "",
        mentions=mentions or [],
        action_intents=action_intents or [],
        requires_knowledge_composer=requires_knowledge_composer,
        composer_stage=composer_stage,
        reason_code=reason_code,
    )


def _send_scope_conflict_text(plan: ExecutionPlan, request: ReplyRequest) -> str:
    for decision in plan.guardrail_decisions:
        if decision.reason_code != "send_scope_destination_outside_current_channel":
            continue
        requested = str(decision.metadata.get("requested_target") or "").strip()
        current = str(decision.metadata.get("current_scope") or request.dist_channel_name).strip()
        label = _artifact_label(plan.artifact_kind)
        if not requested:
            requested = "其他渠道"
        return (
            f"当前群是{current}相关沟通群，你要的是{requested}的{label}，"
            f"我不能把它替换成当前渠道发送。请在对应渠道群操作，或确认是否发送{current}的{label}。"
        )
    return ""


def _artifact_label(artifact_kind: str) -> str:
    if artifact_kind == "weekly_report":
        return "周报"
    if artifact_kind == "monthly_report":
        return "月报"
    return "材料"


def _has_knowledge_answer_evidence(
    plan: ExecutionPlan,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
) -> bool:
    return plan_has_knowledge_evidence(plan, evidence_facts, domain_context)


def _material_pack_product_answer(
    plan: ExecutionPlan,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
) -> str:
    if "material_pack" not in plan.answer_capabilities:
        return ""
    products: list[str] = []
    for fact in evidence_facts_for_plan(plan, evidence_facts, domain_context):
        if fact.fact_type != "material_pack_product_list" or not fact.value:
            continue
        for item in fact.metadata.get("products", []):
            name = _product_name(item)
            if name and name not in products:
                products.append(name)
    if not products:
        return ""
    return "材料包包含：" + "、".join(products)


def _product_name(value: object) -> str:
    if isinstance(value, dict):
        for key in ("product_name", "name", "display_name"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
    return str(value or "").strip()


def _sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped.endswith(("。", "！", "？", ".", "!", "?")):
        return stripped
    return stripped + "。"
