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
    capability_by_action_type,
    capability_by_name,
    capability_by_resolve_type,
    resolve_type_for_action,
)
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.domain.compliance_policy import refusal_text_for_reason
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import ActionIntentSpec, ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.send_scope_guard import (
    conflict_explanation,
    detect_send_scope_conflict,
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
            if _has_knowledge_answer_evidence(evidence_facts):
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
                text="当前没有足够的文档证据安全回复，我先不展开。",
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
            scope_conflict = detect_send_scope_conflict(
                request,
                # Strategy canonicalization is already reflected in the plan. The
                # send-scope conflict check still has request-level strategy values.
                canonical_context=CanonicalContext(),
                artifact_kind=plan.artifact_kind,
            )
            if scope_conflict is not None:
                return _directive(
                    mode="unable",
                    reply_kind="unable_to_answer",
                    text=conflict_explanation(scope_conflict),
                    reason_code="send_scope_conflict",
                )

        if plan.response_mode == "action":
            return _action_directive(plan, business_facts, evidence_facts, request)

        return _directive(
            mode="unable",
            reply_kind="unable_to_answer",
            text="当前没有足够证据安全回复。",
            reason_code="insufficient_evidence",
        )


def _report_period_answer(
    plan: ExecutionPlan,
    business_facts: BusinessFacts,
) -> str:
    if plan.evidence_query:
        return ""
    answer_resolve_types = _answer_report_resolve_types(plan)
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
        if resolve_state.status == "ambiguous":
            return _directive(
                mode="clarification",
                reply_kind="clarification",
                text=_candidate_text(
                    "我需要再确认一下你指的是哪一个材料或策略",
                    resolve_state.candidates or tuple(request.available_strategies),
                ),
                reason_code="ambiguous_action_resolve",
            )
        if resolve_state.status != "available":
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

        report_block = _report_action_block(
            action_intent,
            business_facts.report_state(resolve_type),
        )
        if report_block:
            return _handoff_or_unable(
                business_facts,
                text=report_block + "我帮你请销售/支持同事确认。",
                reason="report action blocked by adapter evidence",
                reason_code="report_action_blocked",
                unable_text=report_block,
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
        if _has_knowledge_answer_evidence(evidence_facts):
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
        reply_kind="answer",
        action_intents=plan.action_intents,
        reason_code="action_ready",
    )


def _report_action_block(
    action_intent: ActionIntentSpec,
    report_state: ReportState | None,
) -> str:
    capability = capability_by_action_type(action_intent.action_type)
    if capability is None or not capability.is_report:
        return ""
    label = capability.prompt_label or "报告"
    if report_state is None:
        return "当前没有足够证据确认该报告可发送。"
    strategy = report_state.strategy or action_intent.strategy or ""
    if report_state.scope_status == "excluded":
        return f"{label}未覆盖当前请求的策略，我不能直接发送该报告。"
    if report_state.contains_strategy is False:
        if strategy:
            return f"{label}未包含{strategy}，我不能直接发送该报告。"
        return f"{label}未覆盖当前请求的策略，我不能直接发送该报告。"
    if (
        action_intent.report_scope == "strategy"
        and report_state.contains_strategy is not True
        and report_state.scope_status != "included"
    ):
        return "当前没有足够证据确认该报告覆盖请求的策略。"
    return ""


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
    slots = set(plan.ambiguity_slots)
    if "strategy" in slots:
        candidates = _best_candidates(business_facts, request)
        return _candidate_text("我需要再确认一下具体策略", candidates)
    if "report_scope" in slots:
        return "我需要再确认是发送这个渠道的整体报告，还是某个策略对应的报告。"
    if "artifact" in slots:
        return "我需要再确认你需要的是材料包、周报、月报，还是文档信息。"
    return "我需要再确认一下具体需求后再处理。"


def _best_candidates(
    business_facts: BusinessFacts,
    request: ReplyRequest,
) -> tuple[str, ...]:
    for state in (
        business_facts.material_pack,
        business_facts.weekly_report,
        business_facts.monthly_report,
    ):
        if state.candidates:
            return state.candidates
    return tuple(strategy for strategy in request.available_strategies if strategy.strip())


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


def _has_knowledge_answer_evidence(evidence_facts: list[EvidenceFact]) -> bool:
    if any(
        fact.fact_type == "document_context"
        and bool(fact.value)
        and (
            fact.source_type == "document_mcp"
            or (
                fact.source_type == "approved_static_knowledge"
                and fact.metadata.get("approved_static_knowledge") is True
                and fact.metadata.get("content_is_data_only") is True
            )
        )
        for fact in evidence_facts
    ):
        return True
    return any(
        (
            (
                fact.source_type == "adapter_report_scope"
                and fact.fact_type
                in {"report_scope_summary", "report_scope_match", "report_scope_products"}
            )
            or (
                fact.source_type == "adapter_resolve"
                and fact.fact_type == "report_period"
            )
        )
        and bool(fact.value)
        for fact in evidence_facts
    )


def _sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped.endswith(("。", "！", "？", ".", "!", "?")):
        return stripped
    return stripped + "。"
