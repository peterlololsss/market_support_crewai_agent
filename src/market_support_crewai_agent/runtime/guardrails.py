from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from market_support_crewai_agent.runtime.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.compliance_policy import safe_fallback_text
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.planning import ReplyPlan
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.schemas import (
    AdapterResolveType,
    PrimaryReply,
    ReplyKind,
    ReplyMention,
    ReplyRequest,
    ReplyResponse,
)

ValidationSeverity = Literal["info", "warning", "error", "fatal"]
ValidationCode = Literal[
    "reply_kind_not_allowed",
    "action_type_not_allowed",
    "action_not_resolvable",
    "material_pack_ambiguous",
    "bank_material_pack_requires_strategy_confirmation",
    "sales_mention_not_resolvable",
    "strategy_not_available",
    "unsupported_report_scope_claim",
    "unsupported_report_content_claim",
    "report_action_strategy_unavailable",
    "knowledge_answer_without_document_evidence",
    "action_not_in_plan_candidate",
    "report_action_selector_missing",
    "report_action_strategy_selector_missing_strategy",
    "ambiguous_plan_has_actions",
    "ambiguous_plan_reply_kind",
    "pre_execution_success_claim",
    "side_effect_action_reply_text_not_empty",
    "sent_claim_without_ledger_evidence",
    "non_compliant_reply_has_actions",
    "non_compliant_reply_has_mentions",
    "non_compliant_reply_kind",
    "non_compliant_reply_text",
    "no_reply_not_empty",
    "internal_locator_leak",
]

_SEVERITY_RANK: dict[ValidationSeverity, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
    "fatal": 3,
}
_ACTION_RESOLVE_TYPE: dict[str, AdapterResolveType] = {
    "send_material_pack": "material_pack",
    "send_weekly_report": "weekly_report",
    "send_monthly_report": "monthly_report",
}
_ACTION_RESOLVABLE_FACT = {
    "send_material_pack": "material_pack_resolvable",
    "send_weekly_report": "weekly_report_resolvable",
    "send_monthly_report": "monthly_report_resolvable",
}
_WEEKLY_TOKENS = ("周报", "weekly")
_MONTHLY_TOKENS = ("月报", "monthly")
_SCOPE_EXCLUSION_TOKENS = (
    "不在{}生成范围",
    "不属于{}生成范围",
    "未纳入{}生成范围",
    "没有纳入{}生成范围",
    "outside {} scope",
    "excluded from {} scope",
)
_REPORT_MISSING_TOKENS = (
    "{}不包含",
    "{}没有包含",
    "{}未包含",
    "{}不含",
    "没有在{}中",
    "未在{}中",
    "not included in {}",
    "{} does not include",
    "{} does not show",
)
_RAW_LOCATOR_TOKENS = (
    "://",
    "file:",
    "/users/",
    "/home/",
    "\\users\\",
)
_COMPLETED_SEND_CLAIM_TOKENS = (
    "已发送",
    "已经发送",
    "已为您发送",
    "已帮您发送",
    "已发",
    "发好了",
    "发送完成",
    "以上是最新",
    "请查收",
    "has been sent",
    "sent successfully",
)
_RECENT_SEND_REFERENCE_TOKENS = (
    "刚发",
    "刚刚发",
    "刚才发",
    "刚发送",
    "上次发",
    "之前发",
    "前面发",
    "前面发送",
    "just sent",
    "sent earlier",
    "previously sent",
)
_MATERIAL_TOKENS = ("材料包", "材料", "material")


@dataclass(frozen=True)
class ValidationIssue:
    code: ValidationCode
    message: str
    severity: ValidationSeverity = "error"
    repairable: bool = True
    fallback_reply_kind: ReplyKind | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def severity(self) -> ValidationSeverity:
        if not self.issues:
            return "info"
        return max(self.issues, key=lambda issue: _SEVERITY_RANK[issue.severity]).severity

    @property
    def repairable(self) -> bool:
        return all(issue.repairable for issue in self.issues)

    @property
    def fallback_reply_kind(self) -> ReplyKind | None:
        for issue in self.issues:
            if issue.fallback_reply_kind is not None:
                return issue.fallback_reply_kind
        return None


@dataclass(frozen=True)
class GuardrailOutcome:
    response: ReplyResponse
    validation: ValidationResult
    fallback_used: bool = False


def validate_reply(
        response: ReplyResponse,
        policy: PolicyManifest,
        business_facts: BusinessFacts,
        request: ReplyRequest | None = None,
        plan: ReplyPlan | None = None,
        evidence_facts: list[EvidenceFact] | None = None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []

    if response.reply.kind not in policy.allowed_reply_kinds:
        issues.append(
            ValidationIssue(
                code="reply_kind_not_allowed",
                message=f"reply kind {response.reply.kind} is not allowed by policy",
                severity="fatal",
                repairable=False,
                fallback_reply_kind="no_reply",
            )
        )

    if response.reply.kind == "no_reply" and (
            response.reply.text.strip() or response.reply.mentions or response.actions
    ):
        issues.append(
            ValidationIssue(
                code="no_reply_not_empty",
                message="no_reply must not include text, mentions, or actions",
                severity="fatal",
                repairable=False,
                fallback_reply_kind="no_reply",
            )
        )

    if _contains_raw_locator(response.reply.text):
        issues.append(
            ValidationIssue(
                code="internal_locator_leak",
                message="reply text contains raw locator-like content",
                severity="fatal",
                repairable=False,
                fallback_reply_kind="no_reply",
            )
        )

    issues.extend(_validate_non_compliant_plan_response(response, plan))
    issues.extend(_validate_ambiguous_plan_response(response, plan))

    if response.reply.mentions and not business_facts.sales_mention.resolvable:
        issues.append(
            ValidationIssue(
                code="sales_mention_not_resolvable",
                message="reply mentions require resolved sales_mention evidence",
                fallback_reply_kind="unable_to_answer",
            )
        )

    issues.extend(_validate_action_plan_alignment(response, plan))
    issues.extend(_validate_report_action_selectors(response, plan))
    issues.extend(
        _validate_bank_material_pack_confirmation(response, business_facts, request)
    )
    issues.extend(_validate_actions(response, policy, business_facts, request))
    issues.extend(_validate_report_action_strategy_availability(response, business_facts))
    issues.extend(_validate_pre_execution_success_claim(response))
    issues.extend(_validate_side_effect_action_reply_text_empty(response))
    issues.extend(_validate_sent_claims_grounded_by_ledger(response, business_facts))
    issues.extend(_validate_report_claims(response.reply.text, business_facts))
    issues.extend(_validate_knowledge_grounding(response, plan, evidence_facts))

    return ValidationResult(valid=not issues, issues=tuple(issues))


def apply_reply_guardrails(
        response: ReplyResponse,
        policy: PolicyManifest,
        business_facts: BusinessFacts,
        request: ReplyRequest | None = None,
        plan: ReplyPlan | None = None,
        evidence_facts: list[EvidenceFact] | None = None,
) -> GuardrailOutcome:
    validation = validate_reply(
        response,
        policy,
        business_facts,
        request,
        plan=plan,
        evidence_facts=evidence_facts,
    )
    if validation.valid:
        return GuardrailOutcome(
            response=response,
            validation=validation,
            fallback_used=False,
        )

    return GuardrailOutcome(
        response=build_deterministic_fallback(
            validation,
            response,
            policy,
            business_facts,
            request,
        ),
        validation=validation,
        fallback_used=True,
    )


def build_deterministic_fallback(
        validation: ValidationResult,
        response: ReplyResponse | None,
        policy: PolicyManifest,
        business_facts: BusinessFacts,
        request: ReplyRequest | None = None,
) -> ReplyResponse:
    del policy, request

    response_id = response.response_id if response is not None else ""
    issue = validation.issues[0] if validation.issues else None
    if issue is None:
        return no_safe_reply(response_id)

    if issue.code == "material_pack_ambiguous":
        candidates = issue.metadata.get("candidates")
        if isinstance(candidates, list) and candidates:
            return ReplyResponse(
                response_id=response_id,
                reply=PrimaryReply(
                    kind="clarification",
                    text="我需要再确认一下你指的是哪一个材料或策略：{}。".format(
                        "、".join(str(candidate) for candidate in candidates)
                    ),
                ),
                actions=[],
            )
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="clarification",
                text="我需要再确认一下你指的是哪一个材料或策略。",
            ),
            actions=[],
        )

    if issue.code == "bank_material_pack_requires_strategy_confirmation":
        candidates = issue.metadata.get("candidates")
        if isinstance(candidates, list) and candidates:
            text = "我需要先确认您要发送哪一个策略的材料包：{}。".format(
                "、".join(str(candidate) for candidate in candidates)
            )
        else:
            text = "我需要先确认您要发送哪一个策略的材料包。"
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(kind="clarification", text=text),
            actions=[],
        )

    if issue.code in {
        "unsupported_report_scope_claim",
        "unsupported_report_content_claim",
    }:
        report_label = str(issue.metadata.get("report_label") or "报告")
        if business_facts.sales_mention.resolvable:
            return ReplyResponse(
                response_id=response_id,
                reply=PrimaryReply(
                    kind="human_handoff",
                    text="当前没有足够的{}范围证据，不能确认该策略是否纳入，我帮你请销售/支持同事确认。".format(
                        report_label
                    ),
                    mentions=[
                        ReplyMention(
                            type="sales",
                            reason="guardrail blocked unsupported report scope claim",
                        )
                    ],
                ),
                actions=[],
            )
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="unable_to_answer",
                text="当前没有足够的{}范围证据，不能确认该策略是否纳入。".format(
                    report_label
                ),
            ),
            actions=[],
        )

    if issue.code == "report_action_strategy_unavailable":
        report_label = str(issue.metadata.get("report_label") or "报告")
        reason = str(issue.metadata.get("reason") or "")
        strategy = str(issue.metadata.get("strategy") or "")
        if strategy and reason == "scope_excluded":
            text = "{}暂未覆盖{}，我不能直接发送该报告。".format(
                report_label,
                strategy,
            )
        elif strategy:
            text = "{}未包含{}，我不能直接发送该报告。".format(
                report_label,
                strategy,
            )
        else:
            text = "{}未覆盖当前请求的策略，我不能直接发送该报告。".format(
                report_label,
            )
        if business_facts.sales_mention.resolvable:
            return ReplyResponse(
                response_id=response_id,
                reply=PrimaryReply(
                    kind="human_handoff",
                    text=text + "我帮你请销售/支持同事确认。",
                    mentions=[
                        ReplyMention(
                            type="sales",
                            reason="report action blocked by negative scope evidence",
                        )
                    ],
                ),
                actions=[],
            )
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(kind="unable_to_answer", text=text),
            actions=[],
        )

    if issue.code == "action_not_resolvable":
        if business_facts.sales_mention.resolvable:
            return ReplyResponse(
                response_id=response_id,
                reply=PrimaryReply(
                    kind="human_handoff",
                    text="目前这个渠道下我没有看到可发送的对应材料，我帮你请销售/支持同事确认。",
                    mentions=[
                        ReplyMention(
                            type="sales",
                            reason="guardrail blocked unresolved side-effect action",
                        )
                    ],
                ),
                actions=[],
            )
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="unable_to_answer",
                text="目前这个渠道下我没有看到可发送的对应材料。",
            ),
            actions=[],
        )

    if issue.code == "sales_mention_not_resolvable":
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="unable_to_answer",
                text="这个问题需要销售/支持同事协助确认，但当前渠道暂未配置可用负责人。",
                mentions=[],
            ),
            actions=[],
        )

    if issue.code == "knowledge_answer_without_document_evidence":
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="unable_to_answer",
                text="当前没有足够的文档证据安全回复，我先不展开。",
            ),
            actions=[],
        )

    if issue.code in {"ambiguous_plan_has_actions", "ambiguous_plan_reply_kind"}:
        reason = str(issue.metadata.get("ambiguity_reason") or "")
        text = "我需要再确认一下具体需求后再处理。"
        if reason:
            text = "我需要再确认一下：{}。".format(reason)
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(kind="clarification", text=text),
            actions=[],
        )

    if issue.code in {
        "report_action_selector_missing",
        "report_action_strategy_selector_missing_strategy",
    }:
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="clarification",
                text="我需要先确认要发送的报告范围后再处理。",
            ),
            actions=[],
        )

    if issue.code == "pre_execution_success_claim":
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="answer",
                text="",
                mentions=response.reply.mentions if response is not None else [],
            ),
            actions=response.actions if response is not None else [],
        )

    if issue.code == "side_effect_action_reply_text_not_empty":
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind=response.reply.kind if response is not None else "answer",
                text="",
                mentions=response.reply.mentions if response is not None else [],
            ),
            actions=response.actions if response is not None else [],
        )

    if issue.code == "sent_claim_without_ledger_evidence":
        if business_facts.sales_mention.resolvable:
            return ReplyResponse(
                response_id=response_id,
                reply=PrimaryReply(
                    kind="human_handoff",
                    text="我没有看到这个会话里已执行的发送记录，我帮你请销售/支持同事确认你指的是哪份材料。",
                    mentions=[
                        ReplyMention(
                            type="sales",
                            reason="sent claim lacks adapter-confirmed ledger evidence",
                        )
                    ],
                ),
                actions=[],
            )
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="clarification",
                text="我没有看到这个会话里已执行的发送记录，需要先确认你指的是哪份材料。",
            ),
            actions=[],
        )

    if issue.code in {
        "non_compliant_reply_has_actions",
        "non_compliant_reply_has_mentions",
        "non_compliant_reply_kind",
        "non_compliant_reply_text",
    }:
        return ReplyResponse(
            response_id=response_id,
            reply=PrimaryReply(
                kind="unable_to_answer",
                text=safe_fallback_text(
                    str(issue.metadata.get("reason_code") or "unknown")
                ),
            ),
            actions=[],
        )

    if issue.fallback_reply_kind == "no_reply":
        return no_safe_reply(response_id)

    return ReplyResponse(
        response_id=response_id,
        reply=PrimaryReply(
            kind="unable_to_answer",
            text="当前没有足够证据安全回复，我帮你保守处理。",
        ),
        actions=[],
    )


def no_safe_reply(response_id: str = "") -> ReplyResponse:
    return ReplyResponse(
        response_id=response_id,
        reply=PrimaryReply(kind="no_reply", text="", mentions=[]),
        actions=[],
    )


def _validate_actions(
        response: ReplyResponse,
        policy: PolicyManifest,
        business_facts: BusinessFacts,
        request: ReplyRequest | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for action in response.actions:
        action_type = getattr(action, "type", "")
        if action_type not in policy.allowed_side_effect_actions:
            issues.append(
                ValidationIssue(
                    code="action_type_not_allowed",
                    message=f"action type {action_type} is not allowed by policy",
                    severity="fatal",
                    repairable=False,
                    fallback_reply_kind="unable_to_answer",
                    metadata={"action_type": action_type},
                )
            )
            continue

        resolve_type = _ACTION_RESOLVE_TYPE.get(action_type)
        fact_type = _ACTION_RESOLVABLE_FACT.get(action_type)
        if resolve_type is None or fact_type is None:
            continue

        resolve_state = business_facts.resolve_state(resolve_type)
        if not resolve_state.resolvable:
            status = resolve_state.status
            if action_type == "send_material_pack" and status == "ambiguous":
                issues.append(
                    ValidationIssue(
                        code="material_pack_ambiguous",
                        message="material pack action is ambiguous",
                        fallback_reply_kind="clarification",
                        metadata={
                            "action_type": action_type,
                            "resolve_type": resolve_type,
                            "status": status,
                            "candidates": list(resolve_state.candidates),
                        },
                    )
                )
                continue

            issues.append(
                ValidationIssue(
                    code="action_not_resolvable",
                    message=f"{action_type} requires resolved {resolve_type} evidence",
                    fallback_reply_kind="human_handoff",
                    metadata={
                        "action_type": action_type,
                        "resolve_type": resolve_type,
                        "status": status,
                    },
                )
            )

        if request is not None and action_type == "send_material_pack":
            strategy = getattr(action, "strategy", None)
            if strategy and request.available_strategies:
                if strategy not in request.available_strategies:
                    issues.append(
                        ValidationIssue(
                            code="strategy_not_available",
                            message="material pack strategy is not in available_strategies",
                            fallback_reply_kind="clarification",
                            metadata={"strategy": strategy},
                        )
                    )

    return issues


def _validate_bank_material_pack_confirmation(
        response: ReplyResponse,
        business_facts: BusinessFacts,
        request: ReplyRequest | None,
) -> list[ValidationIssue]:
    if request is None or request.channel_type != "bank":
        return []
    if len(_available_strategy_candidates(request)) <= 1:
        return []
    if business_facts.material_pack.status != "available":
        return []

    issues: list[ValidationIssue] = []
    for action in response.actions:
        if action.type != "send_material_pack":
            continue
        action_strategy = getattr(action, "strategy", None)
        if action_strategy or business_facts.material_pack.strategy:
            continue
        issues.append(
            ValidationIssue(
                code="bank_material_pack_requires_strategy_confirmation",
                message="bank material pack action requires a confirmed strategy",
                fallback_reply_kind="clarification",
                metadata={
                    "action_type": action.type,
                    "candidates": _material_pack_candidates(business_facts, request),
                },
            )
        )
    return issues


def _validate_report_action_strategy_availability(
        response: ReplyResponse,
        business_facts: BusinessFacts,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for action in response.actions:
        resolve_type = _ACTION_RESOLVE_TYPE.get(getattr(action, "type", ""))
        if resolve_type not in {"weekly_report", "monthly_report"}:
            continue
        report_state = business_facts.report_state(resolve_type)
        if report_state is None or not report_state.resolvable:
            continue

        reason = ""
        if report_state.scope_status == "excluded":
            reason = "scope_excluded"
        elif report_state.contains_strategy is False:
            reason = "strategy_not_in_report"
        if not reason:
            continue

        issues.append(
            ValidationIssue(
                code="report_action_strategy_unavailable",
                message="report action is not allowed when adapter evidence excludes the requested strategy",
                fallback_reply_kind="human_handoff",
                metadata={
                    "action_type": action.type,
                    "resolve_type": resolve_type,
                    "report_label": _report_label(resolve_type),
                    "reason": reason,
                    "strategy": report_state.strategy,
                    "period": report_state.period,
                    "scope_status": report_state.scope_status,
                    "contains_strategy": report_state.contains_strategy,
                },
            )
        )
    return issues


def _validate_action_plan_alignment(
        response: ReplyResponse,
        plan: ReplyPlan | None,
) -> list[ValidationIssue]:
    if plan is None or not response.actions:
        return []

    planned_actions = tuple(plan.candidate_actions)
    issues: list[ValidationIssue] = []
    for action in response.actions:
        if any(_action_matches_plan(action, candidate) for candidate in planned_actions):
            continue
        issues.append(
            ValidationIssue(
                code="action_not_in_plan_candidate",
                message="final side-effect action was not proposed by the validated plan",
                fallback_reply_kind="unable_to_answer",
                metadata={
                    "action_type": action.type,
                    "planned_action_types": [
                        candidate.type for candidate in planned_actions
                    ],
                },
            )
        )
    return issues


def _validate_report_action_selectors(
        response: ReplyResponse,
        plan: ReplyPlan | None,
) -> list[ValidationIssue]:
    if plan is None or not response.actions:
        return []

    issues: list[ValidationIssue] = []
    for action in response.actions:
        if getattr(action, "type", "") not in {
            "send_weekly_report",
            "send_monthly_report",
        }:
            continue
        candidate = _matching_plan_candidate(action, plan)
        if candidate is None:
            continue
        if candidate.report_scope == "unknown":
            issues.append(
                ValidationIssue(
                    code="report_action_selector_missing",
                    message="report send action requires a known plan report_scope",
                    repairable=False,
                    fallback_reply_kind="clarification",
                    metadata={"action_type": action.type},
                )
            )
        elif candidate.report_scope == "strategy" and not candidate.strategy:
            issues.append(
                ValidationIssue(
                    code="report_action_strategy_selector_missing_strategy",
                    message="strategy-scoped report send action requires plan strategy",
                    repairable=False,
                    fallback_reply_kind="clarification",
                    metadata={"action_type": action.type},
                )
            )
    return issues


def _matching_plan_candidate(action, plan: ReplyPlan):
    for candidate in plan.candidate_actions:
        if _action_matches_plan(action, candidate):
            return candidate
    return None


def _action_matches_plan(action, candidate) -> bool:
    if action.type != candidate.type:
        return False
    candidate_strategy = getattr(candidate, "strategy", None)
    action_strategy = getattr(action, "strategy", None)
    if candidate_strategy and action_strategy and candidate_strategy != action_strategy:
        return False
    return True


def _available_strategy_candidates(request: ReplyRequest) -> list[str]:
    return [
        strategy.strip()
        for strategy in request.available_strategies
        if strategy.strip()
    ]


def _material_pack_candidates(
        business_facts: BusinessFacts,
        request: ReplyRequest,
) -> list[str]:
    if business_facts.material_pack.candidates:
        return list(business_facts.material_pack.candidates)
    return _available_strategy_candidates(request)


def _report_label(resolve_type: AdapterResolveType) -> str:
    if resolve_type == "weekly_report":
        return "周报"
    if resolve_type == "monthly_report":
        return "月报"
    return "报告"


def _validate_pre_execution_success_claim(
        response: ReplyResponse,
) -> list[ValidationIssue]:
    if not response.actions:
        return []
    normalized_text = str(response.reply.text or "").lower()
    if not normalized_text:
        return []
    for token in _COMPLETED_SEND_CLAIM_TOKENS:
        if token.lower() in normalized_text:
            return [
                ValidationIssue(
                    code="pre_execution_success_claim",
                    message="reply text claims a send completed before adapter execution",
                    fallback_reply_kind="answer",
                    metadata={"token": token},
                )
            ]
    return []


def _validate_side_effect_action_reply_text_empty(
        response: ReplyResponse,
) -> list[ValidationIssue]:
    if not response.actions or not response.reply.text.strip():
        return []
    return [
        ValidationIssue(
            code="side_effect_action_reply_text_not_empty",
            message=(
                "material/report side-effect responses must leave reply.text "
                "empty because adapter execution owns post-send wording"
            ),
            fallback_reply_kind="answer",
        )
    ]


def _validate_sent_claims_grounded_by_ledger(
        response: ReplyResponse,
        business_facts: BusinessFacts,
) -> list[ValidationIssue]:
    normalized_text = str(response.reply.text or "").lower()
    if not normalized_text:
        return []

    token = _first_send_claim_token(normalized_text)
    if not token:
        return []

    material_type = _claimed_material_type(normalized_text)
    if _has_matching_recent_executed_action(business_facts, material_type):
        return []

    return [
        ValidationIssue(
            code="sent_claim_without_ledger_evidence",
            message=(
                "reply text claims prior send without matching adapter-confirmed "
                "ledger evidence"
            ),
            fallback_reply_kind="clarification",
            metadata={
                "token": token,
                "material_type": material_type,
            },
        )
    ]


def _first_send_claim_token(normalized_text: str) -> str:
    for token in _COMPLETED_SEND_CLAIM_TOKENS + _RECENT_SEND_REFERENCE_TOKENS:
        if token.lower() in normalized_text:
            return token
    return ""


def _claimed_material_type(normalized_text: str) -> str | None:
    if _report_label_in_text(normalized_text, _WEEKLY_TOKENS):
        return "weekly"
    if _report_label_in_text(normalized_text, _MONTHLY_TOKENS):
        return "monthly"
    if any(token.lower() in normalized_text for token in _MATERIAL_TOKENS):
        return "material"
    return None


def _has_matching_recent_executed_action(
        business_facts: BusinessFacts,
        material_type: str | None,
) -> bool:
    for action in business_facts.recent_executed_actions:
        if material_type is None:
            return True
        if action.material_type == material_type:
            return True
    return False


def _validate_report_claims(
        text: str,
        business_facts: BusinessFacts,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    normalized = str(text or "").lower()
    if not normalized:
        return issues

    for resolve_type, label_tokens, report_label in (
            ("weekly_report", _WEEKLY_TOKENS, "周报"),
            ("monthly_report", _MONTHLY_TOKENS, "月报"),
    ):
        label = _report_label_in_text(normalized, label_tokens)
        if not label:
            continue

        if _contains_template_token(normalized, _SCOPE_EXCLUSION_TOKENS, label):
            report_state = business_facts.report_state(resolve_type)
            if report_state is None or report_state.scope_status != "excluded":
                issues.append(
                    ValidationIssue(
                        code="unsupported_report_scope_claim",
                        message=f"{resolve_type} scope exclusion claim lacks excluded evidence",
                        fallback_reply_kind="human_handoff",
                        metadata={
                            "resolve_type": resolve_type,
                            "report_label": report_label,
                        },
                    )
                )
        if _contains_template_token(normalized, _REPORT_MISSING_TOKENS, label):
            report_state = business_facts.report_state(resolve_type)
            if report_state is None or report_state.contains_strategy is not False:
                issues.append(
                    ValidationIssue(
                        code="unsupported_report_content_claim",
                        message=f"{resolve_type} non-inclusion claim lacks negative evidence",
                        fallback_reply_kind="human_handoff",
                        metadata={
                            "resolve_type": resolve_type,
                            "report_label": report_label,
                        },
                    )
                )
    return issues


def _validate_non_compliant_plan_response(
        response: ReplyResponse,
        plan: ReplyPlan | None,
) -> list[ValidationIssue]:
    if plan is None or plan.compliance.is_compliant is not False:
        return []

    metadata = {
        "reason_code": plan.compliance.reason_code,
        "intent": plan.intent,
    }
    issues: list[ValidationIssue] = []
    if response.actions:
        issues.append(
            ValidationIssue(
                code="non_compliant_reply_has_actions",
                message="non-compliant plan response must not include side-effect actions",
                repairable=False,
                fallback_reply_kind="unable_to_answer",
                metadata=metadata,
            )
        )
    if response.reply.mentions:
        issues.append(
            ValidationIssue(
                code="non_compliant_reply_has_mentions",
                message="non-compliant plan response must not mention sales",
                repairable=False,
                fallback_reply_kind="unable_to_answer",
                metadata=metadata,
            )
        )
    if response.reply.kind != "unable_to_answer":
        issues.append(
            ValidationIssue(
                code="non_compliant_reply_kind",
                message="non-compliant plan response must use unable_to_answer",
                repairable=False,
                fallback_reply_kind="unable_to_answer",
                metadata=metadata,
            )
        )
    expected_text = safe_fallback_text(plan.compliance.reason_code).strip()
    if response.reply.text.strip() != expected_text:
        issues.append(
            ValidationIssue(
                code="non_compliant_reply_text",
                message="non-compliant plan response must use the harness-owned safe fallback text",
                repairable=False,
                fallback_reply_kind="unable_to_answer",
                metadata=metadata,
            )
        )
    return issues


def _validate_ambiguous_plan_response(
        response: ReplyResponse,
        plan: ReplyPlan | None,
) -> list[ValidationIssue]:
    if plan is None or not plan.ambiguity:
        return []

    metadata = {
        "ambiguity_reason": plan.ambiguity_reason,
        "intent": plan.intent,
    }
    issues: list[ValidationIssue] = []
    if response.actions:
        issues.append(
            ValidationIssue(
                code="ambiguous_plan_has_actions",
                message="ambiguous plan response must not include side-effect actions",
                fallback_reply_kind="clarification",
                metadata=metadata,
            )
        )
    if response.reply.kind not in {
        "clarification",
        "human_handoff",
        "unable_to_answer",
        "no_reply",
    }:
        issues.append(
            ValidationIssue(
                code="ambiguous_plan_reply_kind",
                message="ambiguous plan response must clarify, hand off, or decline safely",
                fallback_reply_kind="clarification",
                metadata=metadata,
            )
        )
    return issues


def _validate_knowledge_grounding(
        response: ReplyResponse,
        plan: ReplyPlan | None,
        evidence_facts: list[EvidenceFact] | None,
) -> list[ValidationIssue]:
    if plan is None or plan.intent != "knowledge_qa":
        return []
    if response.reply.kind != "answer" or not response.reply.text.strip():
        return []
    if any(
        fact.fact_type == "document_context" and fact.source_type == "document_mcp"
        for fact in evidence_facts or []
    ):
        return []
    return [
        ValidationIssue(
            code="knowledge_answer_without_document_evidence",
            message="knowledge_qa answer requires document MCP evidence",
            fallback_reply_kind="unable_to_answer",
        )
    ]


def _report_label_in_text(normalized_text: str, tokens: tuple[str, ...]) -> str:
    for token in tokens:
        if token.lower() in normalized_text:
            return token.lower()
    return ""


def _contains_template_token(
        normalized_text: str,
        templates: tuple[str, ...],
        label: str,
) -> bool:
    return any(template.format(label) in normalized_text for template in templates)


def _contains_raw_locator(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(token in normalized for token in _RAW_LOCATOR_TOKENS)
