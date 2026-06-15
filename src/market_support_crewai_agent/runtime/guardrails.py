from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from market_support_crewai_agent.runtime.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.capabilities import (
    capability_by_action_type,
    capability_by_resolve_type,
    report_action_types,
    resolve_type_for_action,
)
from market_support_crewai_agent.runtime.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.planning import ActionIntentSpec, ExecutionPlan
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.schemas import AdapterResolveType, ReplyResponse

ValidationSeverity = Literal["info", "warning", "error", "fatal"]
ValidationCode = Literal[
    "reply_mode_not_allowed",
    "reply_kind_mismatch",
    "no_reply_not_empty",
    "internal_locator_leak",
    "action_not_allowed_for_directive",
    "action_count_mismatch",
    "action_not_in_directive",
    "action_type_not_allowed",
    "action_missing_resolve_ref",
    "action_not_resolvable",
    "action_resolve_ref_mismatch",
    "outbound_action_reply_text_not_empty",
    "outbound_action_reply_mentions_not_empty",
    "non_compliant_reply_has_actions",
    "non_compliant_reply_has_mentions",
    "non_compliant_reply_kind",
    "non_compliant_reply_text",
    "handoff_missing_sales_mention",
    "sales_mention_not_resolvable",
    "knowledge_answer_without_document_evidence",
    "image_marker_not_allowed",
    "image_marker_not_in_evidence",
    "sent_claim_without_ledger_evidence",
    "unsupported_report_scope_claim",
    "unsupported_report_content_claim",
    "report_action_strategy_unavailable",
]

_SEVERITY_RANK: dict[ValidationSeverity, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
    "fatal": 3,
}
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
_WEEKLY_TOKENS = ("周报", "weekly")
_MONTHLY_TOKENS = ("月报", "monthly")
_MATERIAL_TOKENS = ("材料包", "材料", "material")
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
_IMAGE_MARKER_RE = re.compile(r"%%([\w\d_.-]+\.png)%%")
_ALLOWED_IMAGE_MARKERS = frozenset(
    {
        "comp_wx_qr_code.png",
        "alpha_beta_comparison.png",
        "quant_difference.png",
        "company_shareholders.png",
        "SH000985_weights.png",
        "SH000985_features.png",
        "SH000985_constituents.png",
        "company_historical_aum.png",
        "indice_comparison.png",
    }
)


class ReplyContractError(RuntimeError):
    """Raised when a rendered ReplyResponse violates postcondition validation."""


@dataclass(frozen=True)
class ValidationIssue:
    code: ValidationCode
    message: str
    severity: ValidationSeverity = "error"
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


def validate_reply(
    response: ReplyResponse,
    directive: ResponseDirective,
    plan: ExecutionPlan,
    business_facts: BusinessFacts,
    evidence_facts: list[EvidenceFact],
    policy: PolicyManifest,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    issues.extend(_validate_policy_and_kind(response, directive, policy))
    issues.extend(_validate_no_reply(response))
    issues.extend(_validate_locator_leaks(response))
    issues.extend(_validate_non_compliant_response(response, directive, plan))
    issues.extend(_validate_handoff(response, directive, business_facts))
    issues.extend(_validate_actions(response, directive, business_facts, policy))
    issues.extend(_validate_knowledge_grounding(response, directive, evidence_facts))
    issues.extend(_validate_image_markers(response.reply.text, evidence_facts))
    issues.extend(_validate_sent_claims_grounded_by_ledger(response, business_facts))
    issues.extend(_validate_report_claims(response.reply.text, business_facts))
    return ValidationResult(valid=not issues, issues=tuple(issues))


def _validate_policy_and_kind(
    response: ReplyResponse,
    directive: ResponseDirective,
    policy: PolicyManifest,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if directive.mode not in policy.allowed_reply_modes:
        issues.append(
            ValidationIssue(
                code="reply_mode_not_allowed",
                message=f"directive mode {directive.mode} is not allowed by policy",
                severity="fatal",
                metadata={"mode": directive.mode},
            )
        )
    if _reply_kind_matches_directive(response, directive):
        return issues
    if response.reply.kind != directive.reply_kind:
        issues.append(
            ValidationIssue(
                code="reply_kind_mismatch",
                message="reply.kind must match the response directive",
                severity="fatal",
                metadata={
                    "reply_kind": response.reply.kind,
                    "directive_reply_kind": directive.reply_kind,
                },
            )
        )
    return issues


def _reply_kind_matches_directive(
    response: ReplyResponse,
    directive: ResponseDirective,
) -> bool:
    if response.reply.kind == directive.reply_kind:
        return True
    return (
        directive.mode == "knowledge_answer"
        and directive.reply_kind == "answer"
        and response.reply.kind == "unable_to_answer"
    )


def _validate_no_reply(response: ReplyResponse) -> list[ValidationIssue]:
    if response.reply.kind == "no_reply" and (
        response.reply.text.strip() or response.reply.mentions or response.actions
    ):
        return [
            ValidationIssue(
                code="no_reply_not_empty",
                message="no_reply must not include text, mentions, or actions",
                severity="fatal",
            )
        ]
    return []


def _validate_locator_leaks(response: ReplyResponse) -> list[ValidationIssue]:
    if _contains_raw_locator(response.reply.text):
        return [
            ValidationIssue(
                code="internal_locator_leak",
                message="reply text contains raw locator-like content",
                severity="fatal",
            )
        ]
    return []


def _validate_non_compliant_response(
    response: ReplyResponse,
    directive: ResponseDirective,
    plan: ExecutionPlan,
) -> list[ValidationIssue]:
    if directive.mode != "refusal" and plan.compliance.is_compliant is not False:
        return []

    metadata = {
        "reason_code": plan.compliance.reason_code,
        "artifact_kind": plan.artifact_kind,
    }
    issues: list[ValidationIssue] = []
    if response.actions:
        issues.append(
            ValidationIssue(
                code="non_compliant_reply_has_actions",
                message="non-compliant response must not include outbound actions",
                metadata=metadata,
            )
        )
    if response.reply.mentions:
        issues.append(
            ValidationIssue(
                code="non_compliant_reply_has_mentions",
                message="non-compliant response must not include mentions",
                metadata=metadata,
            )
        )
    if response.reply.kind != "unable_to_answer":
        issues.append(
            ValidationIssue(
                code="non_compliant_reply_kind",
                message="non-compliant response must use unable_to_answer",
                metadata=metadata,
            )
        )
    if response.reply.text.strip() != directive.text.strip():
        issues.append(
            ValidationIssue(
                code="non_compliant_reply_text",
                message="non-compliant response must use directive refusal text",
                metadata=metadata,
            )
        )
    return issues


def _validate_handoff(
    response: ReplyResponse,
    directive: ResponseDirective,
    business_facts: BusinessFacts,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if directive.mode == "handoff" and not any(
        mention.type == "sales" for mention in response.reply.mentions
    ):
        issues.append(
            ValidationIssue(
                code="handoff_missing_sales_mention",
                message="human_handoff must include a sales mention",
                metadata={"sales_mention_status": business_facts.sales_mention.status},
            )
        )
    if response.reply.mentions and not business_facts.sales_mention.resolvable:
        issues.append(
            ValidationIssue(
                code="sales_mention_not_resolvable",
                message="reply mentions require resolved sales_mention evidence",
            )
        )
    return issues


def _validate_actions(
    response: ReplyResponse,
    directive: ResponseDirective,
    business_facts: BusinessFacts,
    policy: PolicyManifest,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if response.actions and directive.mode != "action":
        issues.append(
            ValidationIssue(
                code="action_not_allowed_for_directive",
                message="actions are only allowed when directive.mode is action",
                severity="fatal",
                metadata={"mode": directive.mode},
            )
        )
    if directive.mode == "action" and len(response.actions) != len(directive.action_intents):
        issues.append(
            ValidationIssue(
                code="action_count_mismatch",
                message="rendered actions must match directive action intents",
                metadata={
                    "action_count": len(response.actions),
                    "intent_count": len(directive.action_intents),
                },
            )
        )

    for action in response.actions:
        action_type = getattr(action, "type", "")
        if action_type not in policy.allowed_side_effect_actions:
            issues.append(
                ValidationIssue(
                    code="action_type_not_allowed",
                    message=f"action type {action_type} is not allowed by policy",
                    severity="fatal",
                    metadata={"action_type": action_type},
                )
            )
            continue
        candidate = _matching_action_intent(action, directive.action_intents)
        if directive.mode == "action" and candidate is None:
            issues.append(
                ValidationIssue(
                    code="action_not_in_directive",
                    message="rendered action was not requested by the directive",
                    metadata={"action_type": action_type},
                )
            )
        issues.extend(_validate_action_resolve(action, business_facts))
        issues.extend(_validate_report_action(action, candidate, business_facts))

    if response.actions and response.reply.text.strip():
        issues.append(
            ValidationIssue(
                code="outbound_action_reply_text_not_empty",
                message="side-effect action responses must leave reply.text empty",
            )
        )
    if response.actions and response.reply.mentions:
        issues.append(
            ValidationIssue(
                code="outbound_action_reply_mentions_not_empty",
                message="side-effect action responses must leave reply.mentions empty",
            )
        )
    return issues


def _validate_action_resolve(action, business_facts: BusinessFacts) -> list[ValidationIssue]:
    action_type = getattr(action, "type", "")
    resolve_type = resolve_type_for_action(action_type)
    if resolve_type is None:
        return []
    if getattr(action, "resolve_type", None) != resolve_type:
        return [
            ValidationIssue(
                code="action_not_resolvable",
                message="action resolve_type does not match registry",
                severity="fatal",
                metadata={
                    "action_type": action_type,
                    "resolve_type": getattr(action, "resolve_type", None),
                    "expected_resolve_type": resolve_type,
                },
            )
        ]
    resolve_ref = getattr(action, "resolve_ref", "")
    if not str(resolve_ref).strip():
        return [
            ValidationIssue(
                code="action_missing_resolve_ref",
                message="side-effect action must include resolve_ref",
                severity="fatal",
                metadata={"action_type": action_type, "resolve_type": resolve_type},
            )
        ]
    resolve_state = business_facts.resolve_state(resolve_type)
    if not resolve_state.resolvable:
        return [
            ValidationIssue(
                code="action_not_resolvable",
                message=f"{action_type} requires resolved {resolve_type} evidence",
                metadata={
                    "action_type": action_type,
                    "resolve_type": resolve_type,
                    "status": resolve_state.status,
                },
            )
        ]
    if resolve_state.resolve_ref and resolve_ref != resolve_state.resolve_ref:
        return [
            ValidationIssue(
                code="action_resolve_ref_mismatch",
                message="action resolve_ref does not match adapter evidence",
                severity="fatal",
                metadata={
                    "action_type": action_type,
                    "resolve_type": resolve_type,
                    "expected_ref_available": True,
                },
            )
        ]
    return []


def _validate_report_action(
    action,
    candidate: ActionIntentSpec | None,
    business_facts: BusinessFacts,
) -> list[ValidationIssue]:
    action_type = getattr(action, "type", "")
    capability = capability_by_action_type(action_type)
    if capability is None or not capability.is_report or capability.resolve_type is None:
        return []

    report_state = business_facts.report_state(capability.resolve_type)
    if report_state is not None and report_state.resolvable:
        if report_state.scope_status == "excluded" or report_state.contains_strategy is False:
            return [
                ValidationIssue(
                    code="report_action_strategy_unavailable",
                    message="report action is not allowed when adapter evidence excludes the requested strategy",
                    metadata={
                        "action_type": action_type,
                        "resolve_type": capability.resolve_type,
                        "strategy": report_state.strategy,
                        "period": report_state.period,
                        "scope_status": report_state.scope_status,
                        "contains_strategy": report_state.contains_strategy,
                    },
                )
            ]
        if (
            candidate is not None
            and candidate.report_scope == "strategy"
            and report_state.contains_strategy is not True
            and report_state.scope_status != "included"
        ):
            return [
                ValidationIssue(
                    code="report_action_strategy_unavailable",
                    message="strategy-scoped report action lacks positive inclusion evidence",
                    metadata={
                        "action_type": action_type,
                        "resolve_type": capability.resolve_type,
                        "strategy": candidate.strategy,
                    },
                )
            ]
    return []


def _validate_knowledge_grounding(
    response: ReplyResponse,
    directive: ResponseDirective,
    evidence_facts: list[EvidenceFact],
) -> list[ValidationIssue]:
    if directive.mode != "knowledge_answer":
        return []
    if response.reply.kind != "answer" or not response.reply.text.strip():
        return []
    if _has_document_context_evidence(evidence_facts):
        return []
    return [
        ValidationIssue(
            code="knowledge_answer_without_document_evidence",
            message="knowledge answer requires document_context evidence",
        )
    ]


def _validate_image_markers(
    text: str,
    evidence_facts: list[EvidenceFact],
) -> list[ValidationIssue]:
    markers = _image_markers(text)
    if not markers:
        return []
    evidence_text = "\n".join(
        str(fact.value or "")
        for fact in evidence_facts
        if fact.fact_type == "document_context"
    )
    issues: list[ValidationIssue] = []
    for filename in markers:
        marker = f"%%{filename}%%"
        if filename not in _ALLOWED_IMAGE_MARKERS:
            issues.append(
                ValidationIssue(
                    code="image_marker_not_allowed",
                    message="reply text contains an image marker outside the whitelist",
                    severity="fatal",
                    metadata={"filename": filename},
                )
            )
            continue
        if marker not in evidence_text:
            issues.append(
                ValidationIssue(
                    code="image_marker_not_in_evidence",
                    message="reply text image marker must appear in document evidence",
                    severity="fatal",
                    metadata={"filename": filename},
                )
            )
    return issues


def _image_markers(text: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for match in _IMAGE_MARKER_RE.finditer(str(text or "")):
        filename = match.group(1)
        if filename in seen:
            continue
        seen.add(filename)
        output.append(filename)
    return output


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
            message="reply text claims prior send without matching adapter-confirmed ledger evidence",
            metadata={"token": token, "material_type": material_type},
        )
    ]


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
            report_state = business_facts.report_state(resolve_type)  # type: ignore[arg-type]
            if report_state is None or report_state.scope_status != "excluded":
                issues.append(
                    ValidationIssue(
                        code="unsupported_report_scope_claim",
                        message=f"{resolve_type} scope exclusion claim lacks excluded evidence",
                        metadata={
                            "resolve_type": resolve_type,
                            "report_label": report_label,
                        },
                    )
                )
        if _contains_template_token(normalized, _REPORT_MISSING_TOKENS, label):
            report_state = business_facts.report_state(resolve_type)  # type: ignore[arg-type]
            if report_state is None or report_state.contains_strategy is not False:
                issues.append(
                    ValidationIssue(
                        code="unsupported_report_content_claim",
                        message=f"{resolve_type} non-inclusion claim lacks negative evidence",
                        metadata={
                            "resolve_type": resolve_type,
                            "report_label": report_label,
                        },
                    )
                )
    return issues


def _matching_action_intent(action, candidates: list[ActionIntentSpec]) -> ActionIntentSpec | None:
    for candidate in candidates:
        if action.type != candidate.action_type:
            continue
        candidate_strategy = getattr(candidate, "strategy", None)
        action_strategy = getattr(action, "strategy", None)
        if candidate_strategy and action_strategy and candidate_strategy != action_strategy:
            continue
        candidate_scope = getattr(candidate, "report_scope", "none")
        action_scope = getattr(action, "report_scope", "none")
        if candidate_scope != "none" and action_scope != candidate_scope:
            continue
        return candidate
    return None


def _has_document_context_evidence(evidence_facts: list[EvidenceFact]) -> bool:
    return any(
        fact.fact_type == "document_context"
        and fact.source_type == "document_mcp"
        and bool(fact.value)
        for fact in evidence_facts
    )


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
        if material_type is None or action.material_type == material_type:
            return True
    return False


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
