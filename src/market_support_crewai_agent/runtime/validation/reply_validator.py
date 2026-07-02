from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.knowledge.approved_knowledge import approved_image_markers
from market_support_crewai_agent.runtime.domain.capabilities import (
    resolve_type_for_action,
)
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import ActionIntentSpec, ExecutionPlan
from market_support_crewai_agent.runtime.domain.planning import plan_spec_for_execution_plan
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.domain.sources.precedence import (
    plan_has_knowledge_evidence,
    select_evidence_for_capabilities,
)
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.runtime.validation.plan_spec_verifier import (
    verify_plan_spec,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    image_marker_filenames,
    marker_in_trusted_document_context,
    trusted_document_context,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import GuardrailDecision
from market_support_crewai_agent.runtime.validation.output_guard import (
    output_guard as evaluate_output_guard,
)
from market_support_crewai_agent.schemas import ReplyResponse

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
    "unsupported_evidence_claim",
    "image_marker_not_allowed",
    "image_marker_not_in_evidence",
    "sent_claim_without_ledger_evidence",
    "unsupported_report_scope_claim",
    "unsupported_report_content_claim",
    "plan_spec_contract_failed",
]

_SEVERITY_RANK: dict[ValidationSeverity, int] = {
    "info": 0,
    "warning": 1,
    "error": 2,
    "fatal": 3,
}
_RAW_LOCATOR_TOKENS = (
    "file:",
    "mcp://",
    "wecom-adapter:",
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
_CLAIM_UNIT_SPLIT_RE = re.compile(r"((?:[。！？!?]\s*)|\n+)")


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
    domain_context: DomainContext | None = None,
    composer_output: ComposerReplyOutput | None = None,
    output_decision: GuardrailDecision | None = None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    issues.extend(_validate_policy_and_kind(response, directive, policy))
    issues.extend(_validate_no_reply(response))
    issues.extend(_validate_locator_leaks(response))
    issues.extend(_validate_non_compliant_response(response, directive, plan))
    issues.extend(_validate_handoff(response, directive, business_facts))
    issues.extend(_validate_actions(response, directive, business_facts, policy))
    issues.extend(
        _validate_knowledge_grounding(
            response,
            directive,
            plan,
            evidence_facts,
            domain_context,
            composer_output,
            output_decision,
        )
    )
    issues.extend(_validate_image_markers(response.reply.text, evidence_facts))
    issues.extend(
        _validate_plan_spec_contract(
            response,
            directive,
            plan,
            evidence_facts,
            domain_context,
            composer_output,
        )
    )
    return ValidationResult(valid=not issues, issues=tuple(issues))


def remove_pre_execution_send_claims(text: str) -> str:
    """Strip composer text that claims outbound actions already executed."""
    if not _first_completed_send_claim_token(str(text or "").lower()):
        return text

    units = _text_units(text)
    kept: list[str] = []
    removed = False
    for unit in units:
        if _first_completed_send_claim_token(unit.lower()):
            removed = True
            continue
        kept.append(unit)

    if not removed:
        return text
    return re.sub(r"\n{3,}", "\n\n", "".join(kept)).strip()


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
        if action_type not in policy.allowed_outbound_actions:
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

    if response.actions and response.reply.text.strip():
        directive_text = directive.text.strip()
        composer_text_allowed = (
            directive.requires_knowledge_composer
            and directive.composer_stage == "knowledge_composer"
        )
        unexpected_text = (
            response.reply.text.strip() != directive_text
            if directive_text
            else not composer_text_allowed
        )
    else:
        unexpected_text = False
    if unexpected_text:
        issues.append(
            ValidationIssue(
                code="outbound_action_reply_text_not_empty",
                message="outbound action reply.text must be empty unless supplied by the directive",
            )
        )
    if response.actions and response.reply.mentions:
        issues.append(
            ValidationIssue(
                code="outbound_action_reply_mentions_not_empty",
                message="outbound action responses must leave reply.mentions empty",
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
                message="outbound action must include resolve_ref",
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


def _validate_knowledge_grounding(
    response: ReplyResponse,
    directive: ResponseDirective,
    plan: ExecutionPlan,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
    composer_output: ComposerReplyOutput | None = None,
    output_decision: GuardrailDecision | None = None,
) -> list[ValidationIssue]:
    if (
        directive.mode != "knowledge_answer"
        and directive.composer_stage != "knowledge_composer"
    ):
        return []
    if response.reply.kind != "answer" or not response.reply.text.strip():
        return []
    if output_decision is None:
        output_decision = evaluate_output_guard(
            response=response,
            directive=directive,
            plan=plan,
            policy=PolicyManifest(
                policy_id="reply-validation-inline",
                allowed_reply_modes=frozenset(),
                allowed_capabilities=frozenset(),
                allowed_outbound_actions=frozenset(),
                allowed_read_capabilities=frozenset(),
                allowed_adapter_resolves=frozenset(),
            ),
            evidence_facts=evidence_facts,
            domain_context=domain_context,
            composer_output=composer_output,
        )
    if output_decision.reason_code in {
        "unsupported_product_claim",
        "composer_evidence_ids_missing",
        "composer_evidence_id_not_allowed",
    } or (
        output_decision.outcome != "allow"
        and composer_output is not None
        and (composer_output.claims or composer_output.evidence_ids)
    ):
        return [
            ValidationIssue(
                code="unsupported_evidence_claim",
                message=output_decision.human_readable_reason,
                metadata=output_decision.metadata,
            )
        ]
    if _has_knowledge_answer_evidence(plan, directive, evidence_facts, domain_context):
        return []
    return [
        ValidationIssue(
            code="knowledge_answer_without_document_evidence",
            message="knowledge answer requires document_context or report_scope evidence",
        )
    ]


def _validate_image_markers(
    text: str,
    evidence_facts: list[EvidenceFact],
) -> list[ValidationIssue]:
    markers = image_marker_filenames(text)
    if not markers:
        return []
    issues: list[ValidationIssue] = []
    for filename in markers:
        marker = f"%%{filename}%%"
        if filename not in allowed_image_markers():
            issues.append(
                ValidationIssue(
                    code="image_marker_not_allowed",
                    message="reply text contains an image marker outside the whitelist",
                    severity="fatal",
                    metadata={"filename": filename},
                )
            )
            continue
        if not marker_in_trusted_document_context(marker, evidence_facts):
            issues.append(
                ValidationIssue(
                    code="image_marker_not_in_evidence",
                    message="reply text image marker must appear in document evidence",
                    severity="fatal",
                    metadata={"filename": filename},
                )
            )
    return issues


def allowed_image_markers() -> frozenset[str]:
    return approved_image_markers()

def _validate_plan_spec_contract(
    response: ReplyResponse,
    directive: ResponseDirective,
    plan: ExecutionPlan,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None,
    composer_output: ComposerReplyOutput | None,
) -> list[ValidationIssue]:
    plan_spec = plan_spec_for_execution_plan(plan, domain_context=domain_context)
    if not directive.action_intents:
        answer_units = [
            unit for unit in plan_spec.plan_units if unit.answerability_policy != "send"
        ]
        if answer_units:
            plan_spec = plan_spec.model_copy(update={"plan_units": answer_units})
    result = verify_plan_spec(
        plan_spec,
        output_payload=response,
        evidence_facts=evidence_facts,
        cited_evidence_ids=(
            composer_output.evidence_ids if composer_output is not None else None
        ),
        abstained=response.reply.kind
        in {"unable_to_answer", "clarification", "human_handoff", "no_reply"},
    )
    if result.valid:
        return []
    return [
        ValidationIssue(
            code="plan_spec_contract_failed",
            message=issue.message,
            severity=issue.severity,
            metadata={
                "plan_id": plan_spec.plan_id,
                "selected_capability_ids": [
                    unit.selected_capability_id for unit in plan_spec.plan_units
                ],
                "contract_issue_code": issue.code,
                **issue.metadata,
            },
        )
        for issue in result.issues
    ]


def _matching_action_intent(action, candidates: list[ActionIntentSpec]) -> ActionIntentSpec | None:
    for candidate in candidates:
        if action.type != candidate.action_type:
            continue
        candidate_option = getattr(candidate, "material_pack_option", None)
        action_option = getattr(action, "material_pack_option", None)
        if candidate_option and action_option and candidate_option != action_option:
            continue
        return candidate
    return None


def _has_document_context_evidence(evidence_facts: list[EvidenceFact]) -> bool:
    return any(trusted_document_context(fact) for fact in evidence_facts)


def _has_knowledge_answer_evidence(
    plan: ExecutionPlan,
    directive: ResponseDirective,
    evidence_facts: list[EvidenceFact],
    domain_context: DomainContext | None = None,
) -> bool:
    if plan_has_knowledge_evidence(plan, evidence_facts, domain_context):
        return True
    if plan.answer_capabilities:
        return False
    return select_evidence_for_capabilities(
        tuple(action.capability for action in directive.action_intents),
        evidence_facts,
    ).has_evidence


def _first_completed_send_claim_token(normalized_text: str) -> str:
    for token in _COMPLETED_SEND_CLAIM_TOKENS:
        if token.lower() in normalized_text:
            return token
    return ""


def _text_units(text: str) -> list[str]:
    parts = _CLAIM_UNIT_SPLIT_RE.split(str(text or ""))
    units: list[str] = []
    for index in range(0, len(parts), 2):
        unit = parts[index]
        if index + 1 < len(parts):
            unit += parts[index + 1]
        if unit:
            units.append(unit)
    return units


def _contains_raw_locator(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(token in normalized for token in _RAW_LOCATOR_TOKENS)
