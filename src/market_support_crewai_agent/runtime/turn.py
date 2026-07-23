from __future__ import annotations

from dataclasses import dataclass

from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.planning import (
    ExecutionPlan,
    PlanValidationResult,
)
from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
    AdapterPreflightService,
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.evidence.document_mcp import (
    DocumentMcpEvidenceService,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.report_scope import (
    ReportScopeEvidenceService,
)
from market_support_crewai_agent.runtime.evidence.models import EvidenceFact
from market_support_crewai_agent.runtime.knowledge.approved_knowledge import (
    ApprovedKnowledgeEvidenceService,
)
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectOutboundDraft,
)
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.state.action_ledger import (
    ActionLedger,
    get_action_ledger,
)
from market_support_crewai_agent.runtime.state.audit import AuditStore, get_audit_store
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.validation.answerability import (
    AnswerabilityAssessment,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerifier,
)
from market_support_crewai_agent.runtime.validation.reply_validator import (
    ValidationResult,
)
from market_support_crewai_agent.schemas import ReplyResponse
from market_support_crewai_agent.settings import Settings, get_settings


class AgentRuntimeError(RuntimeError):
    """Raised when the CrewAI runtime cannot produce a valid reply."""


_APP_SETTINGS = get_settings()
_APP_CONVERSATION_STORE = ConversationStore.from_settings(_APP_SETTINGS)
_APP_ACTION_LEDGER = get_action_ledger()
_APP_ADAPTER_PREFLIGHT = AdapterPreflightService(settings=_APP_SETTINGS)
_APP_AUDIT_STORE = get_audit_store()
_APP_DOCUMENT_EVIDENCE_SERVICE = DocumentMcpEvidenceService(_APP_SETTINGS)
_APP_APPROVED_KNOWLEDGE_EVIDENCE_SERVICE = ApprovedKnowledgeEvidenceService(
    settings=_APP_SETTINGS
)
_APP_REPORT_SCOPE_EVIDENCE_SERVICE = ReportScopeEvidenceService(_APP_SETTINGS)


@dataclass(frozen=True, slots=True)
class RuntimeDeps:
    settings: Settings
    conversation_store: ConversationStore
    action_ledger: ActionLedger
    preflight_service: AdapterPreflightService
    evidence_executor: EvidenceExecutor
    audit_store: AuditStore
    alignment_verifier: ReplyAlignmentVerifier | None = None


@dataclass(frozen=True, slots=True)
class AttemptResult:
    plan: ExecutionPlan
    plan_validation: PlanValidationResult
    preflight: AdapterPreflightSnapshot
    evidence_facts: list[EvidenceFact]
    business_facts: BusinessFacts
    domain_context: DomainContext
    answerability: AnswerabilityAssessment
    directive: ResponseDirective
    response: ReplyResponse
    reply_validation: ValidationResult
    guardrail_decisions: list[GuardrailDecision]
    composer_output: ComposerReplyOutput | None = None
    pending_outbound_draft: DirectOutboundDraft | None = None


def build_runtime_deps(
    *,
    settings: Settings | None = None,
    conversation_store: ConversationStore | None = None,
    action_ledger: ActionLedger | None = None,
    preflight_service: AdapterPreflightService | None = None,
    evidence_executor: EvidenceExecutor | None = None,
    audit_store: AuditStore | None = None,
    alignment_verifier: ReplyAlignmentVerifier | None = None,
) -> RuntimeDeps:
    use_app_singletons = settings is None
    resolved_settings = settings or _APP_SETTINGS
    resolved_preflight_service = preflight_service or (
        _APP_ADAPTER_PREFLIGHT
        if use_app_singletons
        else AdapterPreflightService(settings=resolved_settings)
    )

    return RuntimeDeps(
        settings=resolved_settings,
        conversation_store=conversation_store
        if conversation_store is not None
        else (
            _APP_CONVERSATION_STORE
            if use_app_singletons
            else ConversationStore.from_settings(resolved_settings)
        ),
        action_ledger=action_ledger or _APP_ACTION_LEDGER,
        preflight_service=resolved_preflight_service,
        evidence_executor=evidence_executor
        if evidence_executor is not None
        else EvidenceExecutor(
            resolved_preflight_service,
            _APP_DOCUMENT_EVIDENCE_SERVICE
            if use_app_singletons
            else DocumentMcpEvidenceService(resolved_settings),
            _APP_APPROVED_KNOWLEDGE_EVIDENCE_SERVICE
            if use_app_singletons
            else ApprovedKnowledgeEvidenceService(settings=resolved_settings),
            _APP_REPORT_SCOPE_EVIDENCE_SERVICE
            if use_app_singletons
            else ReportScopeEvidenceService(settings=resolved_settings),
        ),
        audit_store=audit_store or _APP_AUDIT_STORE,
        alignment_verifier=alignment_verifier,
    )
