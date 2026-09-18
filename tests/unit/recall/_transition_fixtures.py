from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    RecallModeV1,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaCandidate,
    KnowledgeQaMatch,
)
from market_support_crewai_agent.runtime.recall.question_models import (
    QuestionRecallCandidate,
    QuestionRecallDecision,
    QuestionRecallMatch,
    QuestionRecallStatus,
)
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
)
from market_support_crewai_agent.runtime.recall.turn_state import (
    ApprovedStaticShortcutPayloadV1,
    recall_evidence_text_hash,
    recall_reply_text_hash,
)
from tests.helpers.reply_contract_requests import make_v2_envelope


class StaticCollectorProtocol(Protocol):
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> ApprovedStaticRecallCollectionV1: ...


class DocumentCollectorProtocol(Protocol):
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> KnowledgeQaMatch: ...


class StaticCollector:
    def __init__(self, result: ApprovedStaticRecallCollectionV1) -> None:
        self.result: ApprovedStaticRecallCollectionV1 = result
        self.calls: int = 0

    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> ApprovedStaticRecallCollectionV1:
        del request, policy
        self.calls += 1
        return self.result


class DocumentCollector:
    def __init__(self, result: KnowledgeQaMatch) -> None:
        self.result: KnowledgeQaMatch = result
        self.calls: int = 0

    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> KnowledgeQaMatch:
        del request, policy
        self.calls += 1
        return self.result


@dataclass(frozen=True, slots=True)
class RecallRuntime:
    question_recall_service: StaticCollectorProtocol
    qa_search_service: DocumentCollectorProtocol


def policy(
    mode: RecallModeV1,
) -> tuple[PolicyManifestV2, BusinessScopeAuthorityV1]:
    envelope = make_v2_envelope("公司网址是什么？")
    scope = business_scope_authority_v1(envelope.request.business_scope)
    core = compile_policy_authority_core_v1(
        envelope.request,
        scope,
        group_recall_mode=mode,
    )
    return PolicyManifestV2.from_core(core, policy_ledger_summary_v1((), 0)), scope


def direct_policy() -> tuple[PolicyManifestV2, BusinessScopeAuthorityV1]:
    envelope = make_v2_envelope(
        "公司网址是什么？",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:recall",
            "direct_thread_ref": "direct:recall",
            "principal_ref": "principal:recall",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "recall",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
    scope = business_scope_authority_v1(envelope.request.business_scope)
    core = compile_policy_authority_core_v1(envelope.request, scope)
    return PolicyManifestV2.from_core(core, policy_ledger_summary_v1((), 0)), scope


def candidate(
    *,
    entry_id: str = "company_basic_contact",
    canonical_id: str = "approved_static_knowledge.company_basic_contact",
    confidence: float = 0.91,
) -> QuestionRecallCandidate:
    return QuestionRecallCandidate(
        entry_id=entry_id,
        canonical_id=canonical_id,
        doc_id=entry_id,
        question="公司网址是什么？",
        source_type="approved_static_knowledge",
        score=confidence,
        confidence=confidence,
        coverage=1.0,
        answer_available=True,
    )


def static_match(
    *,
    decision: QuestionRecallDecision = "recall_hit",
    status: QuestionRecallStatus = "matched",
    candidates: list[QuestionRecallCandidate] | None = None,
) -> QuestionRecallMatch:
    selected = [candidate()] if candidates is None else candidates
    is_hit = decision == "recall_hit"
    return QuestionRecallMatch(
        status=status,
        decision=decision,
        confidence=selected[0].confidence if selected else 0.0,
        candidates=selected,
        reason_code=decision,
        reply_text="公司网址是 https://example.test。" if is_hit else "",
        evidence_text=(
            "Q：公司网址是什么？\nA：公司网址是 https://example.test。"
            if is_hit
            else ""
        ),
        source_id="company_basic_contact" if is_hit else "",
        source_type="approved_static_knowledge" if is_hit else None,
        trace={"selector_output": "must-not-reach-planner"},
    )


def static_no_match() -> ApprovedStaticRecallCollectionV1:
    return ApprovedStaticRecallCollectionV1(
        match=static_match(decision="fail_open", status="no_match", candidates=[]),
    )


def document_no_match() -> KnowledgeQaMatch:
    return KnowledgeQaMatch(status="no_match", reason_code="no_match")


def document_candidate() -> KnowledgeQaMatch:
    return KnowledgeQaMatch(
        status="matched",
        candidates=[
            KnowledgeQaCandidate(
                doc_id="file:///private/provider-locator",
                question="公司官网吗？",
                score=0.99,
            )
        ],
    )


def payload(
    ref: ManifestRefV1,
    *,
    confidence: float = 0.91,
) -> ApprovedStaticShortcutPayloadV1:
    reply_text = "公司网址是 https://example.test。"
    evidence_text = "Q：公司网址是什么？\nA：公司网址是 https://example.test。"
    values = {
        "candidate_id": "company_basic_contact",
        "canonical_id": "approved_static_knowledge.company_basic_contact",
        "selected_manifest_ref": ref,
        "fact_type": "document_context",
        "selected_media_asset_ids": (),
        "confidence": confidence,
        "question": "公司网址是什么？",
        "reply_text": reply_text,
        "evidence_text": evidence_text,
        "source_id": "company_basic_contact",
        "reply_text_hash": recall_reply_text_hash(reply_text),
        "evidence_text_hash": recall_evidence_text_hash(evidence_text),
    }
    if confidence >= 0.72:
        return ApprovedStaticShortcutPayloadV1.model_validate(values)
    return ApprovedStaticShortcutPayloadV1.model_construct(
        candidate_id="company_basic_contact",
        canonical_id="approved_static_knowledge.company_basic_contact",
        selected_manifest_ref=ref,
        fact_type="document_context",
        selected_media_asset_ids=(),
        confidence=confidence,
        question="公司网址是什么？",
        reply_text=reply_text,
        evidence_text=evidence_text,
        source_id="company_basic_contact",
        reply_text_hash=recall_reply_text_hash(reply_text),
        evidence_text_hash=recall_evidence_text_hash(evidence_text),
    )
