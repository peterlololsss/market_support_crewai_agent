from __future__ import annotations

from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
    finalize_execution_plan_v2,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    RecallModeV1,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaCandidate,
    KnowledgeQaMatch,
)
from market_support_crewai_agent.runtime.recall.question_models import (
    QuestionRecallCandidate,
    QuestionRecallMatch,
)
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
)
from market_support_crewai_agent.runtime.recall.turn_state import (
    ApprovedStaticShortcutPayloadV1,
    RecallTurnStateV1,
    recall_evidence_text_hash,
    recall_reply_text_hash,
)
from market_support_crewai_agent.runtime.v2_attempt import (
    V2AttemptResult,
    V2ReplyValidationResult,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from tests.helpers.reply_contract_requests import make_v2_envelope

PipelineContext = tuple[
    VerifiedRequestEnvelopeV1,
    PolicyManifestV2,
    BusinessScopeAuthorityV1,
]


def group_context(mode: RecallModeV1) -> PipelineContext:
    envelope = make_v2_envelope("公司网址是什么？")
    scope = business_scope_authority_v1(envelope.request.business_scope)
    core = compile_policy_authority_core_v1(
        envelope.request,
        scope,
        group_recall_mode=mode,
    )
    policy = PolicyManifestV2.from_core(core, policy_ledger_summary_v1((), 0))
    return envelope, policy, scope


def direct_context() -> PipelineContext:
    envelope = make_v2_envelope(
        "公司网址是什么？",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:pipeline-recall",
            "direct_thread_ref": "direct:pipeline-recall",
            "principal_ref": "principal:pipeline-recall",
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
    policy = PolicyManifestV2.from_core(core, policy_ledger_summary_v1((), 0))
    return envelope, policy, scope


def static_hit(
    policy: PolicyManifestV2,
    *,
    with_payload: bool,
    answer_available: bool = True,
) -> ApprovedStaticRecallCollectionV1:
    candidate = QuestionRecallCandidate(
        entry_id="company_basic_contact",
        canonical_id="approved_static_knowledge.company_basic_contact",
        doc_id="company_basic_contact",
        question="公司网址是什么？",
        source_type="approved_static_knowledge",
        score=0.91,
        confidence=0.91,
        coverage=1.0,
        answer_available=answer_available,
    )
    reply_text = "公司网址是 https://example.test。"
    evidence_text = "Q：公司网址是什么？\nA：公司网址是 https://example.test。"
    match = QuestionRecallMatch(
        status="matched",
        decision="recall_hit",
        confidence=0.91,
        candidates=[candidate],
        reason_code="recall_hit",
        reply_text=reply_text,
        evidence_text=evidence_text,
        source_id="company_basic_contact",
        source_type="approved_static_knowledge",
        trace={"raw_selector_output": "never-visible"},
    )
    if not with_payload:
        return ApprovedStaticRecallCollectionV1(match=match)
    ref = next(
        ref
        for ref in policy.eligible_capabilities
        if ref.manifest_id == "answer_internal_company_knowledge"
    )
    payload = ApprovedStaticShortcutPayloadV1(
        candidate_id=candidate.entry_id,
        canonical_id=candidate.canonical_id,
        selected_manifest_ref=ref,
        selected_media_asset_ids=(),
        confidence=candidate.confidence,
        question=candidate.question,
        reply_text=reply_text,
        evidence_text=evidence_text,
        source_id=candidate.entry_id,
        reply_text_hash=recall_reply_text_hash(reply_text),
        evidence_text_hash=recall_evidence_text_hash(evidence_text),
    )
    return ApprovedStaticRecallCollectionV1(match=match, shortcut_payload=payload)


def document_candidate() -> KnowledgeQaMatch:
    return KnowledgeQaMatch(
        status="matched",
        candidates=[
            KnowledgeQaCandidate(
                doc_id="file:///provider/private",
                question="公司官网吗？",
                score=0.99,
            )
        ],
    )


def planner_attempt_result(
    *,
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
    recall_state: RecallTurnStateV1,
) -> V2AttemptResult:
    plan = finalize_execution_plan_v2(
        DeterministicPlanOriginInputV1(
            user_need="planner recall runtime result",
            units=(
                DeterministicPlanUnitV1(
                    unit_id="planner-recall",
                    manifest_id="answer_internal_company_knowledge",
                    answerability_policy="answer",
                    evidence_query=request.message,
                ),
            ),
            compliance_reason_code="compliant_product_request",
            confidence=0.9,
        ),
        policy,
        scope_authority,
        origin="deterministic",
    )
    return V2AttemptResult(
        plan=plan,
        evidence=CanonicalEvidenceExecutionResultV1(
            preflight=AdapterPreflightSnapshot.empty(),
            canonical_facts=(),
            resolve_bindings=(),
            groundings=(),
            domain_context=DomainContextV1Builder().build(
                request,
                scope_authority=scope_authority,
            ),
        ),
        response=ReplyResponse(
            reply=PrimaryReply(kind="answer", text="planner result"),
            actions=[],
        ),
        reply_validation=V2ReplyValidationResult(valid=True),
        reason_code="compliant_product_request",
        recall_state=recall_state,
    )
