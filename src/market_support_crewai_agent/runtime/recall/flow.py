from __future__ import annotations

from typing import Protocol

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.document_mcp.parsing import (
    DocumentMcpError,
)
from market_support_crewai_agent.runtime.planning import (
    finalize_execution_plan_v2,
    validate_execution_plan_v2,
)
from market_support_crewai_agent.runtime.planning.compiler import (
    DeterministicPlanOriginInputV1,
    DeterministicPlanUnitV1,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaMatch,
    QaCorpusShapeError,
)
from market_support_crewai_agent.runtime.recall.outcome_models import (
    RecallBranchOutcomeV1,
    RecallShortcutSummaryV1,
)
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
    RecallBranchCollectionV1,
    approved_static_branch,
    document_recall_branch,
    make_recall_outcome,
    merge_recall_candidates,
)
from market_support_crewai_agent.runtime.recall.turn_state import (
    ApprovedStaticShortcutPayloadV1,
    PreplannerRecallTransitionV1,
    RecallTurnStateV1,
    disabled_recall_transition,
)


class _ApprovedStaticCollector(Protocol):
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> ApprovedStaticRecallCollectionV1: ...


class _DocumentCollector(Protocol):
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> KnowledgeQaMatch: ...


class PreplannerRecallRuntimeV1(Protocol):
    @property
    def question_recall_service(self) -> _ApprovedStaticCollector: ...

    @property
    def qa_search_service(self) -> _DocumentCollector: ...


async def collect_preplanner_recall(
    runtime: PreplannerRecallRuntimeV1,
    *,
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
) -> PreplannerRecallTransitionV1:
    if policy.recall_mode == "off":
        return disabled_recall_transition()
    static_branch, payload = await _collect_static(runtime, request, policy)
    if policy.recall_mode == "shortcut" and payload is not None:
        shortcut = _shortcut_transition(
            static_branch,
            payload,
            policy,
            scope_authority,
        )
        if shortcut is not None:
            return shortcut
    document_branch = await _collect_document(runtime, request, policy)
    branches = (static_branch.outcome(), document_branch.outcome())
    candidates = merge_recall_candidates(
        static_branch.candidates,
        document_branch.candidates,
    )
    if candidates:
        decision = "advisory_candidates"
    else:
        failed = any(branch.status in {"invalid", "unavailable"} for branch in branches)
        decision = "unavailable" if failed else "no_match"
    outcome = make_recall_outcome(
        policy.recall_mode,
        decision,
        candidates,
        branches,
    )
    return PreplannerRecallTransitionV1(
        turn_state=RecallTurnStateV1(outcome=outcome, shortcut_payload=None),
        shortcut_plan=None,
    )


async def _collect_static(
    runtime: PreplannerRecallRuntimeV1,
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
) -> tuple[RecallBranchCollectionV1, ApprovedStaticShortcutPayloadV1 | None]:
    try:
        collection = await runtime.question_recall_service.collect(request, policy)
    except (ConnectionError, DocumentMcpError, QaCorpusShapeError, TimeoutError):
        return RecallBranchCollectionV1("approved_static", unavailable=True), None
    return approved_static_branch(collection.match), collection.shortcut_payload


async def _collect_document(
    runtime: PreplannerRecallRuntimeV1,
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
) -> RecallBranchCollectionV1:
    try:
        match = await runtime.qa_search_service.collect(request, policy)
    except (ConnectionError, DocumentMcpError, QaCorpusShapeError, TimeoutError):
        return RecallBranchCollectionV1("document_mcp", unavailable=True)
    return document_recall_branch(match)


def _shortcut_transition(
    static_branch: RecallBranchCollectionV1,
    payload: ApprovedStaticShortcutPayloadV1,
    policy: PolicyManifestV2,
    scope_authority: BusinessScopeAuthorityV1,
) -> PreplannerRecallTransitionV1 | None:
    matching = tuple(
        candidate
        for candidate in static_branch.candidates
        if (
            candidate.candidate_id == payload.candidate_id
            and candidate.canonical_id == payload.canonical_id
            and candidate.confidence == payload.confidence
            and candidate.reason_code == "approved_recall_hit"
        )
    )
    if (
        payload.confidence < 0.72
        or len(matching) != 1
        or static_branch.rejected_count
        or not policy.internal_company_knowledge_enabled
        or payload.selected_manifest_ref not in policy.eligible_capabilities
    ):
        return None
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(
        payload.selected_manifest_ref.manifest_id
    )
    if (
        manifest is None
        or manifest.manifest_version != payload.selected_manifest_ref.manifest_version
        or payload.fact_type not in manifest.evidence_contract.allowed_fact_types
        or "approved_static_knowledge"
        not in manifest.evidence_contract.allowed_source_types
    ):
        return None
    source = DeterministicPlanOriginInputV1(
        user_need="answer approved static recall",
        units=(
            DeterministicPlanUnitV1(
                unit_id="approved-static-recall",
                manifest_id=payload.selected_manifest_ref.manifest_id,
                answerability_policy="answer",
                evidence_query=payload.question,
            ),
        ),
        compliance_reason_code="compliant_product_request",
        confidence=payload.confidence,
    )
    try:
        plan = finalize_execution_plan_v2(
            source,
            policy,
            scope_authority,
            origin="approved_static_shortcut",
        )
    except ValueError:
        return None
    if not validate_execution_plan_v2(plan, policy).valid:
        return None
    summary = RecallShortcutSummaryV1(
        candidate_id=payload.candidate_id,
        canonical_id=payload.canonical_id,
        source_id=payload.source_id,
        selected_manifest_ref=payload.selected_manifest_ref,
        confidence=payload.confidence,
        reply_text_hash=payload.reply_text_hash,
        evidence_text_hash=payload.evidence_text_hash,
    )
    branches = (
        RecallBranchOutcomeV1(
            source_class="approved_static",
            status="ok",
            accepted_count=1,
            rejected_count=0,
            reason_code="ok",
        ),
        RecallBranchOutcomeV1(
            source_class="document_mcp",
            status="not_called",
            accepted_count=0,
            rejected_count=0,
            reason_code="not_needed",
        ),
    )
    outcome = make_recall_outcome(
        "shortcut",
        "shortcut_match",
        matching,
        branches,
        summary,
    )
    return PreplannerRecallTransitionV1(
        turn_state=RecallTurnStateV1(
            outcome=outcome,
            shortcut_payload=payload,
        ),
        shortcut_plan=plan,
    )
