from __future__ import annotations

from typing import override

import pytest

from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaMatch,
    QaCorpusShapeError,
)
from market_support_crewai_agent.runtime.recall.flow import collect_preplanner_recall
from market_support_crewai_agent.runtime.recall.question_models import (
    QuestionRecallCandidate,
)
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
)
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.unit.recall._transition_fixtures import (
    DocumentCollector,
    RecallRuntime,
    StaticCollector,
    document_no_match,
    static_match,
    static_no_match,
)
from tests.unit.recall._transition_fixtures import (
    policy as make_policy,
)


class _ExplodingStaticCollector(StaticCollector):
    calls: int

    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> ApprovedStaticRecallCollectionV1:
        del request, policy
        self.calls += 1
        raise TimeoutError


class _MalformedCorpusCollector(DocumentCollector):
    calls: int

    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> KnowledgeQaMatch:
        del request, policy
        self.calls += 1
        raise QaCorpusShapeError


class _UnexpectedRecallDefect(RuntimeError):
    pass


class _DefectiveStaticCollector(StaticCollector):
    calls: int

    @override
    async def collect(
        self,
        request: KernelReplyRequestV1,
        policy: PolicyManifestV2,
    ) -> ApprovedStaticRecallCollectionV1:
        del request, policy
        self.calls += 1
        raise _UnexpectedRecallDefect("unexpected recall defect")


@pytest.mark.anyio
async def test_all_invalid_items_emit_invalid_unavailable_row() -> None:
    # Given: one static candidate that cannot cross the bounded candidate schema.
    policy, scope_authority = make_policy("advisory")
    invalid = QuestionRecallCandidate.model_construct(
        entry_id="invalid/id",
        canonical_id="approved_static_knowledge.invalid",
        doc_id="invalid",
        question="invalid",
        source_type="approved_static_knowledge",
        score=0.8,
        confidence=0.8,
        coverage=1.0,
        matched_entities=(),
        answer_available=True,
    )
    static = StaticCollector(
        ApprovedStaticRecallCollectionV1(
            match=static_match(
                decision="recall_hint",
                status="candidates",
                candidates=[invalid],
            )
        )
    )
    document = DocumentCollector(document_no_match())

    # When: the invalid-only branch is classified.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: no candidate leaks and the invalid branch fails open as unavailable.
    branch = transition.turn_state.outcome.branch_outcomes[0]
    assert transition.turn_state.outcome.decision == "unavailable"
    assert (branch.status, branch.reason_code) == ("invalid", "source_invalid")
    assert (branch.accepted_count, branch.rejected_count) == (0, 1)
    assert (static.calls, document.calls) == (1, 1)


@pytest.mark.anyio
async def test_provider_exception_is_sanitized_and_other_branch_still_runs() -> None:
    # Given: a static provider timeout and an available document provider.
    policy, scope_authority = make_policy("shortcut")
    static = _ExplodingStaticCollector(static_no_match())
    document = DocumentCollector(document_no_match())

    # When: recall evaluates both fail-open branches.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: only the typed unavailable reason survives and collection continues.
    branch = transition.turn_state.outcome.branch_outcomes[0]
    assert transition.turn_state.outcome.decision == "unavailable"
    assert (branch.status, branch.reason_code) == (
        "unavailable",
        "source_unavailable",
    )
    assert (static.calls, document.calls) == (1, 1)


@pytest.mark.anyio
async def test_malformed_corpus_is_sanitized_as_document_unavailable() -> None:
    # Given: static recall misses and Document MCP returns a malformed corpus shape.
    policy, scope_authority = make_policy("advisory")
    static = StaticCollector(static_no_match())
    document = _MalformedCorpusCollector(document_no_match())

    # When: pre-planner recall crosses the corpus boundary.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: the explicit corpus failure is bounded without masking the static branch.
    branch = transition.turn_state.outcome.branch_outcomes[1]
    assert transition.turn_state.outcome.decision == "unavailable"
    assert (branch.status, branch.reason_code) == (
        "unavailable",
        "source_unavailable",
    )
    assert (static.calls, document.calls) == (1, 1)


@pytest.mark.anyio
async def test_unexpected_collector_defect_propagates() -> None:
    # Given: collector implementation code raises an unrepresented programming defect.
    policy, scope_authority = make_policy("advisory")
    static = _DefectiveStaticCollector(static_no_match())
    document = DocumentCollector(document_no_match())

    # When/Then: orchestration preserves the defect instead of fabricating unavailability.
    with pytest.raises(RuntimeError, match="unexpected recall defect"):
        _ = await collect_preplanner_recall(
            RecallRuntime(static, document),
            request=make_v2_envelope("公司网址是什么？").request,
            policy=policy,
            scope_authority=scope_authority,
        )
    assert (static.calls, document.calls) == (1, 0)
