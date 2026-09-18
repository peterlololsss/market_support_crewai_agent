from __future__ import annotations

from collections.abc import Callable

import pytest

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    RecallModeV1,
)
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    KnowledgeQaMatch,
)
from market_support_crewai_agent.runtime.recall.flow import (
    collect_preplanner_recall,
)
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
    candidate,
    direct_policy,
    document_candidate,
    document_no_match,
    payload,
    static_match,
    static_no_match,
)
from tests.unit.recall._transition_fixtures import (
    policy as make_policy,
)

type PolicyFactory = Callable[
    [RecallModeV1], tuple[PolicyManifestV2, BusinessScopeAuthorityV1]
]


def _direct_policy_ignoring_mode(
    mode: RecallModeV1,
) -> tuple[PolicyManifestV2, BusinessScopeAuthorityV1]:
    del mode
    return direct_policy()


@pytest.mark.anyio
@pytest.mark.parametrize("policy_factory", (make_policy, _direct_policy_ignoring_mode))
async def test_off_mode_makes_zero_preplanner_provider_calls(
    policy_factory: PolicyFactory,
) -> None:
    # Given: group/direct compiled policy with recall disabled and armed collectors.
    policy, scope_authority = policy_factory("off")
    envelope = make_v2_envelope("公司网址是什么？")
    static = StaticCollector(static_no_match())
    document = DocumentCollector(document_no_match())

    # When: pre-planner recall runs from the compiled policy.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=envelope.request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: both providers stay untouched and the exhaustive off row is emitted.
    assert (static.calls, document.calls) == (0, 0)
    assert transition.turn_state.outcome.decision == "disabled"
    assert tuple(
        (
            branch.status,
            branch.reason_code,
            branch.accepted_count,
            branch.rejected_count,
        )
        for branch in transition.turn_state.outcome.branch_outcomes
    ) == (
        ("not_called", "disabled", 0, 0),
        ("not_called", "disabled", 0, 0),
    )
    assert transition.shortcut_plan is None


@pytest.mark.anyio
async def test_shortcut_mode_uses_only_valid_approved_static_payload() -> None:
    # Given: one policy-eligible, threshold-passing approved-static payload.
    policy, scope_authority = make_policy("shortcut")
    ref = next(
        ref
        for ref in policy.eligible_capabilities
        if ref.manifest_id == "answer_internal_company_knowledge"
    )
    static = StaticCollector(
        ApprovedStaticRecallCollectionV1(
            match=static_match(),
            shortcut_payload=payload(ref),
        )
    )
    document = DocumentCollector(document_candidate())

    # When: the pre-planner transition is evaluated.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: the static plan is validated and Document MCP is never consulted.
    assert (static.calls, document.calls) == (1, 0)
    assert transition.turn_state.outcome.decision == "shortcut_match"
    assert transition.shortcut_plan is not None
    assert transition.shortcut_plan.origin == "approved_static_shortcut"
    assert transition.shortcut_plan.selected_manifest_refs == (ref,)
    assert transition.shortcut_plan.confidence == 0.91
    assert tuple(
        (branch.status, branch.reason_code, branch.accepted_count)
        for branch in transition.turn_state.outcome.branch_outcomes
    ) == (("ok", "ok", 1), ("not_called", "not_needed", 0))


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ("advisory", "shortcut"))
async def test_non_shortcut_candidates_are_advisory_and_call_both_branches(
    mode: RecallModeV1,
) -> None:
    # Given: an unbound static hit and a high-scoring Document MCP candidate.
    policy, scope_authority = make_policy(mode)
    static = StaticCollector(
        ApprovedStaticRecallCollectionV1(
            match=static_match(),
            shortcut_payload=None,
        )
    )
    document = DocumentCollector(document_candidate())

    # When: recall collects the provider results.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: both bounded candidates are advisory and no plan is synthesized.
    assert (static.calls, document.calls) == (1, 1)
    assert transition.turn_state.outcome.decision == "advisory_candidates"
    assert transition.shortcut_plan is None
    assert {
        candidate.source_class for candidate in transition.turn_state.outcome.candidates
    } == {
        "approved_static",
        "document_mcp",
    }
    visible = transition.turn_state.outcome.model_dump_json()
    assert "https://example.test" not in visible
    assert "file:///private" not in visible
    assert "selector_output" not in visible


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("static_result", "document_result", "decision", "branch_statuses"),
    (
        (
            static_no_match(),
            document_no_match(),
            "no_match",
            ("ok", "ok"),
        ),
        (
            ApprovedStaticRecallCollectionV1(
                match=static_match(
                    decision="fail_open",
                    status="unavailable",
                    candidates=[],
                ),
                shortcut_payload=None,
            ),
            document_no_match(),
            "unavailable",
            ("unavailable", "ok"),
        ),
    ),
)
async def test_empty_candidate_rows_distinguish_clean_miss_from_unavailable(
    static_result: ApprovedStaticRecallCollectionV1,
    document_result: KnowledgeQaMatch,
    decision: str,
    branch_statuses: tuple[str, str],
) -> None:
    # Given: two called branches with no valid candidates.
    policy, scope_authority = make_policy("advisory")
    static = StaticCollector(static_result)
    document = DocumentCollector(document_result)

    # When: the exhaustive no-candidate transition is evaluated.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: provider availability determines the final typed decision.
    assert (static.calls, document.calls) == (1, 1)
    assert transition.turn_state.outcome.decision == decision
    assert (
        tuple(branch.status for branch in transition.turn_state.outcome.branch_outcomes)
        == branch_statuses
    )
    assert transition.turn_state.outcome.candidates == ()


@pytest.mark.anyio
async def test_invalid_static_items_are_counted_without_blocking_valid_advisory() -> (
    None
):
    # Given: one valid candidate and one candidate invalid for the bounded view.
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
                candidates=[candidate(), invalid],
            ),
            shortcut_payload=None,
        )
    )
    document = DocumentCollector(document_no_match())

    # When: candidates cross the provider-neutral view boundary.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: the valid item remains advisory and the invalid item is accounted for.
    assert transition.turn_state.outcome.decision == "advisory_candidates"
    static_branch = transition.turn_state.outcome.branch_outcomes[0]
    assert (
        static_branch.status,
        static_branch.reason_code,
        static_branch.accepted_count,
        static_branch.rejected_count,
    ) == ("partial", "candidate_rejected", 1, 1)
