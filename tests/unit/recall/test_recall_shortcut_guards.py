from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.recall.flow import collect_preplanner_recall
from market_support_crewai_agent.runtime.recall.service import (
    ApprovedStaticRecallCollectionV1,
)
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.unit.recall._transition_fixtures import (
    DocumentCollector,
    RecallRuntime,
    StaticCollector,
    candidate,
    document_no_match,
    payload,
    static_match,
)
from tests.unit.recall._transition_fixtures import (
    policy as make_policy,
)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload_ref",
    (
        ManifestRefV1(
            manifest_id="general.handoff",
            manifest_version="2026-07-18.1",
        ),
        ManifestRefV1(
            manifest_id="weekly_report.send",
            manifest_version="2026-07-18.1",
        ),
    ),
)
async def test_shortcut_payload_fails_closed_when_ref_is_ineligible_or_wrong_contract(
    payload_ref: ManifestRefV1,
) -> None:
    # Given: a typed payload whose manifest cannot authorize document_context recall.
    policy, scope_authority = make_policy("shortcut")
    static = StaticCollector(
        ApprovedStaticRecallCollectionV1(
            match=static_match(),
            shortcut_payload=payload(payload_ref),
        )
    )
    document = DocumentCollector(document_no_match())

    # When: shortcut eligibility is checked against final compiled policy/registry.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: the hit remains advisory and the planner path stays required.
    assert (static.calls, document.calls) == (1, 1)
    assert transition.turn_state.outcome.decision == "advisory_candidates"
    assert transition.shortcut_plan is None


@pytest.mark.anyio
async def test_shortcut_payload_fails_closed_below_exact_threshold() -> None:
    # Given: a model-constructed regression payload below the 0.72 invariant.
    policy, scope_authority = make_policy("shortcut")
    ref = next(
        ref
        for ref in policy.eligible_capabilities
        if ref.manifest_id == "answer_internal_company_knowledge"
    )
    static = StaticCollector(
        ApprovedStaticRecallCollectionV1(
            match=static_match(candidates=[candidate(confidence=0.71)]),
            shortcut_payload=payload(ref, confidence=0.71),
        )
    )
    document = DocumentCollector(document_no_match())

    # When: shortcut eligibility is evaluated.
    transition = await collect_preplanner_recall(
        RecallRuntime(static, document),
        request=make_v2_envelope("公司网址是什么？").request,
        policy=policy,
        scope_authority=scope_authority,
    )

    # Then: exact-threshold enforcement prevents a synthesized plan.
    assert (static.calls, document.calls) == (1, 1)
    assert transition.turn_state.outcome.decision == "advisory_candidates"
    assert transition.shortcut_plan is None
