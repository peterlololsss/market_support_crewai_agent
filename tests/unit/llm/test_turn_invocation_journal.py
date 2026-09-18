from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    InvocationJournalError,
    SelectorCacheIdentityV1,
    TurnLlmInvocationJournalV1,
    TurnLlmInvocationRowV1,
    TurnSelectorOutcomeCacheV1,
    selector_input_hash,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    build_provider_target_from_settings,
)
from market_support_crewai_agent.settings_model import Settings


def test_configured_maximum_invocation_rows_uses_exact_shared_iteration_formula() -> (
    None
):
    # Given: default, maximum, and mixed configured remediation bounds.
    default = Settings()
    maximum = Settings(
        reply_alignment_max_replans=2,
        reply_alignment_max_evidence_refetches=2,
        reply_alignment_max_recomposes=2,
        reply_alignment_max_total_remediations=2,
    )
    mixed = Settings(
        reply_alignment_max_replans=0,
        reply_alignment_max_evidence_refetches=1,
        reply_alignment_max_recomposes=2,
        reply_alignment_max_total_remediations=2,
    )

    # When: production settings derive their maximum dispatch rows.
    rows = tuple(
        settings.maximum_llm_invocation_rows for settings in (default, maximum, mixed)
    )

    # Then: configured bounds produce exact nonconstant totals under the ceiling.
    assert rows == (16, 18, 13)
    assert max(rows) <= 18


def test_journal_reserves_gap_free_ordinals_and_rejects_nineteenth_before_io() -> None:
    # Given: the maximum valid configured bounds and their production capacity.
    settings = Settings(
        reply_alignment_max_replans=2,
        reply_alignment_max_evidence_refetches=2,
        reply_alignment_max_recomposes=2,
        reply_alignment_max_total_remediations=2,
    )
    journal = TurnLlmInvocationJournalV1(max_rows=settings.maximum_llm_invocation_rows)

    # When: the maximum governed dispatch count is reserved.
    for index in range(18):
        journal.reserve(
            stage_kind="planner_intent",
            program_id="planner_intent.wecom_group.v1@1",
            target_slot="planner",
            purpose="test",
            logical_attempt=index + 1,
        )

    # Then: ordinals are 1..18 and the 19th fails before provider I/O.
    assert [row.invocation_ordinal for row in journal.rows] == list(range(1, 19))
    with pytest.raises(
        InvocationJournalError,
        match="llm_invocation_journal_capacity_exceeded",
    ):
        journal.reserve(
            stage_kind="planner_intent",
            program_id="planner_intent.wecom_group.v1@1",
            target_slot="planner",
            purpose="overflow",
        )


def test_journal_schema_accepts_only_literal_transport_attempt_one() -> None:
    # Given/When/Then: Pydantic rejects an unmodeled second transport attempt.
    with pytest.raises(ValueError):
        TurnLlmInvocationRowV1.model_validate(
            {
                "invocation_ordinal": 1,
                "logical_attempt": 1,
                "transport_attempt": 2,
                "stage_kind": "planner_intent",
                "program_id": "planner_intent.wecom_group.v1@1",
                "target_slot": "planner",
                "purpose": "test",
                "status": "reserved",
            }
        )


def test_selector_cache_reuses_only_exact_sih1_hits_and_evicts_to_four() -> None:
    # Given: five distinct selector inputs and a four-entry cache.
    cache = TurnSelectorOutcomeCacheV1()
    identities = tuple(_selector_identity(index) for index in range(5))
    keys = tuple(selector_input_hash(identity) for identity in identities)

    # When: each selector output is cached.
    for index, key in enumerate(keys):
        cache.put(key, {"selected": index})

    # Then: only exact surviving hashes are reused.
    assert cache.get(keys[0]) is None
    assert cache.get(keys[1]) == {"selected": 1}
    assert (
        cache.get(
            selector_input_hash(
                identities[1].model_copy(
                    update={"selector_input_json": '{"changed":true,"selector":1}'}
                )
            )
        )
        is None
    )
    assert cache.keys == (keys[2], keys[3], keys[4], keys[1])


def _selector_identity(index: int) -> SelectorCacheIdentityV1:
    return SelectorCacheIdentityV1(
        selector_input_json=f'{{"selector":{index}}}',
        program_id="approved_knowledge_selector.scene_neutral.v1@1",
        program_version="2026-07-18.1",
        provider_target=build_provider_target_from_settings(
            provider="openai",
            target_slot="selector",
            model="deepseek-v4-pro",
            base_url="https://llm.example/v1",
            api_key_configured=True,
            timeout_seconds=20.0,
            temperature=0.0,
            max_tokens=1200,
        ).identity(),
    )
