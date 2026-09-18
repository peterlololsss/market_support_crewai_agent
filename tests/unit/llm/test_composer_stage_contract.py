from __future__ import annotations

from dataclasses import replace

import pytest

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
    SmalltalkComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    V2ComposerOutputRejected,
    _validate_v2_composer_output,
    build_composer_prompt_input_v1,
    compose_v2_reply,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyMention
from tests.unit.llm._composer_stage_contract_fixtures import (
    ComposerRuntimeStub,
    composer_scenario,
    with_static_facts,
)


def test_knowledge_input_uses_selected_views_and_ordered_unit_groundings() -> None:
    scenario = composer_scenario("knowledge_answer", unit_count=2)

    value = build_composer_prompt_input_v1(scenario.invocation)

    assert isinstance(value, KnowledgeComposerPromptInputV1)
    assert value.contract_version == "knowledge-composer-input.v1"
    assert value.selected_capabilities.manifest_refs == (
        scenario.plan.selected_manifest_refs
    )
    assert len(value.selected_capabilities.items) == 1
    assert tuple(item.unit_id for item in value.unit_groundings) == (
        "unit-1",
        "unit-2",
    )
    assert tuple(item.manifest_ref for item in value.unit_groundings) == (
        scenario.plan.units[0].manifest_ref,
        scenario.plan.units[1].manifest_ref,
    )
    rendered = value.model_dump_json()
    for forbidden in (
        "canonical_facts",
        "domain_context",
        "eligible_capabilities",
        "identity",
        "ledger",
        "provider",
        "recall",
        '"resolve_ref":',
    ):
        assert forbidden not in rendered


def test_knowledge_input_rejects_cross_unit_grounding_swap_before_model() -> None:
    scenario = composer_scenario("knowledge_answer", unit_count=2)
    swapped = replace(
        scenario.invocation,
        groundings=tuple(reversed(scenario.groundings)),
    )

    with pytest.raises(ContextViewInvariantError, match="unit_grounding_unit_mismatch"):
        build_composer_prompt_input_v1(swapped)


def test_smalltalk_input_has_no_knowledge_or_evidence_fields() -> None:
    scenario = composer_scenario("smalltalk")

    value = build_composer_prompt_input_v1(scenario.invocation)

    assert isinstance(value, SmalltalkComposerPromptInputV1)
    assert value.contract_version == "smalltalk-composer-input.v1"
    payload = value.model_dump(mode="json")
    assert {
        "runtime_clock",
        "validated_plan",
        "preflight",
        "unit_groundings",
        "recall",
        "evidence",
        "business_facts",
        "answerability",
    }.isdisjoint(payload)


def test_flat_citations_cannot_span_multiple_unit_groundings() -> None:
    scenario = with_static_facts(
        composer_scenario("knowledge_answer", unit_count=2),
        ("company_shareholders", "company_historical_aum"),
    )
    input_value = build_composer_prompt_input_v1(scenario.invocation)
    output = ComposerReplyOutput(
        response_mode="answer",
        evidence_ids=[
            scenario.groundings[0].allowed_evidence_ids[0],
            scenario.groundings[1].allowed_evidence_ids[0],
        ],
        reply=PrimaryReply(kind="answer", text="组合回答", mentions=[]),
    )

    with pytest.raises(
        V2ComposerOutputRejected,
        match="v2_composer_cross_unit_authority_forbidden",
    ):
        _validate_v2_composer_output(output, input_value, scenario.invocation)


def test_group_registered_marker_requires_enabled_unified_knowledge() -> None:
    scenario = with_static_facts(
        composer_scenario("knowledge_answer"),
        ("company_shareholders",),
    )
    input_value = build_composer_prompt_input_v1(scenario.invocation)
    output = ComposerReplyOutput(
        response_mode="answer",
        evidence_ids=list(scenario.groundings[0].allowed_evidence_ids),
        reply=PrimaryReply(
            kind="answer",
            text="%%company_shareholders.png%%",
            mentions=[],
        ),
    )

    _validate_v2_composer_output(output, input_value, scenario.invocation)

    disabled_policy = scenario.invocation.policy.model_copy(
        update={"internal_company_knowledge_enabled": False}
    )
    with pytest.raises(V2ComposerOutputRejected, match="v2_composer_media_forbidden"):
        _validate_v2_composer_output(
            output,
            input_value,
            replace(scenario.invocation, policy=disabled_policy),
        )


def test_direct_registered_marker_is_forbidden_with_selected_evidence() -> None:
    scenario = with_static_facts(
        composer_scenario("knowledge_answer", direct=True),
        ("company_shareholders",),
    )
    input_value = build_composer_prompt_input_v1(scenario.invocation)
    admitted_id = scenario.groundings[0].allowed_evidence_ids[0]
    plain_output = ComposerReplyOutput(
        response_mode="answer",
        evidence_ids=[admitted_id],
        reply=PrimaryReply(kind="answer", text="已基于公司资料回答", mentions=[]),
    )
    marker_output = ComposerReplyOutput(
        response_mode="answer",
        evidence_ids=[admitted_id],
        reply=PrimaryReply(
            kind="answer",
            text="%%company_shareholders.png%%",
            mentions=[],
        ),
    )

    _validate_v2_composer_output(plain_output, input_value, scenario.invocation)
    with pytest.raises(V2ComposerOutputRejected, match="v2_composer_media_forbidden"):
        _validate_v2_composer_output(marker_output, input_value, scenario.invocation)


@pytest.mark.anyio
async def test_composer_rejects_nonempty_model_response_id() -> None:
    scenario = composer_scenario("knowledge_answer")

    class BadComposer:
        async def compose(self, input_value):
            del input_value
            return ComposerReplyOutput(
                response_id="model-owned-id",
                response_mode="answer",
                reply=PrimaryReply(kind="answer", text="答案", mentions=[]),
            )

    with pytest.raises(
        V2ComposerOutputRejected,
        match="v2_composer_response_id_forbidden",
    ):
        await compose_v2_reply(
            ComposerRuntimeStub(v2_composer=BadComposer()),
            scenario.invocation,
        )


@pytest.mark.anyio
async def test_direct_composer_cannot_add_mentions_or_actions() -> None:
    scenario = composer_scenario("knowledge_answer", direct=True)
    input_value = build_composer_prompt_input_v1(scenario.invocation)
    mentioned = ComposerReplyOutput(
        response_mode="answer",
        reply=PrimaryReply(
            kind="answer",
            text="答案",
            mentions=[ReplyMention(type="sales", reason="model")],
        ),
    )
    actioned = ComposerReplyOutput(
        response_mode="answer",
        reply=PrimaryReply(kind="answer", text="答案", mentions=[]),
    ).model_copy(update={"actions": [object()]})

    with pytest.raises(
        V2ComposerOutputRejected, match="v2_composer_mentions_forbidden"
    ):
        _validate_v2_composer_output(mentioned, input_value, scenario.invocation)
    with pytest.raises(V2ComposerOutputRejected, match="v2_composer_actions_forbidden"):
        _validate_v2_composer_output(actioned, input_value, scenario.invocation)


def test_composer_output_schema_keeps_empty_id_and_no_actions() -> None:
    output = ComposerReplyOutput(
        response_mode="answer",
        reply=PrimaryReply(kind="answer", text="答案", mentions=[]),
    )

    response = output.to_reply_response()

    assert response.response_id == ""
    assert response.reply.text == "答案"
    assert response.actions == []
