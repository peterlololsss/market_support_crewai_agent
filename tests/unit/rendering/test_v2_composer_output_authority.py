from __future__ import annotations

from dataclasses import dataclass

import pytest

from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerPromptInputV1,
    V2ComposerOutputRejected,
    compose_v2_reply,
)
from market_support_crewai_agent.schemas.reply import (
    PrimaryReply,
    ReplyMention,
    SendWeeklyReportAction,
)
from tests.unit.llm._composer_stage_contract_fixtures import (
    ComposerRuntimeStub,
    composer_scenario,
    with_static_facts,
)


@dataclass(frozen=True, slots=True)
class _MentionComposer:
    async def compose(self, input_value: ComposerPromptInputV1) -> ComposerReplyOutput:
        del input_value
        return ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(
                kind="answer",
                text="答案",
                mentions=[ReplyMention(type="sales", reason="model")],
            ),
        )


@dataclass(frozen=True, slots=True)
class _ActionComposer:
    async def compose(self, input_value: ComposerPromptInputV1) -> ComposerReplyOutput:
        del input_value
        action = SendWeeklyReportAction(
            type="send_weekly_report",
            resolve_type="weekly_report",
            resolve_ref="adapter:weekly_report:current",
            period="20260612",
            report_date="2026-06-12",
        )
        return ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(kind="answer", text="答案", mentions=[]),
        ).model_copy(update={"actions": [action]})


@dataclass(frozen=True, slots=True)
class _MarkerComposer:
    marker: str
    evidence_id: str

    async def compose(self, input_value: ComposerPromptInputV1) -> ComposerReplyOutput:
        del input_value
        return ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(kind="answer", text=self.marker, mentions=[]),
            evidence_ids=[self.evidence_id],
        )


@pytest.mark.anyio
async def test_compose_rejects_model_mentions() -> None:
    # Given: a model response that proposes a customer-visible sales mention.
    scenario = composer_scenario("knowledge_answer")

    # When/Then: the composer boundary rejects that uncontrolled mention.
    with pytest.raises(
        V2ComposerOutputRejected, match="v2_composer_mentions_forbidden"
    ):
        _ = await compose_v2_reply(
            ComposerRuntimeStub(v2_composer=_MentionComposer()),
            scenario.invocation,
        )


@pytest.mark.anyio
async def test_compose_rejects_model_action_selection() -> None:
    # Given: a model response with a fully shaped outbound action.
    scenario = composer_scenario("knowledge_answer")

    # When/Then: only the deterministic action renderer may select actions.
    with pytest.raises(V2ComposerOutputRejected, match="v2_composer_actions_forbidden"):
        _ = await compose_v2_reply(
            ComposerRuntimeStub(v2_composer=_ActionComposer()),
            scenario.invocation,
        )


@pytest.mark.anyio
async def test_direct_composer_rejects_catalog_valid_image_marker() -> None:
    # Given: a direct-scene composer returns an otherwise catalog-valid marker.
    scenario = with_static_facts(
        composer_scenario("knowledge_answer", direct=True),
        ("company_shareholders",),
    )
    marker = "%%company_shareholders.png%%"

    # When/Then: direct responses reject media even when evidence is admitted.
    with pytest.raises(V2ComposerOutputRejected, match="v2_composer_media_forbidden"):
        _ = await compose_v2_reply(
            ComposerRuntimeStub(
                v2_composer=_MarkerComposer(
                    marker=marker,
                    evidence_id=scenario.groundings[0].allowed_evidence_ids[0],
                )
            ),
            scenario.invocation,
        )


@pytest.mark.anyio
async def test_group_composer_preserves_registered_image_marker() -> None:
    # Given: a group-scene composer returns a registered marker and its evidence.
    scenario = with_static_facts(
        composer_scenario("knowledge_answer"),
        ("company_shareholders",),
    )
    marker = "%%company_shareholders.png%%"

    # When: the boundary validates the group-scene media binding.
    output = await compose_v2_reply(
        ComposerRuntimeStub(
            v2_composer=_MarkerComposer(
                marker=marker,
                evidence_id=scenario.groundings[0].allowed_evidence_ids[0],
            )
        ),
        scenario.invocation,
    )

    # Then: the registered marker remains customer-visible.
    assert output.reply.text == marker
