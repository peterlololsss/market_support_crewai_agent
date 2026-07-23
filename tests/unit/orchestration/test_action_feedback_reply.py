from __future__ import annotations

from types import SimpleNamespace

import anyio
import pytest

from market_support_crewai_agent.runtime.context.projection import (
    ContextProjectionManager,
)
from market_support_crewai_agent.runtime.llm.composer_output import ComposerReplyOutput
from market_support_crewai_agent.runtime.orchestration import action_feedback_reply
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.turn import AgentRuntimeError
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    PrimaryReply,
)
from market_support_crewai_agent.settings import Settings


class FakeRuntime:
    def __init__(self) -> None:
        self.settings = Settings(llm_model="deepseek-v4-pro")
        self.conversation_store = ConversationStore()
        self.projection = ContextProjectionManager.from_settings(self.settings)

    def _project_context(self, **kwargs):
        return self.projection.project_for_stage(**kwargs)

    def _build_agent(self, stage):
        assert stage == "action_feedback_composer"
        return object()


def _feedback(outcome="partial") -> ActionFeedbackRequest:
    accepted_count = {"complete": 2, "partial": 1, "failed": 0}[outcome]
    failed_count = {"complete": 0, "partial": 1, "failed": 2}[outcome]
    return ActionFeedbackRequest.model_validate(
        {
            "conversation_key": "wecom:dm:sender",
            "group_id": "dm",
            "sender_id": "sender",
            "context_id": "msg-execute",
            "response_id": "resp-execute",
            "executions": [
                {
                    "action_type": "execute_prepared_outbound_message",
                    "status": "executed",
                    "action_id": "act-execute",
                    "artifact": None,
                    "adapter_result": {
                        "ok": outcome != "failed",
                        "outcome": outcome,
                        "target_count": 2,
                        "attempted_count": 2,
                        "accepted_count": accepted_count,
                        "failed_count": failed_count,
                        "unattempted_count": 0,
                        "replayed": False,
                        "confirmation_ref": "must-not-reach-the-model",
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize("outcome", ["complete", "partial", "failed"])
def test_terminal_outcomes_are_eligible_for_feedback_composition(outcome):
    summary = action_feedback_reply._terminal_execution_summary(_feedback(outcome))

    assert summary is not None
    assert summary["outcome"] == outcome


def test_terminal_feedback_composes_one_no_action_primary_reply(monkeypatch):
    captured = {}
    output = ComposerReplyOutput(
        response_mode="answer",
        reply=PrimaryReply(
            kind="answer",
            text="一个目标已提交，另一个未成功。",
            mentions=[],
        ),
        actions=[],
    )

    async def fake_kickoff(_agent, program, **_kwargs):
        captured["program"] = program
        return SimpleNamespace(pydantic=output, raw=""), []

    monkeypatch.setattr(
        action_feedback_reply,
        "run_composer_kickoff_with_retry",
        fake_kickoff,
    )

    reply = anyio.run(
        action_feedback_reply.compose_action_feedback_reply,
        FakeRuntime(),
        _feedback(),
    )

    assert reply == output.reply
    prompt_text = captured["program"].prompt_text
    assert '"outcome":"partial"' in prompt_text
    assert '"accepted_count":1' in prompt_text
    assert "must-not-reach-the-model" not in prompt_text


def test_non_execute_feedback_does_not_call_composer(monkeypatch):
    feedback = _feedback()
    feedback = feedback.model_copy(
        update={
            "executions": [
                feedback.executions[0].model_copy(
                    update={"action_type": "prepare_outbound_message"}
                )
            ]
        }
    )

    async def unexpected_kickoff(*_args, **_kwargs):
        raise AssertionError("composer must not run")

    monkeypatch.setattr(
        action_feedback_reply,
        "run_composer_kickoff_with_retry",
        unexpected_kickoff,
    )

    reply = anyio.run(
        action_feedback_reply.compose_action_feedback_reply,
        FakeRuntime(),
        feedback,
    )

    assert reply is None


def test_feedback_composer_rejects_reply_with_actions(monkeypatch):
    invalid = SimpleNamespace(
        pydantic=None,
        raw=(
            '{"contract_version":"composer-reply","response_mode":"answer",'
            '"claims":[],"evidence_ids":[],"missing_inputs":[],'
            '"reply":{"kind":"answer","text":"已提交","mentions":[]},'
            '"actions":[{"type":"execute_prepared_outbound_message",'
            '"confirmation_ref":"invalid"}]}'
        ),
    )

    async def fake_kickoff(*_args, **_kwargs):
        return invalid, []

    monkeypatch.setattr(
        action_feedback_reply,
        "run_composer_kickoff_with_retry",
        fake_kickoff,
    )

    try:
        anyio.run(
            action_feedback_reply.compose_action_feedback_reply,
            FakeRuntime(),
            _feedback("failed"),
        )
    except AgentRuntimeError as exc:
        assert "invalid reply contract" in str(exc)
    else:
        raise AssertionError("feedback composer accepted an action-bearing reply")
