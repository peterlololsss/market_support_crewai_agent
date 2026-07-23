from market_support_crewai_agent.runtime.context.pending import (
    PendingOutboundConfirmation,
)
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
    DirectTargetDraft,
    DirectTextDraft,
)
from market_support_crewai_agent.runtime.validation.direct_pending_confirmation import (
    pending_confirmation_resolution_issue,
)
from market_support_crewai_agent.schemas import (
    OutboundMessageTarget,
    OutboundTextContent,
    PrepareOutboundMessageAction,
    PrimaryReply,
)


def _pending() -> PendingOutboundConfirmation:
    return PendingOutboundConfirmation(
        response_id="resp-prepare",
        action=PrepareOutboundMessageAction(
            action_id="act-1",
            type="prepare_outbound_message",
            target=OutboundMessageTarget(
                kind="channel",
                name="银河证券",
                resolve_ref="outbound-target:" + "b" * 64,
            ),
            content=OutboundTextContent(kind="text", text="原文"),
        ),
    )


def test_active_correction_cannot_resolve_as_smalltalk() -> None:
    output = DirectComposerOutput(
        response_mode="smalltalk",
        pending_confirmation_resolution="correct",
        reply=PrimaryReply(kind="answer", text="好的，原文是你好。", mentions=[]),
    )

    assert pending_confirmation_resolution_issue(output, _pending()) == (
        "pending_confirmation_resolution_mode_mismatch"
    )


def test_active_confirmation_requires_an_explicit_resolution() -> None:
    output = DirectComposerOutput(
        response_mode="smalltalk",
        reply=PrimaryReply(kind="answer", text="好的。", mentions=[]),
    )

    assert pending_confirmation_resolution_issue(output, _pending()) == (
        "pending_confirmation_resolution_required"
    )


def test_active_correction_accepts_new_prepare() -> None:
    output = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        pending_confirmation_resolution="correct",
        reply=PrimaryReply(kind="clarification", text="请确认是否发送？", mentions=[]),
        target=DirectTargetDraft(kind="channel", name="银河证券"),
        content=DirectTextDraft(kind="text", text="你好"),
    )

    assert pending_confirmation_resolution_issue(output, _pending()) is None


def test_resolution_is_not_required_without_active_confirmation() -> None:
    output = DirectComposerOutput(
        response_mode="smalltalk",
        reply=PrimaryReply(kind="answer", text="老师好。", mentions=[]),
    )

    assert pending_confirmation_resolution_issue(output, None) is None
