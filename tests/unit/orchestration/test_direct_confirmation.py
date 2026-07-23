from __future__ import annotations

import base64
from datetime import datetime, timezone
from functools import partial

import anyio

from market_support_crewai_agent.runtime.context.pending import (
    PendingOutboundConfirmation,
)
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
)
from market_support_crewai_agent.runtime.orchestration.direct_actions import (
    materialize_direct_output,
)
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.schemas import (
    ActionExecutionFeedback,
    AdapterResolveRequest,
    AdapterResolveResult,
    ExecutePreparedOutboundMessageAction,
    OutboundMessageTarget,
    OutboundTextContent,
    PrepareOutboundMessageAction,
    PrimaryReply,
    OutboundTargetResolveResult,
)


def _confirmation_ref(seed: int) -> str:
    token = base64.urlsafe_b64encode(bytes([seed]) * 32).decode("ascii").rstrip("=")
    return f"wecom-adapter-confirmation:{token}"


class _NoCallAdapter:
    async def resolve_outbound_target_async(
        self,
        target_kind: str | None,
        target_name: str,
    ) -> OutboundTargetResolveResult:
        raise AssertionError(f"unexpected target resolve: {target_kind=} {target_name=}")

    async def resolve_async(self, request: AdapterResolveRequest) -> AdapterResolveResult:
        raise AssertionError(f"unexpected report resolve: {request}")


def _feedback(response_id: str, confirmation_ref: str) -> ActionLedgerRecord:
    return ActionLedgerRecord(
        conversation_key="wecom:dm",
        group_id="dm",
        sender_id="sender",
        context_id=response_id,
        response_id=response_id,
        execution=ActionExecutionFeedback(
            action_type="prepare_outbound_message",
            status="executed",
            action_id="act-1",
            adapter_result={
                "confirmation_ref": confirmation_ref,
                "state": "prepared",
            },
        ),
        received_at=datetime.now(timezone.utc),
        dedupe_key=(response_id,),
    )


def _pending(response_id: str) -> PendingOutboundConfirmation:
    return PendingOutboundConfirmation(
        response_id=response_id,
        action=PrepareOutboundMessageAction(
            action_id="act-1",
            type="prepare_outbound_message",
            target=OutboundMessageTarget(
                kind="channel",
                name="银河证券",
                resolve_ref="outbound-target:" + "b" * 64,
            ),
            content=OutboundTextContent(kind="text", text="你好"),
        ),
    )


def _execute_output(confirmation_ref: str) -> DirectComposerOutput:
    return DirectComposerOutput(
        response_mode="execute_prepared_outbound_message",
        reply=PrimaryReply(kind="answer", text="", mentions=[]),
        confirmation_ref=confirmation_ref,
    )


def test_execute_binds_latest_pending_prepare_not_model_selected_old_version():
    old_ref = _confirmation_ref(1)
    latest_ref = _confirmation_ref(2)

    result = anyio.run(
        partial(
            materialize_direct_output,
            _execute_output(old_ref),
            adapter_client=_NoCallAdapter(),
            action_history=[
                _feedback("resp-old", old_ref),
                _feedback("resp-latest", latest_ref),
            ],
            pending_confirmation=_pending("resp-latest"),
        )
    )

    action = result.response.actions[0]
    assert isinstance(action, ExecutePreparedOutboundMessageAction)
    assert action.confirmation_ref == latest_ref


def test_execute_does_not_fall_back_to_old_feedback_while_latest_prepare_is_pending():
    old_ref = _confirmation_ref(1)

    result = anyio.run(
        partial(
            materialize_direct_output,
            _execute_output(old_ref),
            adapter_client=_NoCallAdapter(),
            action_history=[_feedback("resp-old", old_ref)],
            pending_confirmation=_pending("resp-latest"),
        )
    )

    assert result.mode == "clarification"
    assert result.response.actions == []
    assert result.response.reply.text == "发送准备尚未完成，请稍后再确认。"
