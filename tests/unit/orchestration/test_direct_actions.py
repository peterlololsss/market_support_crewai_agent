from __future__ import annotations

from functools import partial
from typing import Literal

import anyio

from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
    DirectTargetDraft,
    DirectTextDraft,
)
from market_support_crewai_agent.runtime.orchestration.direct_actions import (
    materialize_direct_output,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveRequest,
    AdapterResolveResult,
    OutboundTargetKind,
    OutboundTargetResolveResult,
    PrepareOutboundMessageAction,
    PrimaryReply,
)


TargetStatus = Literal["resolved", "missing", "ambiguous", "temporarily_unavailable"]


class FakeTargetAdapter:
    def __init__(
        self,
        statuses: dict[str, TargetStatus],
        *,
        target_counts: dict[str, int] | None = None,
    ) -> None:
        self.statuses: dict[str, TargetStatus] = statuses
        self.target_counts = target_counts or {}
        self.requests: list[tuple[str | None, str]] = []

    async def resolve_outbound_target_async(
        self,
        target_kind: str | None,
        target_name: str,
    ) -> OutboundTargetResolveResult:
        self.requests.append((target_kind, target_name))
        status = self.statuses.get(target_kind or "", "missing")
        resolved = status == "resolved"
        target_count = self.target_counts.get(
            target_kind or "",
            1 if resolved else 0,
        )
        resolved_kind: OutboundTargetKind = (
            target_kind if target_kind in ("channel", "group") else "group"
        )
        return OutboundTargetResolveResult(
            status=status,
            reason_code=(
                "ok"
                if resolved
                else "target_incomplete"
                if target_count
                else "target_not_found"
            ),
            display_name=target_name,
            target_kind=resolved_kind,
            target_count=target_count,
            resolved_count=1 if resolved else 0,
            resolve_ref=("outbound-target:" + "b" * 64) if resolved else "",
        )

    async def resolve_async(
        self,
        request: AdapterResolveRequest,
    ) -> AdapterResolveResult:
        raise AssertionError(f"unexpected report resolve: {request}")


def _prepare(
    target_kind: OutboundTargetKind | None = None,
) -> DirectComposerOutput:
    return DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="请确认目标是群还是渠道。",
            mentions=[],
        ),
        target=DirectTargetDraft(kind=target_kind, name="同名目标"),
        content=DirectTextDraft(kind="text", text="测试公告"),
    )


def _materialize(
    adapter: FakeTargetAdapter,
    target_kind: OutboundTargetKind | None = None,
):
    return anyio.run(
        partial(
            materialize_direct_output,
            _prepare(target_kind),
            adapter_client=adapter,
            action_history=[],
        )
    )


def test_unspecified_target_kind_uses_the_only_sendable_candidate():
    for sendable_kind in ("channel", "group"):
        adapter = FakeTargetAdapter(
            {
                "channel": "resolved" if sendable_kind == "channel" else "missing",
                "group": "resolved" if sendable_kind == "group" else "missing",
            }
        )

        result = _materialize(adapter)

        assert adapter.requests == [
            ("channel", "同名目标"),
            ("group", "同名目标"),
        ]
        action = result.response.actions[0]
        assert isinstance(action, PrepareOutboundMessageAction)
        assert action.target.kind == sendable_kind
        target_label = "渠道" if sendable_kind == "channel" else "群"
        assert f"已找到可发送的{target_label}" in result.response.reply.text
        assert "还是渠道" not in result.response.reply.text


def test_explicit_target_kind_falls_back_to_sendable_same_name_candidate():
    adapter = FakeTargetAdapter({"channel": "missing", "group": "resolved"})

    result = _materialize(adapter, "channel")

    assert adapter.requests == [
        ("channel", "同名目标"),
        ("group", "同名目标"),
    ]
    assert result.mode == "action"
    action = result.response.actions[0]
    assert isinstance(action, PrepareOutboundMessageAction)
    assert action.target.kind == "group"
    assert "同名群" in result.response.reply.text


def test_prepare_confirmation_echoes_resolved_target_and_exact_text():
    adapter = FakeTargetAdapter({"channel": "resolved"})

    result = _materialize(adapter, "channel")

    assert result.response.reply.text == (
        "发送目标：渠道「同名目标」\n\n"
        "待发送原文：\n测试公告\n\n"
        "请确认是否发送？"
    )


def test_unspecified_target_kind_clarifies_when_both_candidates_are_sendable():
    adapter = FakeTargetAdapter({"channel": "resolved", "group": "resolved"})

    result = _materialize(adapter)

    assert result.mode == "clarification"
    assert result.response.actions == []
    assert "同名群和渠道" in result.response.reply.text


def test_unspecified_target_kind_explains_when_neither_candidate_is_sendable():
    adapter = FakeTargetAdapter({"channel": "missing", "group": "missing"})

    result = _materialize(adapter)

    assert result.mode == "unable"
    assert result.response.actions == []
    assert "没有准备发送" in result.response.reply.text


def test_known_channel_without_reachable_groups_preserves_retry_draft():
    adapter = FakeTargetAdapter(
        {"channel": "missing", "group": "missing"},
        target_counts={"channel": 11},
    )

    result = _materialize(adapter, "channel")

    assert result.mode == "unable"
    assert result.pending_outbound_draft is not None
    assert result.pending_outbound_draft.target is None
    assert result.pending_outbound_draft.content == _prepare("channel").content
    assert "已识别渠道" in result.response.reply.text
    assert "0/11" in result.response.reply.text
