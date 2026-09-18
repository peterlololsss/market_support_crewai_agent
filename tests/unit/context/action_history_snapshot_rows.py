from __future__ import annotations

from typing import Literal

from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.effect_records import (
    ActionLedgerRecordV2,
)

ActionType = Literal["send_material_pack", "send_weekly_report", "send_monthly_report"]
ActionStatus = Literal["executed", "failed", "skipped"]


def make_group_state_key() -> ConversationStateKey:
    return ConversationStateKey(
        surface="wecom",
        adapter_namespace="test-adapter",
        tenant_ref="tenant:test",
        scene="group",
        subject_ref="group:test",
        principal_ref="principal:test",
    )


def action_history_row(
    *,
    action_type: ActionType = "send_weekly_report",
    status: ActionStatus = "executed",
    received_at_epoch_ms: int = 1_000,
    state_key: ConversationStateKey | None = None,
) -> ActionLedgerRecordV2:
    return ActionLedgerRecordV2(
        state_key=state_key or make_group_state_key(),
        response_id="resp-" + "1" * 32,
        action_id="act-" + "2" * 32,
        action_type=action_type,
        status=status,
        artifact_ref="weekly:opaque-ref",
        status_revision=1,
        received_at_epoch_ms=received_at_epoch_ms,
        expires_at_monotonic_ns=9_999,
        pol1="pol1:opaque",
        par1="par1:test",
        grh1="grh1:test",
        bsh1="bsh1:test",
    )
