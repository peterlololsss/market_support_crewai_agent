from __future__ import annotations

import anyio
import pytest

from market_support_crewai_agent.runtime.context.action_history import (
    recent_action_summaries_from_snapshot,
)
from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.unit.context.action_history_lifecycle_spy import (
    LifecycleRuntimeSpy,
    SnapshotCoordinatorSpy,
)
from tests.unit.context.action_history_snapshot_rows import action_history_row


def test_snapshot_rows_change_typed_planner_action_summary_without_private_fields() -> (
    None
):
    snapshot_rows = (action_history_row(),)

    summaries = recent_action_summaries_from_snapshot(
        snapshot_rows,
        snapshot_at_epoch_ms=2_001_000,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.action_type == "send_weekly_report"
    assert summary.status == "executed"
    assert summary.artifact_type == "weekly_report"
    assert summary.age_seconds == 2_000
    payload = summary.model_dump(mode="json", exclude_none=True)
    assert payload == {
        "contract_version": "recent-action-summary-view.v1",
        "action_type": "send_weekly_report",
        "status": "executed",
        "artifact_type": "weekly_report",
        "age_seconds": 2_000,
    }
    assert not any(
        key in payload
        for key in (
            "state_key",
            "response_id",
            "action_id",
            "artifact_ref",
            "pol1",
            "par1",
        )
    )


def test_snapshot_projection_is_fixed_at_snapshot_and_empty_rows_stay_empty() -> None:
    snapshot_rows = (action_history_row(received_at_epoch_ms=1_000),)

    summaries = recent_action_summaries_from_snapshot(
        snapshot_rows,
        snapshot_at_epoch_ms=2_000,
    )
    later_rows = (action_history_row(received_at_epoch_ms=1_999_000),)

    assert summaries[0].age_seconds == 1
    assert recent_action_summaries_from_snapshot((), snapshot_at_epoch_ms=2_000) == ()
    assert summaries[0].age_seconds == 1
    assert later_rows[0].received_at_epoch_ms == 1_999_000


def test_snapshot_projection_keeps_newest_twenty_executed_rows_in_order() -> None:
    rows = tuple(
        action_history_row(received_at_epoch_ms=1_000 * index) for index in range(1, 22)
    ) + (
        action_history_row(received_at_epoch_ms=30_000, status="failed"),
        action_history_row(received_at_epoch_ms=31_000, status="skipped"),
    )

    summaries = recent_action_summaries_from_snapshot(
        rows,
        snapshot_at_epoch_ms=40_000,
    )

    assert len(summaries) == 20
    assert [summary.age_seconds for summary in summaries] == list(range(19, 39))
    assert all(summary.status == "executed" for summary in summaries)


def test_direct_snapshot_projection_never_exposes_identity_or_provider_details() -> (
    None
):
    direct_key = ConversationStateKey(
        surface="wecom",
        adapter_namespace="test-adapter",
        tenant_ref="tenant:private",
        scene="direct",
        subject_ref="user:private",
        principal_ref="principal:private",
    )

    payload = recent_action_summaries_from_snapshot(
        (action_history_row(state_key=direct_key),),
        snapshot_at_epoch_ms=2_000,
    )[0].model_dump(mode="json", exclude_none=True)

    rendered = str(payload)
    assert "private" not in rendered
    assert "opaque" not in rendered
    assert "resolve_ref" not in rendered
    assert "response_id" not in payload
    assert "action_id" not in payload


def test_lifecycle_reads_snapshot_once_and_passes_immutable_typed_history_to_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREWAI_MAX_RETRY_LIMIT", "0")
    from market_support_crewai_agent.runtime.lifecycle import run_reply_turn

    envelope = make_v2_envelope("hello")
    row = action_history_row(state_key=envelope.state_key, received_at_epoch_ms=1_000)
    coordinator = SnapshotCoordinatorSpy(envelope.state_key, row)
    runtime = LifecycleRuntimeSpy(
        coordinator,
        Settings(llm_api_key="test-key", reply_alignment_verifier_enabled=False),
    )

    response = anyio.run(run_reply_turn, runtime, envelope)

    captured_action_history = runtime.captured_action_history
    assert response.reply.text == "ok"
    assert coordinator.snapshot_reads == 1
    assert coordinator.committed is True
    assert captured_action_history is not None
    assert len(captured_action_history) == 1
    assert captured_action_history[0].action_type == "send_weekly_report"
    assert captured_action_history[0].age_seconds == 2_000
