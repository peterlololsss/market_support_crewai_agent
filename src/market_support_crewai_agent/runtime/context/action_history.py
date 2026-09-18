from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from market_support_crewai_agent.runtime.context.models import (
    RecentExecutedActionSummaryViewV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    ActionLedgerRecordV2,
)

_MAX_RECENT_ACTIONS: Final = 20
_MAX_ACTION_AGE_SECONDS: Final = 2_592_000
_ActionType = Literal["send_material_pack", "send_weekly_report", "send_monthly_report"]
_ArtifactType = Literal["material_pack", "weekly_report", "monthly_report"]


def recent_action_summaries_from_snapshot(
    rows: Sequence[ActionLedgerRecordV2],
    *,
    snapshot_at_epoch_ms: int,
) -> tuple[RecentExecutedActionSummaryViewV1, ...]:
    ordered_rows = sorted(
        enumerate(rows),
        key=lambda item: (-item[1].received_at_epoch_ms, item[0]),
    )
    summaries: list[RecentExecutedActionSummaryViewV1] = []
    for _, row in ordered_rows:
        summary = _summary_for_row(row, snapshot_at_epoch_ms)
        if summary is not None:
            summaries.append(summary)
        if len(summaries) == _MAX_RECENT_ACTIONS:
            break
    return tuple(summaries)


def _summary_for_row(
    row: ActionLedgerRecordV2,
    snapshot_at_epoch_ms: int,
) -> RecentExecutedActionSummaryViewV1 | None:
    if row.status != "executed":
        return None
    age_seconds = max(0, snapshot_at_epoch_ms - row.received_at_epoch_ms) // 1_000
    return RecentExecutedActionSummaryViewV1(
        action_type=row.action_type,
        artifact_type=_expected_artifact_type(row.action_type),
        material_pack_option=None,
        period=None,
        report_date=None,
        age_seconds=min(age_seconds, _MAX_ACTION_AGE_SECONDS),
    )


def _expected_artifact_type(action_type: _ActionType) -> _ArtifactType:
    return _ARTIFACT_TYPE_BY_ACTION[action_type]


_ARTIFACT_TYPE_BY_ACTION: Final[dict[_ActionType, _ArtifactType]] = {
    "send_material_pack": "material_pack",
    "send_weekly_report": "weekly_report",
    "send_monthly_report": "monthly_report",
}
