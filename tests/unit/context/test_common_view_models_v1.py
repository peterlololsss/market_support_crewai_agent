from __future__ import annotations

from typing import TypeAlias

import pytest
from pydantic import ValidationError

import market_support_crewai_agent.runtime.context.models as context_models


JsonScalar: TypeAlias = str | int | float | bool | None


def test_current_message_preserves_legacy_text_without_a_schema_maximum() -> None:
    # Given: accepted legacy text beyond the v2 public request ceiling.
    text = "x" * 65_536

    # When: the text crosses the model-visible message boundary.
    view = context_models.CurrentMessageViewV1(text=text)

    # Then: the exact text survives without clipping or saturation.
    assert view.text == text
    assert len(view.text) == 65_536


def test_current_message_is_frozen_and_rejects_unknown_fields() -> None:
    # Given: a valid strict current-message view.
    view = context_models.CurrentMessageViewV1(text="hello")

    # When/Then: mutation and identity-like extras are rejected.
    with pytest.raises(ValidationError):
        view.text = "changed"
    with pytest.raises(ValidationError):
        context_models.CurrentMessageViewV1.model_validate(
            {"text": "hello", "request_id": "req:secret"}
        )


def test_history_turn_enforces_role_text_and_age_bounds() -> None:
    # Given: history values at every declared upper bound.
    text = "问" * 1_200

    # When: the history turn is parsed.
    view = context_models.HistoryTurnViewV1(
        role="assistant", text=text, age_seconds=2_592_000
    )

    # Then: the exact bounded values are retained.
    assert view.text == text
    assert view.age_seconds == 2_592_000

    # Given: values outside each closed field contract.
    invalid_payloads: tuple[dict[str, JsonScalar], ...] = (
        {"role": "system", "text": "", "age_seconds": 0},
        {"role": "user", "text": "x" * 1_201, "age_seconds": 0},
        {"role": "user", "text": "", "age_seconds": -1},
        {"role": "user", "text": "", "age_seconds": 2_592_001},
    )

    # When/Then: every out-of-contract history value is rejected.
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            context_models.HistoryTurnViewV1.model_validate(payload)


def test_runtime_clock_requires_one_consistent_shanghai_snapshot() -> None:
    # Given: one exact Saturday instant in Asia/Shanghai.
    payload = {
        "timezone": "Asia/Shanghai",
        "current_date": "2026-07-18",
        "current_datetime": "2026-07-18T14:20:00+08:00",
        "weekday": 6,
        "relative_years": {
            "current": 2026,
            "last": 2025,
            "two_years_ago": 2024,
        },
    }

    # When: the clock projection is parsed.
    view = context_models.RuntimeClockViewV1.model_validate(payload)

    # Then: the typed relative-year object is retained exactly.
    assert view.relative_years.current == 2026
    assert view.current_datetime.endswith("+08:00")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timezone", "UTC"),
        ("current_date", "2026-02-30"),
        ("current_datetime", "2026-07-18T14:20:00+09:00"),
        ("weekday", 5),
        (
            "relative_years",
            {"current": 2026, "last": 2024, "two_years_ago": 2023},
        ),
    ],
)
def test_runtime_clock_rejects_inconsistent_snapshot_fields(
    field: str,
    value: str | int | dict[str, int],
) -> None:
    # Given: a valid clock with one inconsistent field replacement.
    payload = {
        "timezone": "Asia/Shanghai",
        "current_date": "2026-07-18",
        "current_datetime": "2026-07-18T14:20:00+08:00",
        "weekday": 6,
        "relative_years": {
            "current": 2026,
            "last": 2025,
            "two_years_ago": 2024,
        },
        field: value,
    }

    # When/Then: the impossible one-clock snapshot is rejected.
    with pytest.raises(ValidationError):
        context_models.RuntimeClockViewV1.model_validate(payload)


def test_pending_clarification_requires_canonical_safe_slots() -> None:
    # Given: a pending clarification with sorted unique slots.
    view = context_models.PendingClarificationViewV1(
        kind="report_scope",
        slots=("artifact", "period"),
        question="请确认报告范围",
        age_turns=12,
    )

    # When: its bounded fields are inspected.
    payload = view.model_dump(mode="json")

    # Then: only the safe clarification summary is present.
    assert payload["slots"] == ["artifact", "period"]
    assert set(payload) == {
        "contract_version",
        "kind",
        "slots",
        "question",
        "age_turns",
    }

    # Given: non-canonical or control-bearing clarification text.
    invalid_payloads = (
        {"slots": ("period", "artifact"), "question": "ok"},
        {"slots": ("period", "period"), "question": "ok"},
        {"slots": ("period",), "question": "bad\x00text"},
    )

    # When/Then: every unsafe variant is rejected.
    for invalid in invalid_payloads:
        with pytest.raises(ValidationError):
            context_models.PendingClarificationViewV1.model_validate(
                {"kind": "report_scope", "age_turns": 0, **invalid}
            )
