from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings


_ALIGNMENT_LIMIT_FIELDS = (
    "reply_alignment_max_replans",
    "reply_alignment_max_evidence_refetches",
    "reply_alignment_max_recomposes",
    "reply_alignment_max_total_remediations",
)


@pytest.mark.parametrize("field_name", _ALIGNMENT_LIMIT_FIELDS)
def test_alignment_limits_accept_closed_zero_to_two_range(field_name: str) -> None:
    # Given: each supported boundary value for one alignment limit.
    values = (0, 1, 2)

    # When: settings parse each boundary value.
    parsed = tuple(
        Settings.model_validate(
            {
                field_name: value,
                **(
                    {
                        "reply_alignment_max_replans": value,
                        "reply_alignment_max_evidence_refetches": value,
                        "reply_alignment_max_recomposes": value,
                    }
                    if field_name == "reply_alignment_max_total_remediations"
                    else {}
                ),
            }
        )
        for value in values
    )

    # Then: no value is clamped or rewritten.
    assert tuple(getattr(settings, field_name) for settings in parsed) == values


@pytest.mark.parametrize("field_name", _ALIGNMENT_LIMIT_FIELDS)
def test_alignment_limits_reject_values_above_two(field_name: str) -> None:
    # Given: an existing environment-compatible value above the V1 ceiling.
    payload = {field_name: 3}

    # When/Then: startup validation rejects instead of clamping it.
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


@pytest.mark.parametrize(
    "env_name",
    (
        "MARKET_AGENT_REPLY_ALIGNMENT_MAX_REPLANS",
        "MARKET_AGENT_REPLY_ALIGNMENT_MAX_EVIDENCE_REFETCHES",
        "MARKET_AGENT_REPLY_ALIGNMENT_MAX_RECOMPOSES",
        "MARKET_AGENT_REPLY_ALIGNMENT_MAX_TOTAL_REMEDIATIONS",
    ),
)
def test_alignment_environment_rejects_values_above_two(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
) -> None:
    # Given: one persisted deployment value above the bounded remediation ceiling.
    monkeypatch.setenv(env_name, "3")

    # When/Then: settings startup fails closed at the typed boundary.
    with pytest.raises(ValidationError):
        get_settings()


@pytest.mark.parametrize(
    "field_name",
    (
        "reply_alignment_max_replans",
        "reply_alignment_max_evidence_refetches",
        "reply_alignment_max_recomposes",
    ),
)
def test_alignment_action_caps_cannot_exceed_shared_total(
    field_name: str,
) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "reply_alignment_max_total_remediations": 0,
                field_name: 1,
            }
        )
