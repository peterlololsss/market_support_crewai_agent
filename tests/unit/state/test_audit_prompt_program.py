from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.integrations.crewai.io import (
    DirectSafeProviderTransportV1,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationRowV1,
)


def test_audit_transport_row_uses_only_redacted_hashes_and_closed_status() -> None:
    # Given: a successful provider dispatch projected for audit.
    row = DirectSafeProviderTransportV1(
        osh1="osh1:" + "0" * 64,
        poh1="poh1:" + "1" * 64,
        prh1="prh1:" + "2" * 64,
        out1="out1:" + "3" * 64,
        status="success",
    )

    # When/Then: no raw prompt, provider, URL, or output body is modelled.
    dumped = row.model_dump(mode="json")
    assert set(dumped) == {
        "contract_version",
        "error_code",
        "osh1",
        "out1",
        "poh1",
        "prh1",
        "status",
    }
    assert "https://" not in repr(dumped)
    assert "prompt" not in repr(dumped)


def test_audit_journal_rejects_invalid_ordinal_and_attempt_values() -> None:
    # Given/When/Then: audit schema admits only ordinals and logical attempts 1..18.
    valid = {
        "stage_kind": "planner_intent",
        "program_id": "planner_intent.wecom_group.v1@1",
        "target_slot": "planner",
        "purpose": "planner",
        "status": "reserved",
    }
    for field_name in ("invocation_ordinal", "logical_attempt"):
        payload = valid | {
            "invocation_ordinal": 1,
            "logical_attempt": 1,
            field_name: 19,
        }
        with pytest.raises(ValueError):
            _ = TurnLlmInvocationRowV1.model_validate(payload)
