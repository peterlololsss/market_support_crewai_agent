from __future__ import annotations

# pyright: reportAny=false

import hashlib
import json
from pathlib import Path

from market_support_crewai_agent.runtime.planning.input_policy import (
    DEFAULT_INPUT_POLICY_RULES,
)


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures"


def test_target_manifest_fixture_has_thirteen_predeclared_records_and_sidecar() -> None:
    # Given: reviewed target bytes and their immutable hash sidecar.
    path = FIXTURES / "capability_registry_target.2026-07-18.1.json"
    target_bytes = path.read_bytes()
    expected = (
        (FIXTURES / "capability_registry_target.2026-07-18.1.sha256")
        .read_text(encoding="ascii")
        .strip()
    )
    target = json.loads(target_bytes)

    # When/Then: all target semantic rows are present and digest-bound.
    assert hashlib.sha256(target_bytes).hexdigest() == expected
    assert len(target["manifests"]) == 13
    assert {row["manifest_version"] for row in target["manifests"]} == {"2026-07-18.1"}
    assert {row["manifest_id"] for row in target["manifests"]} >= {
        "answer_internal_company_knowledge",
        "weekly_report.product_list",
        "monthly_report.product_list",
        "general.handoff",
    }


def test_reason_and_group_handoff_fixtures_are_closed() -> None:
    # Given: reason-code and denied-sales handoff baselines.
    reasons = json.loads(
        (FIXTURES / "reply_reason_codes.v1.json").read_text(encoding="utf-8")
    )
    handoff = json.loads(
        (FIXTURES / "legacy_group_handoff_no_sales.v1.json").read_text(encoding="utf-8")
    )

    # When/Then: planned codes are additive and denied sales has no effects.
    assert reasons["reason_codes"] == sorted(set(reasons["reason_codes"]))
    closed_reasons = set(reasons["reason_codes"])
    reachable_input_policy_reasons = {
        rule.reason_code for rule in DEFAULT_INPUT_POLICY_RULES
    }
    assert {"direct_human_handoff", "internal_error"} <= closed_reasons
    assert reachable_input_policy_reasons <= closed_reasons
    assert handoff["reason_code"] in reachable_input_policy_reasons
    assert handoff["reply"]["kind"] == "unable_to_answer"
    assert handoff["reply"]["mentions"] == []
    assert handoff["actions"] == []
