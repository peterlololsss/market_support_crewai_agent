from __future__ import annotations

import json

from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIProviderResultV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.result_coercion import (
    coerce_plan_spec,
    plan_spec_error_summary,
)
from tests.helpers.reply_contract_plan_fixtures import make_weekly_plan_spec

_JSON_MAPPING = TypeAdapter(dict[str, JsonValue])
_JSON_LIST = TypeAdapter(list[JsonValue])


def _raw_result(payload: dict[str, JsonValue]) -> CrewAIProviderResultV1:
    return CrewAIProviderResultV1(
        pydantic=None,
        raw=json.dumps(payload, ensure_ascii=False),
        agent_role="planner",
        usage_metrics=None,
    )


def test_plan_spec_allows_missing_evidence_contract_ref() -> None:
    plan = make_weekly_plan_spec()
    unit = plan.plan_units[0].model_copy(
        update={"evidence_contract_ref": None, "evidence_contract": None}
    )
    result = CrewAIProviderResultV1(
        pydantic=None,
        raw=plan.model_copy(update={"plan_units": [unit]}).model_dump_json(),
        agent_role="planner",
        usage_metrics=None,
    )

    assert coerce_plan_spec(result) is not None
    assert plan_spec_error_summary(result) == ""


def test_plan_spec_error_summary_names_invalid_unit_path() -> None:
    payload = _JSON_MAPPING.validate_json(make_weekly_plan_spec().model_dump_json())
    units = _JSON_LIST.validate_python(payload["plan_units"])
    unit = _JSON_MAPPING.validate_python(units[0])
    _ = unit.pop("selected_capability_id", None)
    payload["plan_units"] = [unit, *units[1:]]

    summary = plan_spec_error_summary(_raw_result(payload))

    assert "plan_units.0" in summary
    assert "selected_capability_id" in summary
