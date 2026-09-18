from __future__ import annotations

import json
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.runtime.hashing import CanonicalValue, sha256_frame


class ExecutionPlanHashSource(Protocol):
    def model_dump_json(self, *, exclude_none: bool) -> str: ...


_JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def execution_plan_id_v2(plan: ExecutionPlanHashSource) -> str:
    root = _JSON_VALUE_ADAPTER.validate_python(
        json.loads(plan.model_dump_json(exclude_none=False), parse_float=str)
    )
    payload = _json_object(root)
    del payload["execution_plan_id"]
    payload["plan_spec"] = None
    compliance = _canonical_object(payload["compliance"])
    compliance["reason"] = ""
    guardrail_decisions = payload["guardrail_decisions"]
    if isinstance(guardrail_decisions, list):
        for decision in guardrail_decisions:
            if isinstance(decision, dict):
                _ = decision.pop("message", None)
    return sha256_frame("execution-plan.v2", payload, prefix="epl1")


def _canonical_value(value: JsonValue) -> CanonicalValue:
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    return {key: _canonical_value(item) for key, item in value.items()}


def _json_object(value: JsonValue) -> dict[str, CanonicalValue]:
    if isinstance(value, dict):
        return {key: _canonical_value(item) for key, item in value.items()}
    raise PlanHashError("execution_plan_json_object_expected")


def _canonical_object(value: CanonicalValue) -> dict[str, CanonicalValue]:
    if isinstance(value, dict):
        return value
    raise PlanHashError("execution_plan_json_object_expected")


class PlanHashError(ValueError):
    def __init__(self, code: str) -> None:
        self.code: str = code
        super().__init__(code)
