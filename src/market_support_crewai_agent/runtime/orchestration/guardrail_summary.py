from __future__ import annotations

from pydantic import JsonValue

from market_support_crewai_agent.runtime.domain.business_facts import BusinessFacts
from market_support_crewai_agent.runtime.domain.capabilities import (
    resolve_type_for_action,
)
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
)


def guardrail_decisions_from_directive(
    directive: ResponseDirective,
    plan: ExecutionPlan,
    business_facts: BusinessFacts,
) -> list[GuardrailDecision]:
    if directive.reason_code != "ambiguous_action_resolve":
        return []
    candidates: list[str] = []
    action_types: list[str] = []
    for action in plan.action_intents:
        action_types.append(action.action_type)
        resolve_type = resolve_type_for_action(action.action_type)
        if resolve_type is None:
            continue
        candidates.extend(business_facts.resolve_state(resolve_type).candidates)
    return [
        GuardrailDecision(
            outcome="require_clarification",
            phase="output",
            reason_code="ambiguous_action_resolve",
            metadata={
                "action_types": _unique_strings(action_types),
                "candidates": _unique_strings(candidates),
            },
        )
    ]


def _unique_strings(values: list[str]) -> list[JsonValue]:
    seen: set[str] = set()
    output: list[JsonValue] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
