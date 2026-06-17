from __future__ import annotations

from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.evidence_source_guard import (
    manifests_for_capability,
)
from market_support_crewai_agent.runtime.validation.guardrail_common import ordered_unique
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
    make_decision,
)


def execution_tool_guard(
    *,
    plan: object,
    policy: PolicyManifest,
    domain_context: DomainContext | None = None,
) -> GuardrailDecision:
    allowed_artifact_ids = {
        artifact.id
        for artifact in (domain_context.artifacts if domain_context is not None else ())
    }
    allowed_tools = allowed_tools_for_policy(policy)
    seen_tools = requested_tools_for_plan(plan)

    for tool_name in seen_tools:
        if tool_name not in allowed_tools:
            return make_decision(
                "block",
                "execution_tool",
                "tool_not_allowed",
                human_reason="Execution plan requested a tool outside policy.",
                metadata={"tool_name": tool_name, "allowed_tools": sorted(allowed_tools)},
            )

    for resolve_spec in getattr(plan, "adapter_resolves", []) or []:
        resolve_type = str(getattr(resolve_spec, "resolve_type", "") or "")
        tool_name = f"adapter_resolve.{resolve_type}"
        artifact_id = str(getattr(resolve_spec, "artifact_id", "") or "")
        if artifact_id and artifact_id not in allowed_artifact_ids:
            return make_decision(
                "block",
                "execution_tool",
                "invalid_artifact_id",
                human_reason="Tool call referenced an artifact outside DomainContext.",
                artifact_ids=[artifact_id],
                metadata={"tool_name": tool_name},
            )

    return make_decision(
        "allow",
        "execution_tool",
        "tool_calls_allowed",
        metadata={"tools_seen": seen_tools},
    )


def requested_tools_for_plan(plan: object) -> list[str]:
    tools: list[str] = []
    for resolve_spec in getattr(plan, "adapter_resolves", []) or []:
        resolve_type = str(getattr(resolve_spec, "resolve_type", "") or "")
        if resolve_type:
            tools.append(f"adapter_resolve.{resolve_type}")
    tools.extend(
        tool
        for capability in getattr(plan, "capabilities", []) or []
        for manifest in manifests_for_capability(str(capability), plan)
        for tool in manifest.required_tools
    )
    return ordered_unique(tools)


def allowed_tools_for_policy(policy: PolicyManifest) -> set[str]:
    tools = {
        f"adapter_resolve.{resolve_type}"
        for resolve_type in policy.allowed_adapter_resolves
    }
    for capability in policy.allowed_capabilities:
        for manifest in manifests_for_capability(str(capability)):
            tools.update(manifest.required_tools)
    return tools
