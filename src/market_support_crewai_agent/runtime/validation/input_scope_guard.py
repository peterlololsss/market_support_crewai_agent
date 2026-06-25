from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities import (
    capability_by_name,
)
from market_support_crewai_agent.runtime.domain.ontology import DomainContext
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.validation.guardrail_common import (
    adapter_channel_id,
    artifact_type_for_capability,
    ordered_unique,
    requested_scope,
    requested_scope_dict,
)
from market_support_crewai_agent.runtime.validation.guardrail_types import (
    GuardrailDecision,
    RequestedScope,
    SendScopePolicy,
    make_decision,
)
from market_support_crewai_agent.schemas import ReplyRequest


def send_scope_policy(
    policy: PolicyManifest,
    domain_context: DomainContext | None = None,
) -> SendScopePolicy:
    artifact_types = []
    for capability_name in policy.allowed_capabilities:
        capability = capability_by_name(capability_name)
        if capability is None:
            continue
        artifact_type = artifact_type_for_capability(capability_name)
        if artifact_type not in artifact_types:
            artifact_types.append(artifact_type)

    destinations: list[str] = []
    if domain_context is not None:
        destinations.extend(
            [
                "current_channel",
                domain_context.channel.id,
                domain_context.channel.name,
                adapter_channel_id(domain_context),
            ]
        )
        for strategy in domain_context.strategies:
            destinations.extend([strategy.id, strategy.name])

    return SendScopePolicy(
        allowed_capabilities=sorted(policy.allowed_capabilities),
        allowed_artifact_types=artifact_types,
        allowed_destinations=ordered_unique(destinations),
        allowed_actions=sorted(policy.allowed_outbound_actions),
        required_user_confirmation=[],
        redaction_policy={
            "sensitive_fields": ["resolve_ref", "internal_locator"],
            "user_visible_text": "redact_internal_locators",
        },
    )


def input_guard(
    *,
    request: ReplyRequest,
    intent_frame: object,
    policy: PolicyManifest,
    domain_context: DomainContext | None = None,
) -> GuardrailDecision:
    scope_policy = send_scope_policy(policy, domain_context)
    if request.dist_channel_name not in scope_policy.allowed_destinations:
        scope_policy = scope_policy.model_copy(
            update={
                "allowed_destinations": [
                    *scope_policy.allowed_destinations,
                    request.dist_channel_name,
                ]
            }
        )
    scope = requested_scope(intent_frame)
    action_intent = str(getattr(intent_frame, "action_intent", "none") or "none")
    requested_capabilities = [
        str(item)
        for item in getattr(intent_frame, "requested_capabilities", []) or []
    ]
    work_items = list(getattr(intent_frame, "work_items", []) or [])
    outbound_action_requested = action_intent == "send" or any(
        str(getattr(item, "intent", "") or "") == "send" for item in work_items
    )

    if scope is None:
        return make_decision(
            "allow",
            "input",
            "input_scope_omitted_current_context_allowed",
            capability_id=_first_capability(requested_capabilities),
            human_reason=(
                "No explicit destination scope was provided; current channel "
                "context may be used only for omitted send targets."
            ),
        )

    if scope.capability and scope.capability not in scope_policy.allowed_capabilities:
        return make_decision(
            "block",
            "input",
            "capability_not_allowed_by_send_scope_policy",
            capability_id=scope.capability,
            human_reason="Requested capability is outside the compiled policy.",
            metadata={"allowed_capabilities": scope_policy.allowed_capabilities},
        )

    if scope.action and scope.action not in scope_policy.allowed_actions:
        return make_decision(
            "block",
            "input",
            "action_not_allowed_by_send_scope_policy",
            capability_id=scope.capability,
            human_reason="Requested outbound action is outside the compiled policy.",
            metadata={"allowed_actions": scope_policy.allowed_actions},
        )

    if scope.requires_user_confirmation:
        return make_decision(
            "require_confirmation",
            "input",
            "user_confirmation_required_for_sensitive_scope",
            capability_id=scope.capability,
            human_reason="The requested scope requires explicit user confirmation.",
            source_scopes=[requested_scope_dict(scope)],
        )

    if not outbound_action_requested and not scope.has_destination:
        return make_decision(
            "allow",
            "input",
            "input_scope_allowed",
            capability_id=scope.capability,
            source_scopes=[requested_scope_dict(scope)],
        )

    if scope.destination_type == "unknown":
        return make_decision(
            "require_clarification",
            "input",
            "ambiguous_destination_scope",
            capability_id=scope.capability,
            human_reason="The requested destination or source scope is ambiguous.",
            source_scopes=[requested_scope_dict(scope)],
        )

    if scope.destination_type == "channel":
        if _scope_matches_allowed_destination(scope, scope_policy):
            return make_decision(
                "allow",
                "input",
                "input_scope_allowed",
                capability_id=scope.capability,
                source_scopes=[requested_scope_dict(scope)],
            )
        return make_decision(
            "block",
            "input",
            "send_scope_destination_outside_current_channel",
            capability_id=scope.capability,
            human_reason=(
                "The user requested a different destination than the current "
                "adapter channel."
            ),
            source_scopes=[requested_scope_dict(scope)],
            metadata={
                "requested_target": (
                    scope.destination_name
                    or scope.destination_id
                    or ""
                ),
                "current_scope": request.dist_channel_name,
                "allowed_destinations": scope_policy.allowed_destinations,
            },
        )

    if scope.destination_type == "strategy":
        if _scope_matches_allowed_destination(scope, scope_policy):
            return make_decision(
                "allow",
                "input",
                "input_strategy_scope_allowed",
                capability_id=scope.capability,
                source_scopes=[requested_scope_dict(scope)],
            )
        return make_decision(
            "require_clarification",
            "input",
            "unknown_strategy_scope",
            capability_id=scope.capability,
            human_reason="The requested strategy scope is not resolved.",
            source_scopes=[requested_scope_dict(scope)],
        )

    return make_decision(
        "allow",
        "input",
        "input_scope_allowed",
        capability_id=scope.capability,
        source_scopes=[requested_scope_dict(scope)],
    )


def _scope_matches_allowed_destination(
    scope: RequestedScope,
    scope_policy: SendScopePolicy,
) -> bool:
    candidates = {
        item
        for item in (
            scope.destination_id,
            scope.destination_name,
            scope.strategy_id,
            scope.strategy_name,
        )
        if item
    }
    return bool(candidates.intersection(scope_policy.allowed_destinations))


def _first_capability(capabilities: list[str]) -> str | None:
    return capabilities[0] if capabilities else None
