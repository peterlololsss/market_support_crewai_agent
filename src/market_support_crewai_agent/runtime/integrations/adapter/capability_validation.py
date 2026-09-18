from __future__ import annotations

from typing import Final, Literal

from market_support_crewai_agent.runtime.policy.capabilities.runtime_projection import (
    adapter_resolve_types,
)
from market_support_crewai_agent.schemas.adapter_metadata import AdapterCapabilities
from market_support_crewai_agent.schemas.conversation import (
    validate_canonical_tenant_ref,
)

REQUIRED_ADAPTER_SERVICE: Final = "xiaoyan-wecom-market-agent-adapter"
REQUIRED_RESOLVE_CONTRACT_VERSION: Final = "adapter-resolve"
REQUIRED_BATCH_CONTRACT_VERSION: Final = "adapter-resolve-batch"
REQUIRED_ACTION_CONTRACT_VERSION: Final = "adapter-action"
REQUIRED_REPLY_REQUEST_CONTRACT_VERSION: Final = "reply-request.v2"
REQUIRED_ACTION_FEEDBACK_CONTRACT_VERSION: Final = "action-feedback.v2"
REQUIRED_CONVERSATION_IDENTITY_CONTRACT_VERSION: Final = "conversation-identity.v1"
REQUIRED_RESOLVE_TYPES: Final = adapter_resolve_types()
REQUIRED_STATUSES: Final = {
    "resolved",
    "missing",
    "ambiguous",
    "forbidden",
    "temporarily_unavailable",
}
REQUIRED_ENDPOINTS: Final = {
    "health": "/health",
    "capabilities": "/adapter/capabilities",
    "metrics": "/adapter/metrics",
    "resolve": "/adapter/resolve",
    "batch_resolve": "/adapter/resolve/batch",
}
OPTIONAL_ENDPOINTS: Final = {"report_scope": "/adapter/report-scope"}
REQUIRED_MIN_BATCH_REQUESTS: Final = len(REQUIRED_RESOLVE_TYPES)


def adapter_capability_errors(capabilities: AdapterCapabilities) -> list[str]:
    errors: list[str] = []
    if capabilities.endpoints.health != REQUIRED_ENDPOINTS["health"]:
        errors.append("endpoints.health mismatch")
    if capabilities.endpoints.capabilities != REQUIRED_ENDPOINTS["capabilities"]:
        errors.append("endpoints.capabilities mismatch")
    if capabilities.endpoints.metrics != REQUIRED_ENDPOINTS["metrics"]:
        errors.append("endpoints.metrics mismatch")
    if capabilities.endpoints.resolve != REQUIRED_ENDPOINTS["resolve"]:
        errors.append("endpoints.resolve mismatch")
    if capabilities.endpoints.batch_resolve != REQUIRED_ENDPOINTS["batch_resolve"]:
        errors.append("endpoints.batch_resolve mismatch")
    if (
        capabilities.endpoints.report_scope is not None
        and capabilities.endpoints.report_scope != OPTIONAL_ENDPOINTS["report_scope"]
    ):
        errors.append("endpoints.report_scope mismatch")
    missing_types = sorted(REQUIRED_RESOLVE_TYPES - set(capabilities.resolve_types))
    if missing_types:
        errors.append("missing resolve_types={}".format(",".join(missing_types)))
    missing_statuses = sorted(REQUIRED_STATUSES - set(capabilities.statuses))
    if missing_statuses:
        errors.append("missing statuses={}".format(",".join(missing_statuses)))
    if capabilities.max_batch_requests < REQUIRED_MIN_BATCH_REQUESTS:
        errors.append(f"max_batch_requests={capabilities.max_batch_requests}")
    if capabilities.max_request_body_bytes <= 0:
        errors.append(f"max_request_body_bytes={capabilities.max_request_body_bytes}")
    return errors


def canonical_deployment_tenant_ref(deployment_tenant_ref: str) -> str:
    return validate_canonical_tenant_ref(
        deployment_tenant_ref,
        field_name="deployment_tenant_ref",
    )


def scene_compatibility_errors(
    capabilities: AdapterCapabilities,
    scene: Literal["direct", "group"],
    canonical_tenant_ref: str,
) -> list[str]:
    match scene:
        case "group":
            return []
        case "direct":
            errors: list[str] = []
            if (
                capabilities.supported_scenes is None
                or "direct" not in capabilities.supported_scenes
            ):
                errors.append("direct scene unsupported")
            if (
                capabilities.reply_request_contract_versions is None
                or REQUIRED_REPLY_REQUEST_CONTRACT_VERSION
                not in capabilities.reply_request_contract_versions
            ):
                errors.append("reply request contract unsupported")
            if (
                capabilities.action_feedback_contract_versions is None
                or REQUIRED_ACTION_FEEDBACK_CONTRACT_VERSION
                not in capabilities.action_feedback_contract_versions
            ):
                errors.append("action feedback contract unsupported")
            if (
                capabilities.conversation_identity_contract_versions is None
                or REQUIRED_CONVERSATION_IDENTITY_CONTRACT_VERSION
                not in capabilities.conversation_identity_contract_versions
            ):
                errors.append("conversation identity contract unsupported")
            if capabilities.deployment_tenant_ref != canonical_tenant_ref:
                errors.append("deployment tenant mismatch")
            return errors
