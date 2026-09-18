from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    kernel_available_artifacts,
    kernel_scene_is_group,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
    ReadCapability,
    ResponseMode,
)
from market_support_crewai_agent.runtime.policy.manifest_models import (
    PolicyAuthorityCoreV1,
    PolicyLedgerSummaryV1,
    PolicyManifestCompilationError,
    PolicyManifestV2,
    RecallModeV1,
    effective_grants_hash_v1,
)
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveType,
    OutboundActionType,
)

if TYPE_CHECKING:
    from market_support_crewai_agent.runtime.evidence.scope_authority import (
        BusinessScopeAuthorityV1,
    )

_MANIFEST_READ_REQUIREMENTS: Final[dict[str, ReadCapability]] = {
    "material_pack.send": "resolve_material_pack",
    "weekly_report.send": "resolve_weekly_report",
    "monthly_report.send": "resolve_monthly_report",
    "sales.handoff": "resolve_sales_mention",
    "answer_internal_company_knowledge": "query_internal_company_info",
    "weekly_report.product_list": "query_weekly_report_product_list",
    "monthly_report.product_list": "query_monthly_report_product_list",
}
_MANIFEST_ACTION_REQUIREMENTS: Final[dict[str, OutboundActionType]] = {
    "material_pack.send": "send_material_pack",
    "weekly_report.send": "send_weekly_report",
    "monthly_report.send": "send_monthly_report",
}
_MANIFEST_RESOLVE_REQUIREMENTS: Final[dict[str, AdapterResolveType]] = {
    "material_pack.send": "material_pack",
    "weekly_report.send": "weekly_report",
    "monthly_report.send": "monthly_report",
    "sales.handoff": "sales_mention",
    "weekly_report.product_list": "weekly_report",
    "monthly_report.product_list": "monthly_report",
}


def compile_policy_manifest_v2(
    request: KernelReplyRequestV1,
    scope_authority: BusinessScopeAuthorityV1,
    ledger_summary: PolicyLedgerSummaryV1,
    *,
    group_recall_mode: RecallModeV1 = "shortcut",
) -> PolicyManifestV2:
    core = compile_policy_authority_core_v1(
        request,
        scope_authority,
        group_recall_mode=group_recall_mode,
    )
    return PolicyManifestV2.from_core(core, ledger_summary)


def compile_policy_authority_core_v1(
    request: KernelReplyRequestV1,
    scope_authority: BusinessScopeAuthorityV1,
    *,
    group_recall_mode: RecallModeV1 = "shortcut",
) -> PolicyAuthorityCoreV1:
    if request.business_scope != scope_authority.scope:
        raise PolicyManifestCompilationError("business_scope_authority_mismatch")
    if group_recall_mode not in {"shortcut", "advisory", "off"}:
        raise PolicyManifestCompilationError("invalid_group_recall_mode")
    scene: Literal["group", "direct"] = (
        "group" if kernel_scene_is_group(request) else "direct"
    )
    read_capabilities: tuple[ReadCapability, ...] = tuple(
        sorted(set(request.grants.read_capabilities))
    )
    if scene == "direct":
        read_capabilities = tuple(
            capability
            for capability in read_capabilities
            if capability == "query_internal_company_info"
            and request.business_scope.kind == "unscoped"
        )
    internal_company_knowledge_enabled = (
        "query_internal_company_info" in read_capabilities
    )
    manifest_refs = _eligible_manifest_refs(
        request,
        scene=scene,
        internal_company_knowledge_enabled=internal_company_knowledge_enabled,
    )
    action_capabilities = tuple(sorted(set(request.grants.outbound_actions)))
    mention_types = tuple(sorted(set(request.grants.mention_types)))
    if scene == "direct":
        action_capabilities = ()
        mention_types = ()
    allowed_resolves = _allowed_resolves(manifest_refs, scene)
    return PolicyAuthorityCoreV1(
        scene=scene,
        allowed_reply_modes=_policy_reply_modes(manifest_refs),
        eligible_capabilities=manifest_refs,
        allowed_read_capabilities=read_capabilities,
        allowed_outbound_actions=action_capabilities,
        allowed_mention_types=mention_types,
        allowed_adapter_resolves=allowed_resolves,
        internal_company_knowledge_enabled=internal_company_knowledge_enabled,
        recall_mode="off" if scene == "direct" else group_recall_mode,
        evidence_call_limit=min(len(allowed_resolves), 16),
        actions_allowed=bool(action_capabilities),
        mentions_allowed=bool(mention_types),
        material_pack_options=_canonical_material_options(request),
        effective_grants_hash=effective_grants_hash_v1(request),
        business_scope_hash=scope_authority.business_scope_hash,
    )


def _eligible_manifest_refs(
    request: KernelReplyRequestV1,
    *,
    scene: Literal["group", "direct"],
    internal_company_knowledge_enabled: bool,
) -> tuple[ManifestRefV1, ...]:
    read_grants = frozenset(request.grants.read_capabilities)
    action_grants = frozenset(request.grants.outbound_actions)
    mention_grants = frozenset(request.grants.mention_types)
    artifact_types = frozenset(
        artifact.type for artifact in kernel_available_artifacts(request)
    )
    eligible: list[ManifestRefV1] = []
    for manifest in CAPABILITY_MANIFEST_REGISTRY.list():
        manifest_id = manifest.manifest_id
        read_requirement = _MANIFEST_READ_REQUIREMENTS.get(manifest_id)
        action_requirement = _MANIFEST_ACTION_REQUIREMENTS.get(manifest_id)
        if read_requirement is not None and read_requirement not in read_grants:
            continue
        if (
            read_requirement == "query_internal_company_info"
            and not internal_company_knowledge_enabled
        ):
            continue
        if action_requirement is not None and action_requirement not in action_grants:
            continue
        if manifest_id == "sales.handoff" and "sales" not in mention_grants:
            continue
        if _artifact_is_missing(manifest_id, artifact_types):
            continue
        if manifest_id == "general.handoff" and scene != "direct":
            continue
        if scene == "direct" and manifest_id in {
            "material_pack.send",
            "weekly_report.send",
            "monthly_report.send",
            "sales.handoff",
            "weekly_report.product_list",
            "monthly_report.product_list",
        }:
            continue
        eligible.append(
            ManifestRefV1(
                manifest_id=manifest_id, manifest_version=manifest.manifest_version
            )
        )
    return tuple(eligible)


def _artifact_is_missing(manifest_id: str, artifact_types: frozenset[str]) -> bool:
    return (
        (manifest_id == "material_pack.send" and "material_pack" not in artifact_types)
        or (
            manifest_id in {"weekly_report.send", "weekly_report.product_list"}
            and "weekly_report" not in artifact_types
        )
        or (
            manifest_id in {"monthly_report.send", "monthly_report.product_list"}
            and "monthly_report" not in artifact_types
        )
    )


def _allowed_resolves(
    refs: tuple[ManifestRefV1, ...],
    scene: Literal["group", "direct"],
) -> tuple[AdapterResolveType, ...]:
    if scene == "direct":
        return ()
    return tuple(
        sorted(
            {
                resolve
                for ref in refs
                if (resolve := _MANIFEST_RESOLVE_REQUIREMENTS.get(ref.manifest_id))
                is not None
            }
        )
    )


def _policy_reply_modes(refs: tuple[ManifestRefV1, ...]) -> tuple[ResponseMode, ...]:
    manifest_ids = frozenset(ref.manifest_id for ref in refs)
    modes: set[ResponseMode] = {
        "clarification",
        "handoff",
        "no_reply",
        "refusal",
        "smalltalk",
        "unable",
    }
    if manifest_ids & {
        "material_pack.send",
        "weekly_report.send",
        "monthly_report.send",
    }:
        modes.add("action")
    if manifest_ids & {
        "answer_internal_company_knowledge",
        "weekly_report.product_list",
        "monthly_report.product_list",
    }:
        modes.add("knowledge_answer")
    return tuple(sorted(modes))


def _canonical_material_options(request: KernelReplyRequestV1) -> tuple[str, ...]:
    for artifact in kernel_available_artifacts(request):
        if artifact.type == "material_pack":
            return tuple(artifact.options)
    return ()
