from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import get_args

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.runtime.domain.capabilities import (
    CapabilityName,
    ReadCapability,
    ResponseMode,
    capability_by_name,
)
from market_support_crewai_agent.schemas import (
    AdapterResolveType,
    AvailableArtifact,
    ChannelType,
    MaterialType,
    ReplyRequest,
    SideEffectActionType,
)

_DEFAULT_REPLY_MODES: frozenset[ResponseMode] = frozenset(
    mode
    for mode in get_args(ResponseMode)
    if mode != "knowledge_answer"
)


@dataclass(frozen=True)
class LedgerSummary:
    recent_executed_count: int = 0
    recent_material_types: tuple[MaterialType, ...] = ()
    recent_material_pack_options: tuple[str, ...] = ()
    recent_versions: tuple[str, ...] = ()

    @property
    def has_recent_executed_actions(self) -> bool:
        return self.recent_executed_count > 0

    def to_prompt_dict(self) -> dict:
        return {
            "has_recent_executed_actions": self.has_recent_executed_actions,
            "recent_executed_count": self.recent_executed_count,
            "recent_material_types": list(self.recent_material_types),
            "recent_material_pack_options": list(self.recent_material_pack_options),
            "recent_versions": list(self.recent_versions),
        }


@dataclass(frozen=True)
class PolicyManifest:
    policy_id: str
    allowed_reply_modes: frozenset[ResponseMode]
    allowed_capabilities: frozenset[CapabilityName]
    allowed_side_effect_actions: frozenset[SideEffectActionType]
    allowed_read_capabilities: frozenset[ReadCapability]
    allowed_adapter_resolves: frozenset[AdapterResolveType]
    material_pack_options: tuple[str, ...] = ()
    ledger_summary: LedgerSummary = field(default_factory=LedgerSummary)
    evidence_call_limit: int = 4


def compile_policy(
        request: ReplyRequest | None,
        ledger_summary: LedgerSummary | None = None,
        doc_mcp_enabled: bool = False,
        doc_mcp_allowed_channel_types: tuple[ChannelType, ...] = ("bank", "non_bank"),
) -> PolicyManifest:
    policy_scope = request.channel_type if request is not None else "default"
    allowed_capabilities: set[CapabilityName] = {"sales_mention"}
    allowed_capabilities.update(_artifact_capabilities(request.available_artifacts if request else ()))

    if _doc_mcp_allowed_for_request(
        request,
        doc_mcp_enabled,
        doc_mcp_allowed_channel_types,
    ):
        allowed_capabilities.add("document_context")

    if request is not None and request.allowed_read_capabilities:
        adapter_allowed = set(request.allowed_read_capabilities)
        allowed_capabilities = {
            capability_name
            for capability_name in allowed_capabilities
            if (
                capability := capability_by_name(capability_name)
            ) is not None
            and (
                capability.read_capability is None
                or capability.read_capability in adapter_allowed
            )
        }

    allowed_actions = {
        capability.side_effect_action_type
        for capability_name in allowed_capabilities
        if (
            capability := capability_by_name(capability_name)
        ) is not None and capability.side_effect_action_type is not None
    }

    allowed_read_capabilities = frozenset(
        capability.read_capability
        for capability_name in allowed_capabilities
        if (
            capability := capability_by_name(capability_name)
        ) is not None and capability.read_capability is not None
    )
    allowed_adapter_resolves = frozenset(
        capability.resolve_type
        for capability_name in allowed_capabilities
        if (
            capability := capability_by_name(capability_name)
        ) is not None and capability.resolve_type is not None
    )
    allowed_reply_modes = _DEFAULT_REPLY_MODES
    if "document_context" in allowed_capabilities or {
        "weekly_report",
        "monthly_report",
    }.intersection(allowed_capabilities):
        allowed_reply_modes = allowed_reply_modes | frozenset({"knowledge_answer"})

    return PolicyManifest(
        policy_id=f"support-reply-policy:{policy_scope}",
        allowed_reply_modes=allowed_reply_modes,
        allowed_capabilities=frozenset(allowed_capabilities),
        allowed_side_effect_actions=frozenset(allowed_actions),
        allowed_read_capabilities=allowed_read_capabilities,
        allowed_adapter_resolves=allowed_adapter_resolves,
        material_pack_options=tuple(
            _ordered_unique(
                option.strip()
                for option in _material_pack_options(request.available_artifacts if request else ())
                if option and option.strip()
            )
        ),
        ledger_summary=ledger_summary or LedgerSummary(),
        evidence_call_limit=len(allowed_read_capabilities),
    )


def ledger_summary_from_action_history(
        action_history: Sequence[ActionLedgerRecord] | None,
) -> LedgerSummary:
    executed = [
        record.execution
        for record in action_history or []
        if record.execution.status == "executed"
    ]
    return LedgerSummary(
        recent_executed_count=len(executed),
        recent_material_types=tuple(
            _ordered_unique(
                execution.material_type
                for execution in executed
                if execution.material_type is not None
            )
        ),
        recent_material_pack_options=tuple(
            _ordered_unique(
                execution.material_pack_option.strip()
                for execution in executed
                if execution.material_pack_option
                and execution.material_pack_option.strip()
            )
        ),
        recent_versions=tuple(
            _ordered_unique(
                execution.version.strip()
                for execution in executed
                if execution.version and execution.version.strip()
            )
        ),
    )


def _doc_mcp_allowed_for_request(
    request: ReplyRequest | None,
    doc_mcp_enabled: bool,
    allowed_channel_types: tuple[ChannelType, ...],
) -> bool:
    if not doc_mcp_enabled:
        return False
    if request is None:
        return bool(allowed_channel_types)
    return request.channel_type in allowed_channel_types


def _artifact_capabilities(artifacts: Sequence[AvailableArtifact]) -> set[CapabilityName]:
    mapping: dict[str, CapabilityName] = {
        "material_pack": "material_pack",
        "weekly_report": "weekly_report",
        "monthly_report": "monthly_report",
    }
    return {capability for artifact in artifacts if (capability := mapping.get(artifact.type))}


def _material_pack_options(artifacts: Sequence[AvailableArtifact]) -> list[str]:
    for artifact in artifacts:
        if artifact.type == "material_pack":
            return list(artifact.options)
    return []


def _ordered_unique(values) -> list:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
