from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from market_support_crewai_agent.runtime.action_ledger import ActionLedgerRecord
from market_support_crewai_agent.schemas import (
    AdapterResolveType,
    ChannelType,
    MaterialType,
    ReplyKind,
    ReplyRequest,
    SideEffectActionType,
)

ReadCapability = Literal[
    "resolve_material_pack",
    "resolve_weekly_report",
    "resolve_monthly_report",
    "resolve_sales_mention",
    "query_internal_company_info",
]
BusinessCheck = Literal[
    "resolve_strategy_alias",
    "check_material_pack_resolvable",
    "check_weekly_report_resolvable",
    "check_monthly_report_resolvable",
    "check_report_contains_strategy",
    "check_report_generation_scope",
    "check_channel_permission",
]
ForbiddenClaimCategory = Literal[
    "unsupported_report_scope_exclusion",
    "unsupported_report_non_inclusion",
    "pre_execution_success_claim",
    "sent_claim_without_ledger_evidence",
    "internal_locator_leak",
]

_ACTION_BY_MATERIAL: dict[MaterialType, SideEffectActionType] = {
    "material": "send_material_pack",
    "weekly": "send_weekly_report",
    "monthly": "send_monthly_report",
}
_READ_CAPABILITY_BY_RESOLVE: dict[AdapterResolveType, ReadCapability] = {
    "material_pack": "resolve_material_pack",
    "weekly_report": "resolve_weekly_report",
    "monthly_report": "resolve_monthly_report",
    "sales_mention": "resolve_sales_mention",
}
_DOCUMENT_READ_CAPABILITIES: frozenset[ReadCapability] = frozenset(
    {
        "query_internal_company_info",
    }
)
_ALL_REPLY_KINDS: frozenset[ReplyKind] = frozenset(
    {
        "answer",
        "clarification",
        "human_handoff",
        "unable_to_answer",
        "no_reply",
    }
)
_ALL_RESOLVES: frozenset[AdapterResolveType] = frozenset(
    {
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    }
)
_ALL_BUSINESS_CHECKS: frozenset[BusinessCheck] = frozenset(
    {
        "resolve_strategy_alias",
        "check_material_pack_resolvable",
        "check_weekly_report_resolvable",
        "check_monthly_report_resolvable",
        "check_report_contains_strategy",
        "check_report_generation_scope",
        "check_channel_permission",
    }
)
_DEFAULT_FORBIDDEN_CLAIMS: frozenset[ForbiddenClaimCategory] = frozenset(
    {
        "unsupported_report_scope_exclusion",
        "unsupported_report_non_inclusion",
        "pre_execution_success_claim",
        "sent_claim_without_ledger_evidence",
        "internal_locator_leak",
    }
)


@dataclass(frozen=True)
class RepairPolicy:
    allow_repair: bool = True
    max_attempts: int = 1
    fallback_reply_kind: ReplyKind = "unable_to_answer"


@dataclass(frozen=True)
class LedgerSummary:
    recent_executed_count: int = 0
    recent_material_types: tuple[MaterialType, ...] = ()
    recent_strategies: tuple[str, ...] = ()
    recent_versions: tuple[str, ...] = ()

    @property
    def has_recent_executed_actions(self) -> bool:
        return self.recent_executed_count > 0

    def to_prompt_dict(self) -> dict:
        return {
            "has_recent_executed_actions": self.has_recent_executed_actions,
            "recent_executed_count": self.recent_executed_count,
            "recent_material_types": list(self.recent_material_types),
            "recent_strategies": list(self.recent_strategies),
            "recent_versions": list(self.recent_versions),
        }


@dataclass(frozen=True)
class PolicyManifest:
    policy_id: str
    allowed_reply_kinds: frozenset[ReplyKind]
    allowed_side_effect_actions: frozenset[SideEffectActionType]
    required_adapter_resolves: frozenset[AdapterResolveType]
    allowed_read_capabilities: frozenset[ReadCapability]
    allowed_business_checks: frozenset[BusinessCheck]
    forbidden_claim_categories: frozenset[ForbiddenClaimCategory]
    ledger_summary: LedgerSummary = field(default_factory=LedgerSummary)
    evidence_call_limit: int = 4
    repair_policy: RepairPolicy = field(default_factory=RepairPolicy)


def compile_policy(
        request: ReplyRequest | None,
        ledger_summary: LedgerSummary | None = None,
        doc_mcp_enabled: bool = False,
        doc_mcp_allowed_channel_types: tuple[ChannelType, ...] = ("bank", "non_bank"),
) -> PolicyManifest:
    materials = (
        tuple(request.available_materials)
        if request is not None
        else ("material", "weekly", "monthly")
    )
    allowed_actions = frozenset(
        _ACTION_BY_MATERIAL[material]
        for material in materials
        if material in _ACTION_BY_MATERIAL
    )
    policy_scope = request.channel_type if request is not None else "compat"

    read_capabilities = frozenset(
        _READ_CAPABILITY_BY_RESOLVE[resolve_type]
        for resolve_type in _ALL_RESOLVES
    )
    if _doc_mcp_allowed_for_request(
        request,
        doc_mcp_enabled,
        doc_mcp_allowed_channel_types,
    ):
        read_capabilities = read_capabilities | _DOCUMENT_READ_CAPABILITIES

    return PolicyManifest(
        policy_id=f"support-reply-policy.v1:{policy_scope}",
        allowed_reply_kinds=_ALL_REPLY_KINDS,
        allowed_side_effect_actions=allowed_actions,
        required_adapter_resolves=_ALL_RESOLVES,
        allowed_read_capabilities=read_capabilities,
        allowed_business_checks=_ALL_BUSINESS_CHECKS,
        forbidden_claim_categories=_DEFAULT_FORBIDDEN_CLAIMS,
        ledger_summary=ledger_summary or LedgerSummary(),
        evidence_call_limit=len(read_capabilities),
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
        recent_strategies=tuple(
            _ordered_unique(
                execution.strategy.strip()
                for execution in executed
                if execution.strategy and execution.strategy.strip()
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


def _ordered_unique(values) -> list:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
