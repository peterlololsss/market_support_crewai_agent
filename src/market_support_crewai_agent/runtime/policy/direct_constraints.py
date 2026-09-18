from __future__ import annotations

from typing import Final, Literal, Protocol, override

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_hash_v1,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    ManifestRefV1,
    ReadCapability,
    ResponseMode,
)
from market_support_crewai_agent.schemas.conversation import UnscopedScopeV1
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveType,
    OutboundActionType,
    ReplyMentionType,
)


class DirectPolicyCeilingError(ValueError):
    @override
    def __str__(self) -> str:
        return "direct_policy_ceiling_violation"


class DirectPolicyAuthorityFields(Protocol):
    allowed_reply_modes: tuple[ResponseMode, ...]
    eligible_capabilities: tuple[ManifestRefV1, ...]
    allowed_read_capabilities: tuple[ReadCapability, ...]
    allowed_outbound_actions: tuple[OutboundActionType, ...]
    allowed_mention_types: tuple[ReplyMentionType, ...]
    allowed_adapter_resolves: tuple[AdapterResolveType, ...]
    internal_company_knowledge_enabled: bool
    recall_mode: Literal["off", "advisory", "shortcut"]
    evidence_call_limit: int
    actions_allowed: bool
    mentions_allowed: bool
    material_pack_options: tuple[str, ...]
    business_scope_hash: str


class DirectPolicyLedgerFields(Protocol):
    recent_executed_count: int
    recent_artifact_types: tuple[str, ...]
    has_recent_executed_actions: bool


_DIRECT_UNSCOPED_BUSINESS_SCOPE_HASH: Final[str] = business_scope_hash_v1(
    UnscopedScopeV1(kind="unscoped")
)
_DIRECT_GENERAL_REPLY_MODES: Final[tuple[ResponseMode, ...]] = (
    "clarification",
    "handoff",
    "no_reply",
    "refusal",
    "smalltalk",
    "unable",
)
_DIRECT_KNOWLEDGE_REPLY_MODES: Final[tuple[ResponseMode, ...]] = (
    "clarification",
    "handoff",
    "knowledge_answer",
    "no_reply",
    "refusal",
    "smalltalk",
    "unable",
)
_DIRECT_GENERAL_MANIFEST_REFS: Final[tuple[ManifestRefV1, ...]] = (
    ManifestRefV1(
        manifest_id="general.clarification",
        manifest_version="2026-07-18.1",
    ),
    ManifestRefV1(
        manifest_id="general.abstention",
        manifest_version="2026-07-18.1",
    ),
    ManifestRefV1(
        manifest_id="general.refusal",
        manifest_version="2026-07-18.1",
    ),
    ManifestRefV1(
        manifest_id="general.smalltalk",
        manifest_version="2026-07-18.1",
    ),
    ManifestRefV1(
        manifest_id="general.no_reply",
        manifest_version="2026-07-18.1",
    ),
    ManifestRefV1(
        manifest_id="general.handoff",
        manifest_version="2026-07-18.1",
    ),
)
_DIRECT_KNOWLEDGE_MANIFEST_REFS: Final[tuple[ManifestRefV1, ...]] = (
    *_DIRECT_GENERAL_MANIFEST_REFS[:-1],
    ManifestRefV1(
        manifest_id="answer_internal_company_knowledge",
        manifest_version="2026-07-18.1",
    ),
    _DIRECT_GENERAL_MANIFEST_REFS[-1],
)


def validate_direct_policy_authority(fields: DirectPolicyAuthorityFields) -> None:
    expected_reply_modes = (
        _DIRECT_KNOWLEDGE_REPLY_MODES
        if fields.internal_company_knowledge_enabled
        else _DIRECT_GENERAL_REPLY_MODES
    )
    expected_manifest_refs = (
        _DIRECT_KNOWLEDGE_MANIFEST_REFS
        if fields.internal_company_knowledge_enabled
        else _DIRECT_GENERAL_MANIFEST_REFS
    )
    expected_read_capabilities = (
        ("query_internal_company_info",)
        if fields.internal_company_knowledge_enabled
        else ()
    )
    if (
        fields.allowed_reply_modes != expected_reply_modes
        or fields.eligible_capabilities != expected_manifest_refs
        or fields.allowed_read_capabilities != expected_read_capabilities
        or fields.recall_mode != "off"
        or fields.allowed_outbound_actions
        or fields.allowed_mention_types
        or fields.allowed_adapter_resolves
        or fields.actions_allowed
        or fields.mentions_allowed
        or fields.material_pack_options
        or fields.evidence_call_limit != 0
        or fields.business_scope_hash != _DIRECT_UNSCOPED_BUSINESS_SCOPE_HASH
    ):
        raise DirectPolicyCeilingError


def validate_direct_policy_ledger(fields: DirectPolicyLedgerFields) -> None:
    if (
        fields.recent_executed_count != 0
        or fields.recent_artifact_types
        or fields.has_recent_executed_actions
    ):
        raise DirectPolicyCeilingError
