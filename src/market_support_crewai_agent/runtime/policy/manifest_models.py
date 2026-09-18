from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ConfigDict, Field, model_validator

from market_support_crewai_agent.runtime.hashing import sha256_frame
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.capabilities import (
    ManifestRefV1,
    ReadCapability,
    ResponseMode,
)
from market_support_crewai_agent.runtime.policy.direct_constraints import (
    validate_direct_policy_authority,
    validate_direct_policy_ledger,
)
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveType,
    OutboundActionType,
    ReplyMentionType,
)


class _FrozenPolicyModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class PolicyManifestCompilationError(ValueError):
    """Raised when policy authority cannot produce a valid PolicyManifestV2."""


RecallModeV1 = Literal["off", "advisory", "shortcut"]


class PolicyLedgerSummaryV1(_FrozenPolicyModel):
    recent_executed_count: int = Field(ge=0, le=20)
    recent_artifact_types: tuple[str, ...] = Field(max_length=3)
    has_recent_executed_actions: bool

    @model_validator(mode="after")
    def _validate_canonical(self) -> PolicyLedgerSummaryV1:
        if tuple(sorted(set(self.recent_artifact_types))) != self.recent_artifact_types:
            raise PolicyManifestCompilationError(
                "policy_ledger_artifact_types_not_canonical"
            )
        if self.has_recent_executed_actions != (self.recent_executed_count > 0):
            raise PolicyManifestCompilationError(
                "policy_ledger_execution_summary_mismatch"
            )
        return self


class PolicyAuthorityCoreV1(_FrozenPolicyModel):
    contract_version: Literal["policy-authority-core.v1"] = "policy-authority-core.v1"
    scene: Literal["group", "direct"]
    scene_ceiling_version: Literal["scene-ceilings.v1"] = "scene-ceilings.v1"
    allowed_reply_modes: tuple[ResponseMode, ...] = Field(min_length=1, max_length=8)
    eligible_capabilities: tuple[ManifestRefV1, ...] = Field(
        min_length=1, max_length=32
    )
    allowed_read_capabilities: tuple[ReadCapability, ...] = Field(max_length=7)
    allowed_outbound_actions: tuple[OutboundActionType, ...] = Field(max_length=3)
    allowed_mention_types: tuple[ReplyMentionType, ...] = Field(max_length=1)
    allowed_adapter_resolves: tuple[AdapterResolveType, ...] = Field(max_length=4)
    internal_company_knowledge_enabled: bool
    recall_mode: RecallModeV1
    evidence_call_limit: int = Field(ge=0, le=16)
    actions_allowed: bool
    mentions_allowed: bool
    material_pack_options: tuple[str, ...]
    effective_grants_hash: str = Field(pattern=r"^grh1:[0-9a-f]{64}$")
    business_scope_hash: str = Field(pattern=r"^bsh1:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_core(self) -> PolicyAuthorityCoreV1:
        if tuple(sorted(set(self.allowed_reply_modes))) != self.allowed_reply_modes:
            raise PolicyManifestCompilationError("policy_reply_modes_not_canonical")
        if (
            tuple(sorted(set(self.allowed_read_capabilities)))
            != self.allowed_read_capabilities
        ):
            raise PolicyManifestCompilationError(
                "policy_read_capabilities_not_canonical"
            )
        if (
            tuple(sorted(set(self.allowed_outbound_actions)))
            != self.allowed_outbound_actions
        ):
            raise PolicyManifestCompilationError(
                "policy_outbound_actions_not_canonical"
            )
        if tuple(sorted(set(self.allowed_mention_types))) != self.allowed_mention_types:
            raise PolicyManifestCompilationError("policy_mentions_not_canonical")
        if (
            tuple(sorted(set(self.allowed_adapter_resolves)))
            != self.allowed_adapter_resolves
        ):
            raise PolicyManifestCompilationError(
                "policy_adapter_resolves_not_canonical"
            )
        if len({ref.manifest_id for ref in self.eligible_capabilities}) != len(
            self.eligible_capabilities
        ):
            raise PolicyManifestCompilationError("policy_duplicate_manifest_ref")
        if self.actions_allowed != bool(self.allowed_outbound_actions):
            raise PolicyManifestCompilationError("policy_actions_allowed_mismatch")
        if self.mentions_allowed != bool(self.allowed_mention_types):
            raise PolicyManifestCompilationError("policy_mentions_allowed_mismatch")
        if self.internal_company_knowledge_enabled != (
            "query_internal_company_info" in self.allowed_read_capabilities
        ):
            raise PolicyManifestCompilationError(
                "policy_internal_knowledge_grant_mismatch"
            )
        if self.scene == "direct":
            _validate_direct_policy_core(self)
        return self


class PolicyManifestV2(_FrozenPolicyModel):
    contract_version: Literal["policy-manifest.v2"] = "policy-manifest.v2"
    policy_id: str = Field(pattern=r"^pol1:[0-9a-f]{64}$")
    scene: Literal["group", "direct"]
    scene_ceiling_version: Literal["scene-ceilings.v1"] = "scene-ceilings.v1"
    allowed_reply_modes: tuple[ResponseMode, ...] = Field(min_length=1, max_length=8)
    eligible_capabilities: tuple[ManifestRefV1, ...] = Field(
        min_length=1, max_length=32
    )
    allowed_read_capabilities: tuple[ReadCapability, ...] = Field(max_length=7)
    allowed_outbound_actions: tuple[OutboundActionType, ...] = Field(max_length=3)
    allowed_mention_types: tuple[ReplyMentionType, ...] = Field(max_length=1)
    allowed_adapter_resolves: tuple[AdapterResolveType, ...] = Field(max_length=4)
    internal_company_knowledge_enabled: bool
    recall_mode: RecallModeV1
    evidence_call_limit: int = Field(ge=0, le=16)
    actions_allowed: bool
    mentions_allowed: bool
    material_pack_options: tuple[str, ...]
    ledger_summary: PolicyLedgerSummaryV1
    effective_grants_hash: str = Field(pattern=r"^grh1:[0-9a-f]{64}$")
    business_scope_hash: str = Field(pattern=r"^bsh1:[0-9a-f]{64}$")
    state_admission_hash: str = Field(pattern=r"^par1:[0-9a-f]{64}$")

    @classmethod
    def from_core(
        cls,
        core: PolicyAuthorityCoreV1,
        ledger_summary: PolicyLedgerSummaryV1,
    ) -> PolicyManifestV2:
        draft = cls.model_construct(
            policy_id="pol1:" + "0" * 64,
            scene=core.scene,
            scene_ceiling_version=core.scene_ceiling_version,
            allowed_reply_modes=core.allowed_reply_modes,
            eligible_capabilities=core.eligible_capabilities,
            allowed_read_capabilities=core.allowed_read_capabilities,
            allowed_outbound_actions=core.allowed_outbound_actions,
            allowed_mention_types=core.allowed_mention_types,
            allowed_adapter_resolves=core.allowed_adapter_resolves,
            internal_company_knowledge_enabled=core.internal_company_knowledge_enabled,
            recall_mode=core.recall_mode,
            evidence_call_limit=core.evidence_call_limit,
            actions_allowed=core.actions_allowed,
            mentions_allowed=core.mentions_allowed,
            material_pack_options=core.material_pack_options,
            ledger_summary=ledger_summary,
            effective_grants_hash=core.effective_grants_hash,
            business_scope_hash=core.business_scope_hash,
            state_admission_hash=state_admission_hash_v1(core),
        )
        payload = draft.model_dump(mode="json", exclude_none=False)
        payload["policy_id"] = policy_manifest_id_v2(draft)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def _validate_manifest_boundary(self) -> PolicyManifestV2:
        core = _policy_authority_core_from_manifest(self)
        if self.scene == "direct":
            validate_direct_policy_ledger(self.ledger_summary)
        if self.state_admission_hash != state_admission_hash_v1(core):
            raise PolicyManifestCompilationError("policy_state_admission_hash_mismatch")
        if self.policy_id != policy_manifest_id_v2(self):
            raise PolicyManifestCompilationError("policy_manifest_id_mismatch")
        return self


def _policy_authority_core_from_manifest(
    policy: PolicyManifestV2,
) -> PolicyAuthorityCoreV1:
    return PolicyAuthorityCoreV1(
        scene=policy.scene,
        scene_ceiling_version=policy.scene_ceiling_version,
        allowed_reply_modes=policy.allowed_reply_modes,
        eligible_capabilities=policy.eligible_capabilities,
        allowed_read_capabilities=policy.allowed_read_capabilities,
        allowed_outbound_actions=policy.allowed_outbound_actions,
        allowed_mention_types=policy.allowed_mention_types,
        allowed_adapter_resolves=policy.allowed_adapter_resolves,
        internal_company_knowledge_enabled=policy.internal_company_knowledge_enabled,
        recall_mode=policy.recall_mode,
        evidence_call_limit=policy.evidence_call_limit,
        actions_allowed=policy.actions_allowed,
        mentions_allowed=policy.mentions_allowed,
        material_pack_options=policy.material_pack_options,
        effective_grants_hash=policy.effective_grants_hash,
        business_scope_hash=policy.business_scope_hash,
    )


def _validate_direct_policy_core(core: PolicyAuthorityCoreV1) -> None:
    validate_direct_policy_authority(core)


def effective_grants_hash_v1(request: KernelReplyRequestV1) -> str:
    return sha256_frame(
        "effective-grants.v1",
        request.grants.model_dump(mode="json", exclude_none=False),
        prefix="grh1",
    )


def state_admission_hash_v1(core: PolicyAuthorityCoreV1) -> str:
    return sha256_frame(
        "state-admission-policy.v1",
        core.model_dump(mode="json", exclude_none=False),
        prefix="par1",
    )


def policy_manifest_id_v2(policy: PolicyManifestV2) -> str:
    payload = policy.model_dump(mode="json", exclude_none=False)
    del payload["policy_id"]
    return sha256_frame("policy-manifest.v2", payload, prefix="pol1")


def policy_ledger_summary_v1(
    recent_artifact_types: tuple[str, ...],
    recent_executed_count: int,
) -> PolicyLedgerSummaryV1:
    return PolicyLedgerSummaryV1(
        recent_executed_count=min(recent_executed_count, 20),
        recent_artifact_types=tuple(sorted(set(recent_artifact_types))),
        has_recent_executed_actions=recent_executed_count > 0,
    )
