from __future__ import annotations

from typing import Annotated, ClassVar, Literal, TypeAlias, assert_never

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    ManifestRefV1,
    ResponseMode,
)
from market_support_crewai_agent.runtime.policy.manifest import RecallModeV1
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import (
    AdapterResolveType,
    AvailableArtifactType,
    ChannelType,
    OutboundActionType,
    ReadCapability,
    ReplyMentionType,
)


class _FrozenPolicyView(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class DistributionBusinessScopeViewV1(_FrozenPolicyView):
    kind: Literal["distribution"] = "distribution"
    business_scope_ref: str = Field(pattern=r"^bsr:[0-9a-f]{32}$")
    channel_type: ChannelType
    dist_channel_name: str = Field(min_length=1, max_length=120)
    artifact_types: tuple[AvailableArtifactType, ...] = Field(max_length=3)

    @field_validator("dist_channel_name")
    @classmethod
    def validate_channel_name(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ContextViewInvariantError("business_scope_channel_control_character")
        return value

    @field_validator("artifact_types")
    @classmethod
    def validate_artifact_types(
        cls,
        values: tuple[AvailableArtifactType, ...],
    ) -> tuple[AvailableArtifactType, ...]:
        if values != tuple(sorted(set(values))):
            raise ContextViewInvariantError("business_scope_artifacts_not_canonical")
        return values


class UnscopedBusinessScopeViewV1(_FrozenPolicyView):
    kind: Literal["unscoped"] = "unscoped"


BusinessScopeViewV1: TypeAlias = Annotated[
    DistributionBusinessScopeViewV1 | UnscopedBusinessScopeViewV1,
    Field(discriminator="kind"),
]


class EffectivePolicyViewV1(_FrozenPolicyView):
    contract_version: Literal["effective-policy-view.v1"] = "effective-policy-view.v1"
    policy_id: str = Field(pattern=r"^pol1:[0-9a-f]{64}$")
    scene: Literal["group", "direct"]
    eligible_capabilities: tuple[ManifestRefV1, ...] = Field(max_length=32)
    read_capabilities: tuple[ReadCapability, ...] = Field(max_length=7)
    internal_company_knowledge_enabled: bool
    outbound_actions: tuple[OutboundActionType, ...] = Field(max_length=3)
    mention_types: tuple[ReplyMentionType, ...] = Field(max_length=1)
    adapter_resolves: tuple[AdapterResolveType, ...] = Field(max_length=4)
    allowed_reply_modes: tuple[ResponseMode, ...] = Field(max_length=8)
    recall_mode: RecallModeV1
    evidence_call_limit: int = Field(ge=0, le=16)
    actions_allowed: bool
    mentions_allowed: bool

    @model_validator(mode="after")
    def validate_canonical_policy(self) -> EffectivePolicyViewV1:
        canonical_tuples = (
            self.read_capabilities,
            self.outbound_actions,
            self.mention_types,
            self.adapter_resolves,
            self.allowed_reply_modes,
        )
        if any(values != tuple(sorted(set(values))) for values in canonical_tuples):
            raise ContextViewInvariantError("effective_policy_values_not_canonical")
        manifest_ids = tuple(ref.manifest_id for ref in self.eligible_capabilities)
        if len(set(manifest_ids)) != len(manifest_ids):
            raise ContextViewInvariantError("effective_policy_duplicate_manifest_ref")
        knowledge_enabled = "query_internal_company_info" in self.read_capabilities
        if self.internal_company_knowledge_enabled != knowledge_enabled:
            raise ContextViewInvariantError("effective_policy_knowledge_flag_mismatch")
        match self.scene:
            case "group":
                pass
            case "direct":
                direct_violation = (
                    self.recall_mode != "off"
                    or bool(self.outbound_actions)
                    or bool(self.mention_types)
                    or self.actions_allowed
                    or self.mentions_allowed
                    or "sales_mention" in self.adapter_resolves
                )
                if direct_violation:
                    raise ContextViewInvariantError(
                        "effective_policy_direct_ceiling_violation"
                    )
            case unreachable:
                assert_never(unreachable)
        return self
