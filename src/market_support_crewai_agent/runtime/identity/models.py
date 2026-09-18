from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from market_support_crewai_agent.schemas.conversation import (
    AvailableArtifactV2,
    BusinessScopeV1,
    DirectPresentationV1,
    DistributionScopeV1,
    ExplicitPrincipalGrantsV1,
    GroupPresentationV1,
)
from market_support_crewai_agent.schemas.type_ids import ChannelType, ReadCapability


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class ConversationStateKey:
    surface: Literal["wecom"]
    adapter_namespace: str
    tenant_ref: str
    scene: Literal["group", "direct"]
    subject_ref: str
    principal_ref: str


class AuthenticatedAdapterCallerV1(StrictFrozenModel):
    contract_version: Literal["authenticated-adapter-caller.v1"] = (
        "authenticated-adapter-caller.v1"
    )
    adapter_namespace: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,63}$")
    auth_mode: Literal["api_key"] = "api_key"
    authenticated: Literal[True] = True


class VerifiedConversationIdentityV1(StrictFrozenModel):
    contract_version: Literal["verified-conversation-identity.v1"] = (
        "verified-conversation-identity.v1"
    )
    surface: Literal["wecom"] = "wecom"
    scene: Literal["group", "direct"]
    tenant_ref: str
    subject_ref: str
    principal_ref: str


class KernelReplyRequestV1(StrictFrozenModel):
    contract_version: Literal["kernel-reply-request.v1"] = "kernel-reply-request.v1"
    request_id: str
    replay_eligible: bool
    message: str
    context_id: str | None = None
    identity: VerifiedConversationIdentityV1
    presentation: GroupPresentationV1 | DirectPresentationV1
    business_scope: BusinessScopeV1
    grants: ExplicitPrincipalGrantsV1


class PolicyRequestKernelV1(StrictFrozenModel):
    contract_version: Literal["policy-request-kernel.v1"] = "policy-request-kernel.v1"
    adapter_namespace: str
    replay_eligible: bool
    message: str
    context_id: str | None = None
    identity: VerifiedConversationIdentityV1
    presentation: GroupPresentationV1 | DirectPresentationV1
    business_scope: BusinessScopeV1
    grants: ExplicitPrincipalGrantsV1


class VerifiedRequestEnvelopeV1(StrictFrozenModel):
    contract_version: Literal["verified-request-envelope.v1"] = (
        "verified-request-envelope.v1"
    )
    caller: AuthenticatedAdapterCallerV1
    request: KernelReplyRequestV1
    state_key: ConversationStateKey
    state_key_ref: str


def kernel_subject_ref(request: KernelReplyRequestV1) -> str:
    return request.identity.subject_ref


def kernel_principal_ref(request: KernelReplyRequestV1) -> str:
    return request.identity.principal_ref


def kernel_scene_is_group(request: KernelReplyRequestV1) -> bool:
    return request.identity.scene == "group"


def kernel_conversation_name(request: KernelReplyRequestV1) -> str:
    if isinstance(request.presentation, GroupPresentationV1):
        return request.presentation.conversation_name
    return "私聊"


def kernel_principal_name(request: KernelReplyRequestV1) -> str:
    return request.presentation.principal_name or "用户"


def kernel_distribution_name(request: KernelReplyRequestV1) -> str:
    if isinstance(request.business_scope, DistributionScopeV1):
        return request.business_scope.dist_channel_name
    return "未指定渠道"


def kernel_channel_type(request: KernelReplyRequestV1) -> ChannelType:
    if isinstance(request.business_scope, DistributionScopeV1):
        return request.business_scope.channel_type
    return "unknown"


def kernel_available_artifacts(
    request: KernelReplyRequestV1,
) -> tuple[AvailableArtifactV2, ...]:
    if isinstance(request.business_scope, DistributionScopeV1):
        return tuple(request.business_scope.available_artifacts)
    return ()


def kernel_read_capabilities(
    request: KernelReplyRequestV1,
) -> tuple[ReadCapability, ...]:
    return tuple(request.grants.read_capabilities)
