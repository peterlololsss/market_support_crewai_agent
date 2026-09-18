from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Annotated, Literal, TypeVar

from pydantic import Field, field_validator, model_validator

from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.type_ids import (
    AvailableArtifactType,
    ChannelType,
    OutboundActionType,
    ReadCapability,
    ReplyMentionType,
)

_OPAQUE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")
_RESERVED_OPAQUE_TOKENS = frozenset(
    {"unknown", "none", "null", "placeholder", "na", "-", "0"}
)
TCanonicalString = TypeVar("TCanonicalString", bound=str)


def validate_prefixed_opaque_ref(
    value: str,
    *,
    prefix: str,
    field_name: str,
) -> str:
    trimmed = value.strip()
    if trimmed != value or any(ord(char) < 32 for char in trimmed):
        raise ValueError(f"{field_name} must be a trimmed opaque ref")
    if not trimmed.startswith(prefix):
        raise ValueError(f"{field_name} must start with {prefix}")
    token = trimmed[len(prefix) :]
    if (
        not _OPAQUE_REF_PATTERN.fullmatch(token)
        or token.casefold() in _RESERVED_OPAQUE_TOKENS
        or token == "contract-probe"
        or token.startswith("contract-probe-")
    ):
        raise ValueError(f"{field_name} must be a valid opaque ref")
    return trimmed


def validate_canonical_tenant_ref(value: str, *, field_name: str = "tenant_ref") -> str:
    return validate_prefixed_opaque_ref(
        value,
        prefix="tenant:",
        field_name=field_name,
    )


class AvailableArtifactV2(StrictModel):
    type: AvailableArtifactType
    options: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("options")
    @classmethod
    def validate_options(cls, values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            option = value.strip()
            if not option or len(option) > 80:
                raise ValueError("available artifact options must be 1-80 chars")
            if option in seen:
                raise ValueError("available artifact options must be unique")
            seen.add(option)
            output.append(option)
        return output

    @model_validator(mode="after")
    def validate_options_only_for_material_pack_v2(self):
        if self.type != "material_pack" and self.options:
            raise ValueError(
                "available_artifacts options are only valid for material_pack"
            )
        return self


class GroupConversationIdentityV1(StrictModel):
    contract_version: Literal["conversation-identity.v1"]
    surface: Literal["wecom"]
    scene: Literal["group"]
    tenant_ref: str
    group_ref: str
    principal_ref: str

    @field_validator("tenant_ref")
    @classmethod
    def validate_tenant_ref(cls, value: str) -> str:
        return validate_canonical_tenant_ref(value)

    @field_validator("group_ref")
    @classmethod
    def validate_group_ref(cls, value: str) -> str:
        return validate_prefixed_opaque_ref(
            value, prefix="group:", field_name="group_ref"
        )

    @field_validator("principal_ref")
    @classmethod
    def validate_principal_ref(cls, value: str) -> str:
        return validate_prefixed_opaque_ref(
            value, prefix="principal:", field_name="principal_ref"
        )


class DirectConversationIdentityV1(StrictModel):
    contract_version: Literal["conversation-identity.v1"]
    surface: Literal["wecom"]
    scene: Literal["direct"]
    tenant_ref: str
    direct_thread_ref: str
    principal_ref: str

    @field_validator("tenant_ref")
    @classmethod
    def validate_tenant_ref(cls, value: str) -> str:
        return validate_canonical_tenant_ref(value)

    @field_validator("direct_thread_ref")
    @classmethod
    def validate_direct_thread_ref(cls, value: str) -> str:
        return validate_prefixed_opaque_ref(
            value, prefix="direct:", field_name="direct_thread_ref"
        )

    @field_validator("principal_ref")
    @classmethod
    def validate_principal_ref(cls, value: str) -> str:
        return validate_prefixed_opaque_ref(
            value, prefix="principal:", field_name="principal_ref"
        )


ConversationIdentityV1 = Annotated[
    GroupConversationIdentityV1 | DirectConversationIdentityV1,
    Field(discriminator="scene"),
]


class GroupPresentationV1(StrictModel):
    contract_version: Literal["group-presentation.v1"]
    conversation_name: str = Field(min_length=1, max_length=120)
    principal_name: str = Field(min_length=1, max_length=80)


class DirectPresentationV1(StrictModel):
    contract_version: Literal["direct-presentation.v1"]
    principal_name: str | None = Field(default=None, min_length=1, max_length=80)


PresentationV1 = Annotated[
    GroupPresentationV1 | DirectPresentationV1,
    Field(discriminator="contract_version"),
]


class DistributionScopeV1(StrictModel):
    kind: Literal["distribution"]
    dist_channel_name: str = Field(min_length=1, max_length=120)
    channel_type: ChannelType
    available_artifacts: list[AvailableArtifactV2] = Field(max_length=3)

    @field_validator("available_artifacts")
    @classmethod
    def validate_available_artifacts_unique(
        cls, values: list[AvailableArtifactV2]
    ) -> list[AvailableArtifactV2]:
        seen: set[AvailableArtifactType] = set()
        for artifact in values:
            if artifact.type in seen:
                raise ValueError(
                    "available_artifacts must not contain duplicate artifact types"
                )
            seen.add(artifact.type)
        return values


class UnscopedScopeV1(StrictModel):
    kind: Literal["unscoped"]


BusinessScopeV1 = Annotated[
    DistributionScopeV1 | UnscopedScopeV1,
    Field(discriminator="kind"),
]


class ExplicitPrincipalGrantsV1(StrictModel):
    contract_version: Literal["principal-grants.v1"]
    read_capabilities: list[ReadCapability] = Field(max_length=5)
    outbound_actions: list[OutboundActionType] = Field(max_length=3)
    mention_types: list[ReplyMentionType] = Field(max_length=1)

    @field_validator("read_capabilities")
    @classmethod
    def canonicalize_read_capabilities(
        cls, values: list[ReadCapability]
    ) -> list[ReadCapability]:
        return _canonical_unique_list(values, "read_capabilities")

    @field_validator("outbound_actions")
    @classmethod
    def canonicalize_outbound_actions(
        cls, values: list[OutboundActionType]
    ) -> list[OutboundActionType]:
        return _canonical_unique_list(values, "outbound_actions")

    @field_validator("mention_types")
    @classmethod
    def canonicalize_mentions(
        cls, values: list[ReplyMentionType]
    ) -> list[ReplyMentionType]:
        return _canonical_unique_list(values, "mention_types")


def _canonical_unique_list(
    values: Sequence[TCanonicalString], field_name: str
) -> list[TCanonicalString]:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    return sorted(values)


class ReplyRequestV2(StrictModel):
    contract_version: Literal["reply-request.v2"]
    request_id: str
    message: str = Field(min_length=1, max_length=20_000)
    context_id: str | None = None
    identity: ConversationIdentityV1
    presentation: PresentationV1
    business_scope: BusinessScopeV1
    grants: ExplicitPrincipalGrantsV1

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        return validate_prefixed_opaque_ref(
            value, prefix="req:", field_name="request_id"
        )

    @field_validator("context_id")
    @classmethod
    def validate_context_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_prefixed_opaque_ref(
            value, prefix="ctx:", field_name="context_id"
        )

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be blank")
        return trimmed

    @model_validator(mode="after")
    def validate_scene_consistency(self):
        if self.identity.scene == "group":
            if self.presentation.contract_version != "group-presentation.v1":
                raise ValueError("group identity requires group presentation")
            if self.business_scope.kind != "distribution":
                raise ValueError("group identity requires distribution scope")
        if self.identity.scene == "direct":
            if self.presentation.contract_version != "direct-presentation.v1":
                raise ValueError("direct identity requires direct presentation")
            if self.business_scope.kind != "unscoped":
                raise ValueError("direct identity requires unscoped scope")
            if not set(self.grants.read_capabilities) <= {
                "query_internal_company_info"
            }:
                raise ValueError("direct identity has forbidden read capabilities")
            if self.grants.outbound_actions or self.grants.mention_types:
                raise ValueError(
                    "direct identity cannot grant outbound actions or mentions"
                )
        return self
