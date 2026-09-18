from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from market_support_crewai_agent.runtime.hashing import sha256_frame
from market_support_crewai_agent.schemas.conversation import (
    BusinessScopeV1,
    DistributionScopeV1,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BusinessScopeAuthorityV1(_FrozenModel):
    contract_version: Literal["business-scope-authority.v1"] = (
        "business-scope-authority.v1"
    )
    scope: BusinessScopeV1
    business_scope_hash: str = Field(pattern=r"^bsh1:[0-9a-f]{64}$")
    business_scope_ref: str = Field(pattern=r"^bsr:[0-9a-f]{32}$")


def business_scope_hash_v1(scope: BusinessScopeV1) -> str:
    return sha256_frame(
        "business-scope.v1",
        scope.model_dump(mode="json", exclude_none=False, exclude_defaults=False),
        prefix="bsh1",
    )


def business_scope_ref_from_hash(business_scope_hash: str) -> str:
    prefix, separator, digest = business_scope_hash.partition(":")
    if prefix != "bsh1" or separator != ":" or len(digest) != 64:
        raise ValueError("invalid_business_scope_hash")
    if any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("invalid_business_scope_hash")
    return f"bsr:{digest[:32]}"


def business_scope_authority_v1(scope: BusinessScopeV1) -> BusinessScopeAuthorityV1:
    business_scope_hash = business_scope_hash_v1(scope)
    return BusinessScopeAuthorityV1(
        scope=scope,
        business_scope_hash=business_scope_hash,
        business_scope_ref=business_scope_ref_from_hash(business_scope_hash),
    )


def distribution_scope_authority_v1(
    scope: DistributionScopeV1,
) -> BusinessScopeAuthorityV1:
    return business_scope_authority_v1(scope)
