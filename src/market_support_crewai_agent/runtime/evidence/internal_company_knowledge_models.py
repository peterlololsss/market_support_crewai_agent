from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import ConfigDict, Field

from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.schemas.base import StrictModel


class _FrozenGatewayModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GatewayDocumentContextV1(_FrozenGatewayModel):
    document_id: str = Field(min_length=1, max_length=160)
    title: str = Field(default="document", max_length=160)
    text: str = Field(min_length=1, max_length=1_000_000)
    client_contract_version: str = Field(
        default="document-mcp-client.v1",
        min_length=1,
        max_length=120,
    )
    corpus_version: str = Field(default="unknown", min_length=1, max_length=120)


class GatewayStaticContextV1(_FrozenGatewayModel):
    entry_id: str = Field(min_length=1, max_length=160)
    manifest_ref: ManifestRefV1
    text: str = Field(min_length=1, max_length=1_000_000)
    selected_asset_ids: tuple[str, ...] = Field(default=(), max_length=8)


class RegisteredMediaBindingV1(_FrozenGatewayModel):
    evidence_id: str = Field(pattern=r"^eid1:[0-9a-f]{64}$")
    asset_id: str = Field(min_length=1, max_length=160)
    marker: str = Field(pattern=r"^%%[^%\r\n]{1,160}%%$")


@dataclass(frozen=True, slots=True)
class InternalCompanyKnowledgeResultV1:
    facts: tuple[CanonicalEvidenceFactV1, ...]
    media_bindings: tuple[RegisteredMediaBindingV1, ...]


class PostPlanDocumentKnowledgeProvider(Protocol):
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[GatewayDocumentContextV1, ...]: ...


class PostPlanStaticKnowledgeProvider(Protocol):
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
    ) -> tuple[GatewayStaticContextV1, ...]: ...
