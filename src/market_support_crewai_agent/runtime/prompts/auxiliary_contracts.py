from __future__ import annotations

from typing import Annotated, ClassVar, Final, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from market_support_crewai_agent.runtime.hashing import canonical_json_bytes
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.schemas.base import StrictModel

_APPROVED_INPUT_MAX_BYTES: Final = 128 * 1024
_DOCUMENT_INPUT_MAX_BYTES: Final = 256 * 1024

SelectorConfidence = Literal["none", "low", "medium", "high"]
CanonicalSelectorId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$",
    ),
]
ProductCandidateId = Annotated[str, Field(min_length=1, max_length=160)]
Keyword = Annotated[str, Field(max_length=160)]


class AuxiliaryContractError(ValueError):
    pass


class _FrozenAuxiliaryModel(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


def _require_unique(values: tuple[str, ...]) -> None:
    if len(set(values)) != len(values):
        raise AuxiliaryContractError("duplicate_selector_identifier")


class ApprovedImageAssetCandidateViewV1(_FrozenAuxiliaryModel):
    asset_id: CanonicalSelectorId
    semantic_label: str = Field(min_length=1, max_length=120)


class ApprovedKnowledgeCandidateViewV1(_FrozenAuxiliaryModel):
    entry_id: CanonicalSelectorId
    question: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=120)
    semantic_purpose: str = Field(min_length=1, max_length=120)
    manifest_ref: ManifestRefV1
    fact_type: Literal["document_context"]
    image_assets: tuple[ApprovedImageAssetCandidateViewV1, ...] = Field(
        default=(),
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_image_asset_ids(self) -> ApprovedKnowledgeCandidateViewV1:
        _require_unique(tuple(asset.asset_id for asset in self.image_assets))
        return self


class ApprovedKnowledgeSelectorInputV1(_FrozenAuxiliaryModel):
    contract_version: Literal["approved-knowledge-selector-input.v1"] = (
        "approved-knowledge-selector-input.v1"
    )
    user_query: str = Field(min_length=1)
    evidence_query: str = Field(max_length=200)
    candidates: tuple[ApprovedKnowledgeCandidateViewV1, ...] = Field(
        min_length=1,
        max_length=20,
    )
    max_entries: int = Field(ge=1, le=5)
    max_images: int = Field(ge=0, le=8)
    selected_manifest_ref: ManifestRefV1 | None

    @model_validator(mode="after")
    def validate_selector_scope(self) -> ApprovedKnowledgeSelectorInputV1:
        _require_unique(tuple(candidate.entry_id for candidate in self.candidates))
        if self.selected_manifest_ref is not None:
            if self.max_images != 0:
                raise AuxiliaryContractError("direct_selector_images_forbidden")
            if any(
                candidate.manifest_ref != self.selected_manifest_ref
                for candidate in self.candidates
            ):
                raise AuxiliaryContractError("selector_manifest_scope_mismatch")
        if (
            len(canonical_json_bytes(self.model_dump(mode="json", exclude_none=False)))
            > _APPROVED_INPUT_MAX_BYTES
        ):
            raise AuxiliaryContractError("approved_selector_input_too_large")
        return self


class ApprovedKnowledgeSelection(_FrozenAuxiliaryModel):
    selected_entry_ids: tuple[CanonicalSelectorId, ...] = Field(
        default=(),
        max_length=5,
    )
    selected_image_asset_ids: tuple[CanonicalSelectorId, ...] = Field(
        default=(),
        max_length=8,
    )
    confidence: SelectorConfidence = "none"
    rationale: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def validate_selected_ids(self) -> ApprovedKnowledgeSelection:
        _require_unique(self.selected_entry_ids)
        _require_unique(self.selected_image_asset_ids)
        return self


class ProductCandidateViewV1(_FrozenAuxiliaryModel):
    id: ProductCandidateId
    name: str = Field(default="", max_length=160)
    title: str = Field(default="", max_length=240)
    category: str = Field(default="", max_length=80)
    keywords: tuple[Keyword, ...] = Field(default=(), max_length=20)
    summary: str = Field(default="", max_length=1200)


class DocumentProductSelectorInputV1(_FrozenAuxiliaryModel):
    contract_version: Literal["document-product-selector-input.v1"] = (
        "document-product-selector-input.v1"
    )
    user_query: str = Field(min_length=1)
    evidence_query: str = Field(max_length=200)
    candidates: tuple[ProductCandidateViewV1, ...] = Field(
        min_length=1,
        max_length=100,
    )
    max_documents: int = Field(ge=1, le=50)

    @model_validator(mode="after")
    def validate_selector_scope(self) -> DocumentProductSelectorInputV1:
        _require_unique(tuple(candidate.id for candidate in self.candidates))
        if (
            len(canonical_json_bytes(self.model_dump(mode="json", exclude_none=False)))
            > _DOCUMENT_INPUT_MAX_BYTES
        ):
            raise AuxiliaryContractError("document_selector_input_too_large")
        return self


class DocumentProductSelection(_FrozenAuxiliaryModel):
    document_ids: tuple[ProductCandidateId, ...] = Field(
        default=(),
        max_length=50,
    )
    confidence: SelectorConfidence = "none"
    rationale: str = Field(default="", max_length=600)

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(
        cls,
        document_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        _require_unique(document_ids)
        return document_ids


class LlmHealthProbeInputV1(_FrozenAuxiliaryModel):
    contract_version: Literal["llm-health-probe-input.v1"]


class LlmHealthProbeOutputV1(_FrozenAuxiliaryModel):
    contract_version: Literal["llm-health-probe-output.v1"]
    ok: Literal[True]
