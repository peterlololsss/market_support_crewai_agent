from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from pydantic import Field, JsonValue

from market_support_crewai_agent.runtime.integrations.document_mcp.parsing import (
    json_list_from_value,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.sanitizer import (
    safe_document_label as _safe_document_label,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.sanitizer import (
    safe_product_identifier as _safe_product_identifier,
)
from market_support_crewai_agent.schemas.base import StrictModel

ProductSource = Mapping[str, JsonValue]


class DocumentProductCandidate(StrictModel):
    id: str = Field(min_length=1, max_length=160)
    name: str = Field(default="", max_length=160)
    title: str = Field(default="", max_length=240)
    category: str = Field(default="", max_length=80)
    keywords: tuple[str, ...] = Field(default_factory=tuple, max_length=20)
    summary: str = Field(default="", max_length=1200)


class DocumentProductSelectionView(Protocol):
    document_ids: tuple[str, ...]
    confidence: Literal["none", "low", "medium", "high"]


def product_manifest(
    products: Sequence[ProductSource],
) -> tuple[DocumentProductCandidate, ...]:
    candidates: list[DocumentProductCandidate] = []
    for product in products:
        product_id = str(product.get("id") or "").strip()
        if not product_id:
            continue
        keywords = tuple(
            keyword
            for keyword in (
                _safe_document_label(str(item or "").strip())
                for item in json_list_from_value(product.get("keywords"))
            )
            if keyword
        )
        candidates.append(
            DocumentProductCandidate(
                id=_safe_product_identifier(product_id),
                name=_safe_document_label(str(product.get("name") or "").strip()),
                title=_safe_document_label(str(product.get("title") or "").strip()),
                category=_safe_document_label(
                    str(product.get("category") or "").strip()
                ),
                keywords=keywords,
                summary=_safe_document_label(str(product.get("summary") or "").strip()),
            )
        )
    return tuple(candidates)


def validated_selected_document_ids(
    selection: DocumentProductSelectionView,
    products: Sequence[ProductSource],
    *,
    max_documents: int,
) -> list[str]:
    if selection.confidence == "none":
        return []
    safe_to_raw_ids = {
        _safe_product_identifier(product_id): product_id
        for product_id in (str(product.get("id") or "").strip() for product in products)
        if product_id
    }
    valid_ids = {
        product_id
        for product in products
        if (product_id := str(product.get("id") or "").strip())
    }
    output: list[str] = []
    seen: set[str] = set()
    for document_id in selection.document_ids:
        selected_id = str(document_id or "").strip()
        normalized_id = safe_to_raw_ids.get(selected_id, selected_id)
        if not normalized_id or normalized_id in seen or normalized_id not in valid_ids:
            continue
        seen.add(normalized_id)
        output.append(normalized_id)
        if len(output) >= max_documents:
            break
    return output


def fallback_document_ids(
    products: Sequence[ProductSource],
    baseline_categories: tuple[str, ...],
    *,
    max_documents: int,
) -> list[str]:
    wanted = {category.strip() for category in baseline_categories if category.strip()}
    output: list[str] = []
    for product in products:
        product_id = str(product.get("id") or "").strip()
        category = str(product.get("category") or "").strip()
        if not product_id or category not in wanted or product_id in output:
            continue
        output.append(product_id)
        if len(output) >= max_documents:
            return output
    for product in products:
        product_id = str(product.get("id") or "").strip()
        if not product_id or product_id in output:
            continue
        output.append(product_id)
        if len(output) >= max_documents:
            return output
    return output


def selected_with_fallback_ids(
    document_ids: list[str],
    fallback_ids: list[str],
    *,
    max_documents: int,
) -> list[str]:
    if not document_ids:
        return fallback_ids
    output = list(document_ids)
    seen = set(output)
    for fallback_id in fallback_ids:
        if fallback_id in seen:
            continue
        output.append(fallback_id)
        seen.add(fallback_id)
        if len(output) >= max_documents:
            break
    return output
