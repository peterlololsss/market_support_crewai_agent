from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentQueryEvidenceCommandV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactV1,
)


def _raw_fact(*, scope_artifact_type: str) -> dict[str, object]:
    return {
        "evidence_id": "eid1:" + "1" * 64,
        "fact_type": "document_context",
        "source_type": "document_mcp",
        "artifact_type": "document_context",
        "value": {"kind": "string", "value": "trusted document evidence"},
        "scope": {
            "kind": "unscoped",
            "artifact_type": scope_artifact_type,
        },
        "provenance": {
            "source_class": "document_mcp",
            "source_record_ref": "esr1:" + "2" * 64,
            "producer_contract_version": "document-mcp.v1",
            "retrieval_operation": "document_query",
            "scope_ref": "escope1:" + "3" * 64,
            "public_url_hashes": [],
        },
    }


def test_canonical_evidence_rejects_unscoped_artifact_mismatch() -> None:
    with pytest.raises(
        ValidationError, match="canonical_evidence_artifact_scope_mismatch"
    ):
        CanonicalEvidenceFactV1.model_validate(_raw_fact(scope_artifact_type="unknown"))


def test_canonical_evidence_accepts_unscoped_company_document_context() -> None:
    fact = CanonicalEvidenceFactV1.model_validate(
        _raw_fact(scope_artifact_type="document_context")
    )

    assert fact.artifact_type == fact.scope.artifact_type == "document_context"


def test_document_query_command_accepts_160_character_document_id() -> None:
    command = DocumentQueryEvidenceCommandV1(
        source_version="document-mcp-client.v1",
        query="company facts",
        document_ids=("a" * 160,),
        page=1,
        page_size=1,
        max_results=1,
    )

    assert command.document_ids == ("a" * 160,)


def test_document_query_command_rejects_161_character_document_id() -> None:
    with pytest.raises(ValidationError):
        DocumentQueryEvidenceCommandV1(
            source_version="document-mcp-client.v1",
            query="company facts",
            document_ids=("a" * 161,),
            page=1,
            page_size=1,
            max_results=1,
        )
