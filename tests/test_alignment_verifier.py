from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)


def test_aligned_verdict_shape_requires_safe_none_fields():
    verdict = ReplyAlignmentVerdict(
        aligned=True,
        safe_to_return=True,
        failure_code="none",
        remediation="none",
    )

    assert verdict.contract_version == "reply-alignment-verdict"
    assert verdict.aligned is True


def test_aligned_verdict_rejects_non_none_failure_code():
    with pytest.raises(ValueError):
        ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            failure_code="wrong_action",
            remediation="none",
        )


def test_aligned_verdict_requires_safe_to_return():
    with pytest.raises(ValueError):
        ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=False,
            failure_code="none",
            remediation="none",
        )


def test_refetch_document_context_requires_refined_query():
    with pytest.raises(ValueError):
        ReplyAlignmentVerdict(
            aligned=False,
            safe_to_return=False,
            failure_code="missing_evidence",
            remediation="refetch_document_context",
        )
