from __future__ import annotations

import hashlib
import importlib

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.hashing import canonical_json_bytes
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1

AUTHORITATIVE_MODULE = "market_support_crewai_agent.runtime.recall.outcome_models"
TURN_STATE_MODULE = "market_support_crewai_agent.runtime.recall.turn_state"


def _models():
    return importlib.import_module(AUTHORITATIVE_MODULE)


def _turn_state_models():
    return importlib.import_module(TURN_STATE_MODULE)


def _candidate(question: str, *, candidate_id: str = "candidate:1"):
    return _models().RecallCandidateViewV1(
        candidate_id=candidate_id,
        canonical_id="canonical:company-contact",
        source_class="approved_static",
        question=question,
        confidence=0.91,
        reason_code="approved_recall_hit",
    )


def _disabled_outcome():
    models = _models()
    branches = (
        models.RecallBranchOutcomeV1(
            source_class="approved_static",
            status="not_called",
            accepted_count=0,
            rejected_count=0,
            reason_code="disabled",
        ),
        models.RecallBranchOutcomeV1(
            source_class="document_mcp",
            status="not_called",
            accepted_count=0,
            rejected_count=0,
            reason_code="disabled",
        ),
    )
    return models.RecallOutcomeV1(
        mode="off",
        decision="disabled",
        candidates=(),
        branch_outcomes=branches,
        shortcut_summary=None,
        trace_hash="rch1:" + "0" * 64,
    )


@pytest.mark.parametrize("character", ["a", "问", "\U0001f642"])
def test_candidate_round_trips_400_unicode_code_points(character: str) -> None:
    # Given: an accepted question at the exact Unicode code-point ceiling.
    question = character * 400

    # When: the candidate crosses its JSON boundary and returns.
    candidate = _candidate(question)
    serialized = candidate.model_dump_json().encode("utf-8")
    restored = _models().RecallCandidateViewV1.model_validate_json(serialized)

    # Then: no clipping occurs and the standalone row stays below 4,096 bytes.
    assert restored.question == question
    assert len(restored.question) == 400
    assert len(serialized) <= 4_096


@pytest.mark.parametrize("character", ["a", "问", "\U0001f642"])
def test_candidate_rejects_401_unicode_code_points(character: str) -> None:
    # Given: a candidate question one code point over the bound.
    question = character * 401

    # When/Then: Pydantic rejects it instead of clipping it.
    with pytest.raises(ValidationError):
        _candidate(question)


def test_fully_max_populated_candidate_fits_declared_byte_ceiling() -> None:
    # Given: every candidate string field at its maximum declared size.
    models = _models()
    candidate = models.RecallCandidateViewV1(
        candidate_id="c" * 64,
        canonical_id="k" * 128,
        source_class="approved_static",
        question="\U0001f642" * 400,
        confidence=0.123456789,
        reason_code="approved_recall_hint",
    )

    # When: its canonical JSON bytes are materialized.
    payload = candidate.model_dump(mode="json", exclude_none=False)
    encoded = canonical_json_bytes(payload)

    # Then: the complete valid row fits without clipping or escaping expansion.
    assert len(encoded) <= 4_096
    assert models.RecallCandidateViewV1.model_validate_json(encoded) == candidate


def test_candidate_and_outcome_fields_are_provider_neutral_and_body_free() -> None:
    # Given: the two model-visible recall boundary schemas.
    models = _models()

    # When: their exact declared field sets are inspected.
    candidate_fields = set(models.RecallCandidateViewV1.model_fields)
    outcome_fields = set(models.RecallOutcomeV1.model_fields)
    summary_fields = set(models.RecallShortcutSummaryV1.model_fields)

    # Then: no provider-selection flag, provider body, locator, or generic bag exists.
    assert candidate_fields == {
        "contract_version",
        "candidate_id",
        "canonical_id",
        "source_class",
        "question",
        "confidence",
        "reason_code",
    }
    assert outcome_fields == {
        "contract_version",
        "mode",
        "decision",
        "candidates",
        "branch_outcomes",
        "shortcut_summary",
        "trace_hash",
    }
    assert summary_fields == {
        "candidate_id",
        "canonical_id",
        "source_id",
        "selected_manifest_ref",
        "confidence",
        "reply_text_hash",
        "evidence_text_hash",
    }


@pytest.mark.parametrize(
    ("extra_field", "value"),
    [
        ("selected_provider", "document_mcp"),
        ("provider_enabled", True),
        ("answer", "provider body"),
        ("document_body", "provider body"),
        ("locator", "file:///private/secret"),
    ],
)
def test_candidate_rejects_provider_selection_and_body_fields(
    extra_field: str,
    value: str | bool,
) -> None:
    # Given: an otherwise valid model-visible candidate with a forbidden field.
    payload = {
        "candidate_id": "candidate:1",
        "canonical_id": "canonical:company-contact",
        "source_class": "approved_static",
        "question": "公司网址是什么？",
        "confidence": 0.91,
        "reason_code": "approved_recall_hit",
        extra_field: value,
    }

    # When/Then: the closed model rejects provider control and body leakage.
    with pytest.raises(ValidationError):
        _models().RecallCandidateViewV1.model_validate(payload)


def test_candidate_list_rejects_serialized_payload_over_5120_bytes() -> None:
    # Given: individually valid candidates whose combined canonical list is oversized.
    models = _models()
    candidates = tuple(
        _candidate("\U0001f642" * 400, candidate_id=f"candidate:{index}")
        for index in range(3)
    )
    branches = (
        models.RecallBranchOutcomeV1(
            source_class="approved_static",
            status="ok",
            accepted_count=3,
            rejected_count=0,
            reason_code="ok",
        ),
        models.RecallBranchOutcomeV1(
            source_class="document_mcp",
            status="ok",
            accepted_count=0,
            rejected_count=0,
            reason_code="no_match",
        ),
    )

    # When/Then: the outcome rejects the aggregate instead of clipping candidates.
    with pytest.raises(ValidationError):
        models.RecallOutcomeV1(
            mode="advisory",
            decision="advisory_candidates",
            candidates=candidates,
            branch_outcomes=branches,
            shortcut_summary=None,
            trace_hash="rch1:" + "0" * 64,
        )


def test_transient_text_hashes_use_exact_domain_framing() -> None:
    # Given: exact validated Unicode shortcut bodies.
    models = _turn_state_models()
    reply_text = "公司网址是 https://example.com。"
    evidence_text = "Q：公司网址是什么？\nA：公司网址是 https://example.com。"

    # When: each transient content hash is derived.
    reply_hash = models.recall_reply_text_hash(reply_text)
    evidence_hash = models.recall_evidence_text_hash(evidence_text)

    # Then: hashes use their distinct ASCII-domain + NUL + exact UTF-8 frames.
    assert (
        reply_hash
        == "rtx1:"
        + hashlib.sha256(
            b"recall-reply-text.v1\0" + reply_text.encode("utf-8")
        ).hexdigest()
    )
    assert (
        evidence_hash
        == "evx1:"
        + hashlib.sha256(
            b"recall-evidence-text.v1\0" + evidence_text.encode("utf-8")
        ).hexdigest()
    )


def test_shortcut_payload_rejects_content_hash_mismatch() -> None:
    # Given: a shortcut payload whose reply hash belongs to different content.
    models = _turn_state_models()
    evidence_text = "Q：公司网址是什么？\nA：公司网址是 https://example.com。"

    # When/Then: internal construction rejects the mismatched transient hash.
    with pytest.raises(ValidationError):
        models.ApprovedStaticShortcutPayloadV1(
            candidate_id="candidate:1",
            canonical_id="canonical:company-contact",
            selected_manifest_ref=ManifestRefV1(
                manifest_id="answer_internal_company_knowledge",
                manifest_version="2026-07-18.1",
            ),
            fact_type="document_context",
            selected_media_asset_ids=(),
            confidence=0.91,
            question="公司网址是什么？",
            reply_text="公司网址是 https://example.com。",
            evidence_text=evidence_text,
            source_id="company_contact",
            reply_text_hash=models.recall_reply_text_hash("different"),
            evidence_text_hash=models.recall_evidence_text_hash(evidence_text),
        )


def test_recall_outcome_hash_excludes_only_trace_hash() -> None:
    # Given: one valid model-safe outcome with two different trace placeholders.
    models = _turn_state_models()
    outcome = _disabled_outcome()
    alternate_trace = outcome.model_copy(update={"trace_hash": "rch1:" + "f" * 64})
    payload = outcome.model_dump(mode="json", exclude_none=False)
    del payload["trace_hash"]

    # When: both outcomes are hashed.
    digest = models.recall_outcome_hash(outcome)
    alternate_digest = models.recall_outcome_hash(alternate_trace)

    # Then: only the complete non-trace canonical outcome enters the rch1 frame.
    assert (
        digest
        == "rch1:"
        + hashlib.sha256(
            b"recall-outcome.v1\0" + canonical_json_bytes(payload)
        ).hexdigest()
    )
    assert alternate_digest == digest


def test_turn_state_is_frozen_non_serializable_wrapper() -> None:
    # Given: a disabled model-safe outcome and no internal shortcut payload.
    models = _turn_state_models()
    outcome = _disabled_outcome()

    # When: the per-turn wrapper is constructed.
    turn_state = models.RecallTurnStateV1(
        outcome=outcome,
        shortcut_payload=None,
    )

    # Then: callers can project only outcome and cannot generically dump the wrapper.
    assert turn_state.outcome is outcome
    assert turn_state.shortcut_payload is None
    assert not hasattr(turn_state, "model_dump")
