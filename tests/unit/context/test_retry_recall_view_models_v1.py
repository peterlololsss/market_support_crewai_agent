from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

import market_support_crewai_agent.runtime.context.models as context_models
from market_support_crewai_agent.runtime.recall.outcome_models import (
    RecallBranchOutcomeV1,
    RecallCandidateViewV1,
)


def _issue() -> context_models.PlanValidationIssueViewV1:
    return context_models.PlanValidationIssueViewV1(
        code="plan_spec_schema_invalid",
        severity="fatal",
        message="schema mismatch",
    )


@pytest.mark.parametrize(
    ("attempt", "phase", "phase_attempt"),
    [
        (1, "schema_repair", 1),
        (2, "alignment_replan", 1),
        (3, "alignment_replan", 2),
    ],
)
def test_planner_retry_overlay_accepts_only_the_closed_attempt_mapping(
    attempt: int,
    phase: Literal["schema_repair", "alignment_replan"],
    phase_attempt: int,
) -> None:
    # Given: one row from the closed planner retry mapping.
    # When: the overlay is constructed.
    overlay = context_models.PlannerRetryOverlayV1(
        attempt=attempt,
        phase=phase,
        phase_attempt=phase_attempt,
        issues=(_issue(),),
        feedback=None,
    )

    # Then: the overall and phase attempts remain distinct and exact.
    assert (overlay.attempt, overlay.phase, overlay.phase_attempt) == (
        attempt,
        phase,
        phase_attempt,
    )


@pytest.mark.parametrize(
    ("attempt", "phase", "phase_attempt"),
    [
        (1, "alignment_replan", 1),
        (2, "schema_repair", 1),
        (2, "alignment_replan", 2),
        (3, "alignment_replan", 1),
    ],
)
def test_planner_retry_overlay_rejects_every_other_attempt_mapping(
    attempt: int,
    phase: Literal["schema_repair", "alignment_replan"],
    phase_attempt: int,
) -> None:
    # Given: a tuple outside the three permitted planner retry rows.
    # When/Then: strict cross-field validation rejects it.
    with pytest.raises(ValidationError):
        context_models.PlannerRetryOverlayV1(
            attempt=attempt,
            phase=phase,
            phase_attempt=phase_attempt,
            issues=(_issue(),),
            feedback=None,
        )


def test_plan_validation_issue_has_a_closed_sanitized_schema() -> None:
    # Given: the model-visible retry issue type.
    issue = _issue()

    # When: its exact field set is inspected.
    fields = set(type(issue).model_fields)

    # Then: metadata and arbitrary codes are not representable.
    assert fields == {"code", "severity", "message"}
    with pytest.raises(ValidationError):
        context_models.PlanValidationIssueViewV1.model_validate(
            {
                "code": "arbitrary_failure",
                "severity": "error",
                "message": "bad",
            }
        )
    with pytest.raises(ValidationError):
        context_models.PlanValidationIssueViewV1.model_validate(
            {
                "code": "plan_spec_schema_invalid",
                "severity": "error",
                "message": "bad",
                "metadata": {"raw": "secret"},
            }
        )
    with pytest.raises(ValidationError):
        context_models.PlanValidationIssueViewV1(
            code="plan_spec_schema_invalid",
            severity="error",
            message="bad\x00message",
        )


def test_composer_retry_overlay_enforces_attempt_and_feedback_bounds() -> None:
    # Given: the maximum valid composer retry attempt and feedback length.
    overlay = context_models.ComposerRetryOverlayV1(
        attempt=2,
        feedback="x" * 300,
    )

    # When: its values are inspected.
    # Then: both declared maxima round-trip exactly.
    assert overlay.attempt == 2
    assert len(overlay.feedback) == 300

    # Given: values outside the closed composer retry contract.
    # When/Then: attempt zero/three and empty/oversize feedback are rejected.
    for attempt, feedback in ((0, "x"), (3, "x"), (1, ""), (1, "x" * 301)):
        with pytest.raises(ValidationError):
            context_models.ComposerRetryOverlayV1(
                attempt=attempt,
                feedback=feedback,
            )


def _branches() -> tuple[RecallBranchOutcomeV1, RecallBranchOutcomeV1]:
    return (
        RecallBranchOutcomeV1(
            source_class="approved_static",
            status="ok",
            accepted_count=1,
            rejected_count=0,
            reason_code="ok",
        ),
        RecallBranchOutcomeV1(
            source_class="document_mcp",
            status="not_called",
            accepted_count=0,
            rejected_count=0,
            reason_code="not_needed",
        ),
    )


def _candidate() -> RecallCandidateViewV1:
    return RecallCandidateViewV1(
        candidate_id="candidate:1",
        canonical_id="canonical:company-contact",
        source_class="approved_static",
        question="公司网址是什么？",
        confidence=0.91,
        reason_code="approved_recall_hit",
    )


def test_recall_planner_view_preserves_safe_shortcut_identity_and_hashes() -> None:
    # Given: a shortcut with a selected manifest but no source locator or body.
    view = context_models.RecallPlannerViewV1.model_validate(
        {
            "mode": "shortcut",
            "decision": "shortcut_match",
            "candidates": (_candidate(),),
            "branch_outcomes": _branches(),
            "shortcut_summary": {
                "candidate_id": "candidate:1",
                "canonical_id": "canonical:company-contact",
                "selected_manifest_ref": {
                    "manifest_id": "answer_internal_company_knowledge",
                    "manifest_version": "2026-07-18.1",
                },
                "confidence": 0.91,
                "reply_text_hash": "rtx1:" + "a" * 64,
                "evidence_text_hash": "evx1:" + "b" * 64,
            },
            "trace_hash": "rch1:" + "c" * 64,
        }
    )

    # When: both planner and shortcut-summary schemas are inspected.
    summary = view.shortcut_summary
    assert summary is not None
    fields = set(type(view).model_fields)
    summary_fields = set(type(summary).model_fields)

    # Then: candidates, selected ref, and hashes survive while every locator/body is absent.
    assert fields == {
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
        "selected_manifest_ref",
        "confidence",
        "reply_text_hash",
        "evidence_text_hash",
    }
    assert not summary_fields & {
        "source_id",
        "source_ref",
        "reply_text",
        "evidence_text",
        "selected_media_asset_ids",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("answer", "secret answer"),
        ("evidence_text", "secret evidence"),
        ("locator", "file:///private/secret"),
        ("shortcut_payload", {"reply_text": "secret"}),
    ],
)
def test_recall_planner_view_rejects_body_and_locator_canaries(
    field: str,
    value: str | dict[str, str],
) -> None:
    # Given: an otherwise valid advisory planner view plus one forbidden canary.
    payload = {
        "mode": "advisory",
        "decision": "advisory_candidates",
        "candidates": (_candidate(),),
        "branch_outcomes": (
            _branches()[0],
            RecallBranchOutcomeV1(
                source_class="document_mcp",
                status="ok",
                accepted_count=0,
                rejected_count=0,
                reason_code="no_match",
            ),
        ),
        "shortcut_summary": None,
        "trace_hash": "rch1:" + "c" * 64,
        field: value,
    }

    # When/Then: the strict planner projection rejects the leak.
    with pytest.raises(ValidationError):
        context_models.RecallPlannerViewV1.model_validate(payload)


def test_recall_planner_view_rejects_source_locator_and_transition_mismatch() -> None:
    # Given: shortcut data containing an internal source locator.
    shortcut = {
        "candidate_id": "candidate:1",
        "canonical_id": "canonical:company-contact",
        "selected_manifest_ref": {
            "manifest_id": "answer_internal_company_knowledge",
            "manifest_version": "2026-07-18.1",
        },
        "confidence": 0.91,
        "reply_text_hash": "rtx1:" + "a" * 64,
        "evidence_text_hash": "evx1:" + "b" * 64,
        "source_id": "qa-entry:private-locator",
    }

    # When/Then: nested extra rejection prevents the locator leak.
    with pytest.raises(ValidationError):
        context_models.RecallPlannerViewV1.model_validate(
            {
                "mode": "shortcut",
                "decision": "shortcut_match",
                "candidates": (_candidate(),),
                "branch_outcomes": _branches(),
                "shortcut_summary": shortcut,
                "trace_hash": "rch1:" + "c" * 64,
            }
        )

    # Given: a non-shortcut decision carrying shortcut identity.
    shortcut.pop("source_id")

    # When/Then: the decision/summary mismatch is rejected.
    with pytest.raises(ValidationError):
        context_models.RecallPlannerViewV1.model_validate(
            {
                "mode": "advisory",
                "decision": "no_match",
                "candidates": (),
                "branch_outcomes": _branches(),
                "shortcut_summary": shortcut,
                "trace_hash": "rch1:" + "c" * 64,
            }
        )
