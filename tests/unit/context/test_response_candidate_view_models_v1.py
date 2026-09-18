from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_support_crewai_agent.runtime.context import models as context_models


def test_response_directive_is_frozen_and_contains_no_authority_refs() -> None:
    # Given: a deterministic directive carrying output shape but no authority.
    directive = context_models.ResponseDirectiveViewV1(
        mode="clarification",
        reply_kind="clarification",
        reason_code="ambiguous_request",
        requires_composer=False,
        composer_stage=None,
        deterministic_text="Please clarify.",
        mentions_requested=False,
        action_intent_count=0,
    )

    # When: its exact serialized DTO is inspected.
    payload = directive.model_dump(mode="json")

    # Then: the view is bounded, authority-free, and immutable.
    assert set(payload) == {
        "contract_version",
        "mode",
        "reply_kind",
        "reason_code",
        "requires_composer",
        "composer_stage",
        "deterministic_text",
        "mentions_requested",
        "action_intent_count",
    }
    with pytest.raises(ValidationError):
        directive.reason_code = "changed"


def test_response_directive_rejects_raw_resolve_ref() -> None:
    # Given: an otherwise valid directive payload with a private adapter ref.
    payload = {
        "mode": "action",
        "reply_kind": "answer",
        "reason_code": "action_ready",
        "requires_composer": False,
        "mentions_requested": False,
        "action_intent_count": 1,
        "resolve_ref": "private.adapter.ref",
    }

    # When/Then: the strict view rejects the raw authority field.
    with pytest.raises(ValidationError):
        context_models.ResponseDirectiveViewV1.model_validate(payload)


def test_output_ceilings_require_canonical_reply_kinds_and_zero_actions() -> None:
    # Given: deterministic output ceilings for a composer invocation.
    ceilings = context_models.EffectiveOutputCeilingsViewV1(
        allowed_reply_kinds=("answer", "clarification"),
        actions_allowed=False,
        mentions_allowed=False,
        max_actions=0,
        max_mentions=0,
        max_reply_chars=4_000,
    )

    # When/Then: canonical values pass and authority widening fails.
    assert ceilings.max_actions == 0
    with pytest.raises(ValidationError):
        context_models.EffectiveOutputCeilingsViewV1(
            allowed_reply_kinds=("clarification", "answer"),
            actions_allowed=False,
            mentions_allowed=False,
            max_actions=0,
            max_mentions=0,
            max_reply_chars=4_000,
        )
    with pytest.raises(ValidationError):
        context_models.EffectiveOutputCeilingsViewV1(
            allowed_reply_kinds=("answer",),
            actions_allowed=True,
            mentions_allowed=False,
            max_actions=1,
            max_mentions=0,
            max_reply_chars=4_000,
        )


def test_candidate_action_enforces_type_resolve_and_metadata_binding() -> None:
    # Given: one weekly-report candidate action without its private resolve ref.
    action = context_models.CandidateActionViewV1(
        type="send_weekly_report",
        resolve_type="weekly_report",
        resolve_ref_available=True,
        period="2026-W28",
        report_date="2026-07-17",
    )

    # When/Then: safe metadata survives and a cross-type resolve is rejected.
    assert action.resolve_ref_available is True
    with pytest.raises(ValidationError, match="candidate_action_resolve_type_mismatch"):
        context_models.CandidateActionViewV1(
            type="send_weekly_report",
            resolve_type="monthly_report",
            resolve_ref_available=True,
            period="2026-W28",
            report_date="2026-07-17",
        )


def test_candidate_reply_rejects_action_and_identity_refs() -> None:
    # Given: a bounded candidate reply with one safe action summary.
    action = context_models.CandidateActionViewV1(
        type="send_material_pack",
        resolve_type="material_pack",
        resolve_ref_available=True,
        material_pack_option="Option A",
    )
    candidate = context_models.CandidateReplyViewV1(
        reply_kind="answer",
        text="Ready.",
        actions=(action,),
    )

    # When: the candidate is serialized and then polluted with raw identity.
    payload = candidate.model_dump(mode="python")
    payload["response_id"] = "resp-private"

    # Then: safe action availability remains, while raw identity rejects.
    assert candidate.actions[0].resolve_ref_available is True
    with pytest.raises(ValidationError):
        context_models.CandidateReplyViewV1.model_validate(payload)


def test_preflight_fact_uses_boolean_ref_availability_and_strict_iso_date() -> None:
    # Given: a safe preflight summary without adapter result or locator fields.
    fact = context_models.PreflightFactViewV1(
        resolve_type="weekly_report",
        status="resolved",
        display_name="Weekly report",
        reason_code="ok",
        resolve_ref_available=True,
        artifact_type="weekly_report",
        period="2026-W28",
        report_date="2026-07-17",
    )

    # When/Then: the boolean survives and non-ISO dates reject.
    assert fact.resolve_ref_available is True
    with pytest.raises(ValidationError):
        context_models.PreflightFactViewV1(
            resolve_type="weekly_report",
            status="resolved",
            reason_code="ok",
            resolve_ref_available=True,
            artifact_type="weekly_report",
            report_date="17/07/2026",
        )
