from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

import market_support_crewai_agent.runtime.context.models as context_models


def test_material_pack_and_intent_counts_preserve_large_python_integers() -> None:
    # Given: a legacy option count above every 16-bit transport ceiling.
    count = 100_000

    # When: both count-bearing views are constructed.
    options = context_models.MaterialPackOptionSummaryViewV1(
        total_count=count,
        exact_requested_option="稳健型",
        exact_match=True,
        bounded_page_available=True,
    )
    intent = context_models.IntentGateViewV1(
        artifact_hint="material_pack",
        outbound_action_hint=True,
        material_pack_option_count=count,
        compliance_hint="clean",
        confidence=1.0,
    )

    # Then: both JSON projections retain the exact integer without an option list.
    assert options.model_dump(mode="json")["total_count"] == count
    assert intent.model_dump(mode="json")["material_pack_option_count"] == count
    assert "options" not in type(options).model_fields

    # Given: a negative count in either view.
    # When/Then: the non-negative contract rejects it.
    with pytest.raises(ValidationError):
        context_models.MaterialPackOptionSummaryViewV1(
            total_count=-1,
            exact_requested_option=None,
            exact_match=None,
            bounded_page_available=False,
        )
    with pytest.raises(ValidationError):
        context_models.IntentGateViewV1(
            artifact_hint="unclear",
            outbound_action_hint=False,
            material_pack_option_count=-1,
            compliance_hint="unknown",
            confidence=0.0,
        )


def test_scene_presentation_enforces_group_and_individual_allowlists() -> None:
    # Given: valid group and direct presentation summaries.
    group = context_models.ScenePresentationViewV1(
        scene="group",
        audience="group",
        conversation_name="市场支持群",
        principal_name="小王",
        redacted=False,
    )
    direct = context_models.ScenePresentationViewV1(
        scene="direct",
        audience="individual",
        conversation_name=None,
        principal_name="小王",
        redacted=True,
    )

    # When: their scene-safe names are inspected.
    # Then: group may carry both names while direct can never carry a conversation name.
    assert group.conversation_name == "市场支持群"
    assert direct.conversation_name is None

    # Given: a mismatched audience or direct conversation name.
    invalid_payloads = (
        {
            "scene": "group",
            "audience": "individual",
            "conversation_name": None,
        },
        {
            "scene": "direct",
            "audience": "individual",
            "conversation_name": "private-thread-locator",
        },
    )

    # When/Then: both cross-scene leaks are rejected.
    for invalid in invalid_payloads:
        with pytest.raises(ValidationError):
            context_models.ScenePresentationViewV1.model_validate(
                {"principal_name": None, "redacted": True, **invalid}
            )


def test_business_scope_union_exposes_only_safe_canonical_fields() -> None:
    # Given: safe distribution and exact unscoped payloads.
    adapter = TypeAdapter(context_models.BusinessScopeViewV1)
    distribution = adapter.validate_python(
        {
            "kind": "distribution",
            "business_scope_ref": "bsr:" + "a" * 32,
            "channel_type": "bank",
            "dist_channel_name": "示例渠道",
            "artifact_types": ("material_pack", "weekly_report"),
        }
    )
    unscoped = adapter.validate_python({"kind": "unscoped"})

    # When: both discriminated branches are serialized.
    distribution_payload = distribution.model_dump(mode="json")
    unscoped_payload = unscoped.model_dump(mode="json")

    # Then: no delivery identity or option payload is representable.
    assert set(distribution_payload) == {
        "kind",
        "business_scope_ref",
        "channel_type",
        "dist_channel_name",
        "artifact_types",
    }
    assert unscoped_payload == {"kind": "unscoped"}

    # Given: non-canonical artifacts and an identity-like extra.
    invalid_payloads = (
        {
            "kind": "distribution",
            "business_scope_ref": "bsr:" + "a" * 32,
            "channel_type": "bank",
            "dist_channel_name": "示例渠道",
            "artifact_types": ("weekly_report", "material_pack"),
        },
        {"kind": "unscoped", "destination_id": "group:secret"},
    )

    # When/Then: canonical-order and privacy violations are rejected.
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            adapter.validate_python(payload)


def test_effective_policy_is_canonical_and_excludes_authority_hashes() -> None:
    # Given: a direct-scene policy view with no outbound authority.
    policy = context_models.EffectivePolicyViewV1(
        policy_id="pol1:" + "b" * 64,
        scene="direct",
        eligible_capabilities=(),
        read_capabilities=(),
        internal_company_knowledge_enabled=False,
        outbound_actions=(),
        mention_types=(),
        adapter_resolves=(),
        allowed_reply_modes=("smalltalk",),
        recall_mode="off",
        evidence_call_limit=16,
        actions_allowed=False,
        mentions_allowed=False,
    )

    # When: its exact model-visible field set is inspected.
    fields = set(type(policy).model_fields)

    # Then: compiled limits remain while grants and admission hashes are impossible.
    assert fields == {
        "contract_version",
        "policy_id",
        "scene",
        "eligible_capabilities",
        "read_capabilities",
        "internal_company_knowledge_enabled",
        "outbound_actions",
        "mention_types",
        "adapter_resolves",
        "allowed_reply_modes",
        "recall_mode",
        "evidence_call_limit",
        "actions_allowed",
        "mentions_allowed",
    }
    assert not fields & {
        "effective_grants_hash",
        "business_scope_hash",
        "state_admission_hash",
        "material_pack_options",
        "ledger_summary",
    }


def test_effective_policy_rejects_noncanonical_or_direct_outbound_authority() -> None:
    # Given: a common valid direct policy payload.
    payload = {
        "policy_id": "pol1:" + "b" * 64,
        "scene": "direct",
        "eligible_capabilities": (),
        "read_capabilities": (),
        "internal_company_knowledge_enabled": False,
        "outbound_actions": (),
        "mention_types": (),
        "adapter_resolves": (),
        "allowed_reply_modes": ("smalltalk",),
        "recall_mode": "off",
        "evidence_call_limit": 1,
        "actions_allowed": False,
        "mentions_allowed": False,
    }

    # When/Then: unsorted modes and direct outbound authority both fail closed.
    with pytest.raises(ValidationError):
        context_models.EffectivePolicyViewV1.model_validate(
            {**payload, "allowed_reply_modes": ("unable", "smalltalk")}
        )
    with pytest.raises(ValidationError):
        context_models.EffectivePolicyViewV1.model_validate(
            {
                **payload,
                "outbound_actions": ("send_material_pack",),
                "actions_allowed": True,
            }
        )
