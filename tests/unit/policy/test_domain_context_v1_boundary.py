from importlib import import_module

import pytest

from market_support_crewai_agent.runtime.policy.ontology import (
    DomainContextV1Builder,
    normalize_artifact_type,
)


def test_domain_context_v1_is_the_only_runtime_boundary_name() -> None:
    # Given: the orchestration and canonical DTO owner modules are loaded directly.
    ontology = import_module("market_support_crewai_agent.runtime.policy.ontology")
    models = import_module("market_support_crewai_agent.runtime.policy.ontology_models")

    # When/Then: the DTO and builder have distinct canonical owners without a bridge.
    assert hasattr(models, "DomainContextV1")
    assert hasattr(ontology, "DomainContextV1Builder")
    assert not hasattr(ontology, "DomainContextV1")
    assert not hasattr(ontology, "DomainContext")
    assert not hasattr(ontology, "DomainContextBuilder")


def test_normalize_artifact_type_uses_only_canonical_exact_values() -> None:
    # Given: canonical and malformed artifact selectors from adapter-like data.
    malformed_type = "weekly_report_extra"

    # When/Then: exact canonical values pass and malformed values collapse to unknown.
    assert normalize_artifact_type("weekly_report") == "weekly_report"
    assert normalize_artifact_type(malformed_type) == "unknown"
    assert normalize_artifact_type(resolve_type="sales_mention") == "adapter_context"
    assert (
        normalize_artifact_type(
            resolve_type="weekly_report",
            fact_type="report_scope_summary",
        )
        == "weekly_report"
    )


@pytest.mark.parametrize(
    ("artifact_type", "expected"),
    (
        ("material_pack", "material_pack"),
        ("weekly_report", "weekly_report"),
        ("monthly_report", "monthly_report"),
        ("document_context", "document_context"),
    ),
)
def test_normalize_artifact_type_maps_exact_typed_artifact_values(
    artifact_type: str,
    expected: str,
) -> None:
    # Given: an exact closed-set artifact value from a typed boundary.
    # When: the ontology normalizer receives the structured artifact selector.
    actual = normalize_artifact_type(artifact_type)

    # Then: the exact value retains its canonical mapping.
    assert actual == expected


@pytest.mark.parametrize(
    ("resolve_type", "expected"),
    (
        ("material_pack", "material_pack"),
        ("weekly_report", "weekly_report"),
        ("monthly_report", "monthly_report"),
        ("sales_mention", "adapter_context"),
    ),
)
def test_normalize_artifact_type_maps_exact_closed_set_resolve_values(
    resolve_type: str,
    expected: str,
) -> None:
    # Given: an exact adapter resolve value from the closed-set contract.
    # When: the ontology normalizer receives the structured resolve selector.
    actual = normalize_artifact_type(resolve_type=resolve_type)

    # Then: the exact resolve value retains its canonical mapping.
    assert actual == expected


@pytest.mark.parametrize(
    "fact_type",
    (
        "material_pack_arbitrary",
        "weekly_report_arbitrary",
        "monthly_report_arbitrary",
        "report_arbitrary",
        "MATERIAL_PACK_arbitrary",
        " Weekly_report_arbitrary ",
        "prefix_weekly_report_arbitrary",
        "weekly_report_arbitrary_suffix",
        "weekly_report_arbitrary中文",
        "中文monthly_report_arbitrary",
        "monthly_report",
        "report_scope_summary",
    ),
)
def test_normalize_artifact_type_rejects_fact_only_semantic_selectors(
    fact_type: str,
) -> None:
    # Given: untrusted fact text without an exact typed artifact or resolve value.
    # When: the ontology normalizer receives only that fact label.
    actual = normalize_artifact_type(fact_type=fact_type)

    # Then: semantic prefix, case, whitespace, embedding, and adjacency carry no authority.
    assert actual == "unknown"


def test_domain_context_available_artifact_malformed_type_is_kept_as_unknown() -> None:
    # Given: a legacy mapping payload carries a non-canonical artifact type.
    payload = {
        "dist_channel_name": "银河证券",
        "channel_type": "non_bank",
        "available_artifacts": [{"type": "calendar", "options": ["ignored"]}],
    }

    # When: the ontology builder normalizes adapter-available artifacts.
    context = DomainContextV1Builder().build(payload)

    # Then: the artifact is still represented, but only as the canonical unknown type.
    assert len(context.artifacts) == 1
    artifact = context.artifacts[0]
    assert artifact.artifact_type == "unknown"
    assert artifact.title == "calendar"
    assert artifact.scope.source_id == "adapter_channel.available_artifacts:calendar"


def test_domain_context_channel_kind_requires_exact_canonical_value() -> None:
    # Given: a payload with case-drifted channel type text.
    payload = {"dist_channel_name": "银河证券", "channel_type": "Non_Bank"}

    # When: the channel kind is parsed.
    context = DomainContextV1Builder().build(payload)

    # Then: no fuzzy or case-folded channel selection is applied.
    assert context.channel_kind == "unknown"
