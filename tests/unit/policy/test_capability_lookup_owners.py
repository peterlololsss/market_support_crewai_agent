from market_support_crewai_agent.runtime.policy.capabilities.mappings import (
    action_type_for_resolve,
    capability_by_action_type,
    capability_by_read_capability,
    capability_by_resolve_type,
    ordered_resolve_types,
    read_capability_for_resolve,
    resolve_type_for_action,
    resolve_type_for_read_capability,
)
from market_support_crewai_agent.runtime.policy.capabilities.queries import (
    capability_manifest_by_id,
)
from market_support_crewai_agent.runtime.policy.capabilities.runtime_projection import (
    adapter_resolve_types,
    capability_by_name,
    capability_registry_hash,
    outbound_action_types,
    read_capabilities,
)


def test_exact_capability_queries_reject_near_match_ids() -> None:
    # Given: one canonical manifest ID and one semantic near-match.
    # When: the canonical registry query owner performs exact lookups.
    canonical = capability_manifest_by_id("weekly_report.send")
    near_match = capability_manifest_by_id("weekly_report.send.extra")

    # Then: only the exact registry ID resolves.
    assert canonical is not None
    assert canonical.manifest_id == "weekly_report.send"
    assert near_match is None


def test_exact_capability_mappings_preserve_closed_set_relationships() -> None:
    # Given: the canonical weekly-report values at each typed boundary.
    # When: each direct mapping owner projects the related contract value.
    by_resolve = capability_by_resolve_type("weekly_report")
    by_action = capability_by_action_type("send_weekly_report")
    by_read = capability_by_read_capability("resolve_weekly_report")

    # Then: all exact mappings converge without accepting semantic near-matches.
    assert by_resolve == by_action == by_read == capability_by_name("weekly_report")
    assert resolve_type_for_action("send_weekly_report") == "weekly_report"
    assert action_type_for_resolve("weekly_report") == "send_weekly_report"
    assert read_capability_for_resolve("weekly_report") == "resolve_weekly_report"
    assert resolve_type_for_read_capability("resolve_weekly_report") == "weekly_report"
    assert capability_by_resolve_type("weekly_report_extra") is None
    assert capability_by_action_type(" send_weekly_report") is None
    assert capability_by_read_capability("RESOLVE_WEEKLY_REPORT") is None


def test_runtime_capability_projection_and_hash_remain_pinned() -> None:
    # Given: the unchanged canonical capability registry.
    # When: the runtime projection owner derives the bounded sets and registry hash.
    resolves = adapter_resolve_types()
    actions = outbound_action_types()
    reads = read_capabilities()

    # Then: the exact runtime authority and protected registry hash stay unchanged.
    assert resolves == {
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    }
    assert actions == {
        "send_material_pack",
        "send_weekly_report",
        "send_monthly_report",
    }
    assert reads == {
        "resolve_material_pack",
        "resolve_weekly_report",
        "resolve_monthly_report",
        "resolve_sales_mention",
        "query_internal_company_info",
    }
    assert ordered_resolve_types(set(resolves)) == [
        "material_pack",
        "weekly_report",
        "monthly_report",
        "sales_mention",
    ]
    assert (
        capability_registry_hash()
        == "sha256:23524943142ea01dd5d3ba37bf5db05d90b9198e78556c401e4be36ab5fda010"
    )
