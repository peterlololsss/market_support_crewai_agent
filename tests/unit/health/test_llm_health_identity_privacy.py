from __future__ import annotations

# pyright: reportPrivateUsage=false
import base64
import hashlib
import hmac
import logging
from dataclasses import fields
from datetime import datetime, timedelta, timezone

import anyio
import pytest

from market_support_crewai_agent.health import llm_health
from market_support_crewai_agent.health.models import HealthTargetSlot
from market_support_crewai_agent.health.probe import (
    discover_llm_health_targets,
    health_target_key,
)
from market_support_crewai_agent.runtime.hashing import (
    CanonicalValue,
    canonical_json_bytes,
)
from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderIdV1,
    TransportVariantV1,
)
from market_support_crewai_agent.settings_model import Settings

_PROCESS_KEY = b"k" * 32


def test_htk1_is_process_keyed_opaque_and_covers_canonical_target() -> None:
    # Given: one canonical provider target and a fixed process-local key.
    target_payload: CanonicalValue = {
        "target_slot": "composer",
        "provider_id": "openai_compatible",
        "model_family": "gpt",
        "model_name": "private-gpt-model-canary",
        "normalized_endpoint": "https://private-endpoint-canary.invalid/v1",
        "transport_variant": "openai_chat_completions",
    }

    # When: health identities are derived for the baseline and each component mutation.
    baseline = _health_target_key()
    mutations = (
        _health_target_key(target_slot="planner"),
        _health_target_key(provider_id="gemini"),
        _health_target_key(model_family="generic"),
        _health_target_key(model_name="private-gpt-model-canary-v2"),
        _health_target_key(
            normalized_endpoint="https://private-endpoint-canary.invalid/v2"
        ),
        _health_target_key(transport_variant="gemini_generate_content"),
    )
    other_process = _health_target_key(process_health_key=b"z" * 32)

    # Then: the value is the exact domain-separated 16-byte HMAC projection.
    digest = hmac.new(
        _PROCESS_KEY,
        b"llm-health-target.v1\0" + canonical_json_bytes(target_payload),
        hashlib.sha256,
    ).digest()[:16]
    expected = "htk1:" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    assert baseline == expected
    assert len(baseline) == 27
    assert len({baseline, *mutations, other_process}) == 8


def _health_target_key(
    *,
    process_health_key: bytes = _PROCESS_KEY,
    target_slot: HealthTargetSlot = "composer",
    provider_id: ProviderIdV1 = "openai_compatible",
    model_family: ModelFamily = "gpt",
    model_name: str = "private-gpt-model-canary",
    normalized_endpoint: str = "https://private-endpoint-canary.invalid/v1",
    transport_variant: TransportVariantV1 = "openai_chat_completions",
) -> str:
    return health_target_key(
        process_health_key=process_health_key,
        target_slot=target_slot,
        provider_id=provider_id,
        model_family=model_family,
        model_name=model_name,
        normalized_endpoint=normalized_endpoint,
        transport_variant=transport_variant,
    )


def test_health_target_retains_only_opaque_identity() -> None:
    # Given: raw provider coordinates carrying unique privacy canaries.
    settings = Settings(
        llm_provider="yanfu",
        llm_model="private-gpt-model-canary",
        llm_base_url="https://PRIVATE-ENDPOINT-CANARY.invalid/v1/",
        llm_api_key="key",
    )

    # When: health projects the configured provider into durable target identity.
    target = llm_health.LlmHealthMonitor(
        settings,
        process_health_key=_PROCESS_KEY,
    ).targets[0]

    # Then: the retained target has only a slot and process-local opaque ID.
    assert {item.name for item in fields(target)} == {"target_slot", "htk1"}
    assert "yanfu" not in repr(target).lower()
    assert "private-gpt-model-canary" not in repr(target)
    assert "private-endpoint-canary" not in repr(target).lower()


def test_health_state_logs_notifications_and_reports_do_not_expose_raw_target(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: provider coordinates carrying unique privacy canaries.
    settings = Settings(
        llm_provider="yanfu",
        llm_model="private-gpt-model-canary",
        llm_base_url="https://PRIVATE-ENDPOINT-CANARY.invalid/v1/",
        llm_api_key="key",
        planner_llm_provider="yanfu",
        planner_llm_model="private-gpt-model-canary",
        planner_llm_base_url="https://PRIVATE-ENDPOINT-CANARY.invalid/v1",
        planner_llm_api_key="key",
    )
    monitor = llm_health.LlmHealthMonitor(
        settings,
        process_health_key=_PROCESS_KEY,
    )
    notifications: list[str] = []
    monkeypatch.setattr(monitor, "_notify", notifications.append)
    target = monitor.targets[0]
    failed_at = datetime(2026, 7, 19, 9, 0, tzinfo=timezone.utc)
    recovered_at = failed_at + timedelta(minutes=3)

    # When: warning, recovery, daily report, and notification-log paths execute.
    monkeypatch.setattr(monitor, "_now", lambda: failed_at)
    monitor.mark_failure(target, reason="provider_timeout")
    monkeypatch.setattr(monitor, "_now", lambda: recovered_at)
    monitor.mark_success(target)
    report = monitor.format_daily_report(failed_at, recovered_at)

    class FailingSender:
        def send_text(self, text: str) -> None:
            del text
            raise RuntimeError(
                "yanfu private-gpt-model-canary "
                + "https://PRIVATE-ENDPOINT-CANARY.invalid/v1/"
            )

    monkeypatch.setattr(monitor, "sender", FailingSender())
    caplog.set_level(logging.WARNING)
    anyio.run(monitor._send_text, "opaque health message")

    # Then: only opaque identity, slot, status/timing, and closed codes are exposed.
    exposed = (
        repr(monitor.states[target.key]) + "".join(notifications) + report + caplog.text
    )
    assert [item.target_slot for item in monitor.targets] == ["composer", "planner"]
    assert monitor.targets[0].htk1 != monitor.targets[1].htk1
    assert target.htk1 in "".join(notifications) + report
    assert "yanfu" not in exposed.lower()
    assert "private-gpt-model-canary" not in exposed
    assert "private-endpoint-canary" not in exposed.lower()


def test_slash_aliases_share_health_identity_within_one_process() -> None:
    # Given: two settings objects whose provider URLs dispatch identically.
    first = Settings(
        llm_api_key="key",
        llm_base_url="https://EXAMPLE.com/v1",
    )
    second = Settings(
        llm_api_key="key",
        llm_base_url="https://example.com/v1/",
    )

    # When: both target sets use the same process-local health key.
    first_targets = discover_llm_health_targets(
        first,
        process_health_key=_PROCESS_KEY,
    )
    second_targets = discover_llm_health_targets(
        second,
        process_health_key=_PROCESS_KEY,
    )

    # Then: slash/case aliases share identity for each configured target slot.
    assert tuple(target.htk1 for target in first_targets) == tuple(
        target.htk1 for target in second_targets
    )
