from __future__ import annotations

# pyright: reportMissingParameterType=false, reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnusedCallResult=false
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import anyio

from market_support_crewai_agent.health.llm_health import LlmHealthMonitor
from market_support_crewai_agent.health.models import HealthProbeResponse
from market_support_crewai_agent.health.probe import (
    discover_llm_health_targets,
    health_probe_program,
)
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    LlmHealthProbeOutputV1,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    build_provider_target_from_settings,
)
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import make_agent_adapter, make_llm_adapter

_PROCESS_HEALTH_KEY = b"h" * 32


def _monitor(settings: Settings) -> LlmHealthMonitor:
    return LlmHealthMonitor(
        settings,
        process_health_key=_PROCESS_HEALTH_KEY,
    )


def test_discover_health_targets_keeps_distinct_target_slots():
    targets = discover_llm_health_targets(
        Settings(llm_api_key="key"),
        process_health_key=_PROCESS_HEALTH_KEY,
    )

    assert [target.target_slot for target in targets] == ["composer", "planner"]
    assert all(target.htk1.startswith("htk1:") for target in targets)
    assert targets[0].htk1 != targets[1].htk1


def test_discover_health_targets_hashes_planner_override_independently():
    baseline = discover_llm_health_targets(
        Settings(llm_api_key="ds-key"),
        process_health_key=_PROCESS_HEALTH_KEY,
    )
    overridden = discover_llm_health_targets(
        Settings(
            llm_api_key="ds-key",
            planner_llm_provider="gemini",
            planner_llm_model="gemini-3-flash-preview",
            planner_llm_base_url="https://gemini.local/v1",
            planner_llm_api_key="gemini-key",
        ),
        process_health_key=_PROCESS_HEALTH_KEY,
    )

    assert baseline[0].htk1 == overridden[0].htk1
    assert baseline[1].htk1 != overridden[1].htk1


def test_agent_hook_matches_gemini_client_params_base_url():
    settings = Settings(
        llm_api_key="ds-key",
        planner_llm_provider="gemini",
        planner_llm_model="gemini-3-flash-preview",
        planner_llm_base_url="https://gemini.local/v1",
        planner_llm_api_key="gemini-key",
    )
    monitor = _monitor(settings)
    agent = make_agent_adapter(
        llm=make_llm_adapter(
            provider="gemini",
            model="gemini-3-flash-preview",
            api_key="gemini-key",
            client_params={"http_options": {"base_url": "https://gemini.local/v1"}},
        )
    )

    monitor.mark_agent_success(agent, "planner_intent")

    assert len(monitor.targets) == 2
    assert monitor.states[monitor.targets[1].key].status == "healthy"


def test_health_state_tracks_outage_window_and_chinese_daily_report():
    start = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    failed_at = start + timedelta(minutes=10)
    recovered_at = start + timedelta(minutes=15)
    end = start + timedelta(hours=1)

    monitor = _monitor(Settings(llm_api_key="key"))
    target = monitor.targets[0]

    monitor._now = lambda: failed_at  # type: ignore[method-assign]
    monitor.mark_failure(
        target,
        reason="provider_output_contract",
    )
    state = monitor.states[target.key]

    assert state.status == "unhealthy"
    assert state.unhealthy_since == failed_at
    assert len(state.outages) == 1
    assert state.outages[0].ended_at is None

    monitor._now = lambda: recovered_at  # type: ignore[method-assign]
    monitor.mark_success(target)

    assert state.status == "healthy"
    assert state.healthy_since == recovered_at
    assert state.outages[0].ended_at == recovered_at

    monitor._now = lambda: end  # type: ignore[method-assign]
    report = monitor.format_daily_report(start, end)

    assert "【LLM 健康日报】" in report
    assert f"composer（{target.htk1}）：正常" in report
    assert "Provider" not in report
    assert "阶段" not in report
    assert "不可用时间段：09:10~09:15（5分钟）" in report
    assert "今日总不可用：5分钟" in report
    assert "可用率：91.67%" in report


def test_unhealthy_report_uses_now_for_open_outage():
    start = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    failed_at = start + timedelta(minutes=45)
    end = start + timedelta(hours=1)

    monitor = _monitor(Settings(llm_api_key="key"))
    target = monitor.targets[0]

    monitor._now = lambda: failed_at  # type: ignore[method-assign]
    monitor.mark_failure(target, reason="provider_timeout")
    monitor._now = lambda: end  # type: ignore[method-assign]

    report = monitor.format_daily_report(start, end)

    assert f"composer（{target.htk1}）：异常" in report
    assert "已不可用：15分钟" in report
    assert "不可用时间段：09:45~现在（15分钟）" in report
    assert "最后错误：provider_timeout" in report


def test_unknown_report_does_not_claim_full_availability():
    start = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    report = _monitor(Settings(llm_api_key="key")).format_daily_report(
        start,
        end,
    )

    assert "检测结果：尚未完成首次健康检查" in report
    assert "可用率：100.00%" not in report


def test_health_governance_accepts_only_zero_retry_and_bounded_timeout() -> None:
    # Given/When/Then: health probe governance rejects hidden retry settings.
    for kwargs in (
        {"llm_health_probe_retry_attempts": 1},
        {"llm_health_probe_retry_base_seconds": 0.1},
        {"llm_health_probe_timeout_seconds": 0.09},
        {"llm_health_probe_timeout_seconds": 30.1},
    ):
        try:
            _ = Settings.model_validate(kwargs)
        except ValueError:
            continue
        raise AssertionError(f"health governance accepted invalid settings: {kwargs}")


def test_health_probe_program_uses_packaged_static_budget() -> None:
    # Given/When: the operational health probe program is assembled.
    program = health_probe_program()

    # Then: the packaged neutral row and sealed static budget are active.
    assert program.program_id == "llm_health_probe.scene_neutral.v1@1"
    assert program.profile.stage == "llm_health_probe"
    assert program.profile.response_model is LlmHealthProbeOutputV1
    assert program.fragment_ids == ("instruction.registered_over_untrusted_data.v1",)
    assert program.static_bytes == 160
    assert program.baseline_bytes == 160
    assert program.allowed_max_bytes == 1184
    assert len(program.prompt_text.encode("utf-8")) <= program.allowed_max_bytes


def test_health_target_accepts_configured_bounded_timeout() -> None:
    # Given/When: health dispatch targets use the configured bounded timeout.
    target = build_provider_target_from_settings(
        provider="openai",
        target_slot="health",
        model="deepseek-v4-pro",
        base_url="https://LLM.example/v1/",
        api_key_configured=True,
        timeout_seconds=0.1,
        temperature=0.0,
        max_tokens=64,
    )

    # Then: zero-retry governance stays active without forcing the default.
    assert target.timeout_seconds == 0.1
    assert target.retry_attempts == 0
    assert target.retry_base_seconds == 0.0
    assert target.normalized_endpoint == "https://llm.example/v1"
    assert target.transport_variant == "openai_chat_completions"


def test_probe_target_dispatches_packaged_program_with_configured_timeout(
    monkeypatch,
) -> None:
    # Given: a health monitor with a fake dispatch boundary and no network.
    settings = Settings(
        llm_api_key="key",
        llm_health_probe_timeout_seconds=0.5,
    )
    monitor = _monitor(settings)
    target = monitor.targets[0]
    captured = {}

    async def fake_run_crewai_kickoff(
        agent,
        program,
        *,
        timeout_seconds,
        health_log_context,
    ):
        captured["agent"] = agent
        captured["program"] = program
        captured["timeout_seconds"] = timeout_seconds
        captured["health_log_context"] = health_log_context
        result = SimpleNamespace(
            pydantic=LlmHealthProbeOutputV1(
                contract_version="llm-health-probe-output.v1",
                ok=True,
            )
        )
        return result, SimpleNamespace()

    monkeypatch.setattr(
        "market_support_crewai_agent.health.probe.run_crewai_kickoff",
        fake_run_crewai_kickoff,
    )
    monkeypatch.setattr(
        monitor,
        "_build_probe_agent",
        lambda probe_target: SimpleNamespace(probe_target=probe_target),
    )

    # When: the target is probed through the production health method.
    anyio.run(monitor._probe_target, target)

    # Then: dispatch receives the packaged health program and configured timeout.
    assert captured["program"].program_id == "llm_health_probe.scene_neutral.v1@1"
    assert captured["timeout_seconds"] == 0.5
    assert monitor.states[target.key].status == "healthy"


def test_probe_target_redacts_provider_exception_details(monkeypatch) -> None:
    settings = Settings(llm_api_key="key", llm_health_probe_timeout_seconds=0.5)
    monitor = _monitor(settings)
    target = monitor.targets[0]

    async def failing_kickoff(
        agent,
        program,
        *,
        timeout_seconds,
        health_log_context,
    ):
        del agent, program, timeout_seconds, health_log_context
        raise RuntimeError("secret provider endpoint https://private.invalid")

    monkeypatch.setattr(
        "market_support_crewai_agent.health.probe.run_crewai_kickoff",
        failing_kickoff,
    )
    monkeypatch.setattr(
        monitor,
        "_build_probe_agent",
        lambda probe_target: SimpleNamespace(probe_target=probe_target),
    )

    anyio.run(monitor._probe_target, target)

    state = monitor.states[target.key]
    assert state.status == "unhealthy"
    assert state.last_error == "provider_transport_unavailable"
    assert "private.invalid" not in repr(state)


def test_probe_target_records_closed_output_contract_code(monkeypatch) -> None:
    settings = Settings(llm_api_key="key", llm_health_probe_timeout_seconds=0.5)
    monitor = _monitor(settings)
    target = monitor.targets[0]

    async def invalid_probe_output(
        agent,
        program,
        *,
        timeout_seconds,
        health_log_context,
    ):
        del agent, program, timeout_seconds, health_log_context
        return SimpleNamespace(
            pydantic=HealthProbeResponse(ok=False)
        ), SimpleNamespace()

    monkeypatch.setattr(
        "market_support_crewai_agent.health.probe.run_crewai_kickoff",
        invalid_probe_output,
    )
    monkeypatch.setattr(
        monitor,
        "_build_probe_agent",
        lambda probe_target: SimpleNamespace(probe_target=probe_target),
    )

    anyio.run(monitor._probe_target, target)

    state = monitor.states[target.key]
    assert state.status == "unhealthy"
    assert state.last_error == "provider_output_contract"
