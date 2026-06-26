from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from market_support_crewai_agent.health.llm_health import (
    LlmHealthMonitor,
    discover_llm_health_targets,
)
from market_support_crewai_agent.settings import Settings


def test_discover_health_targets_dedupes_default_planner_model():
    targets = discover_llm_health_targets(Settings(llm_api_key="key"))

    assert len(targets) == 1
    assert targets[0].model == "deepseek-v4-pro"
    assert targets[0].stage_label == "composer/alignment/planner"
    assert targets[0].api_key_configured is True


def test_discover_health_targets_includes_planner_override():
    targets = discover_llm_health_targets(
        Settings(
            llm_api_key="ds-key",
            planner_llm_provider="gemini",
            planner_llm_model="gemini-3-flash-preview",
            planner_llm_base_url="https://gemini.local/v1",
            planner_llm_api_key="gemini-key",
        )
    )

    assert [target.model for target in targets] == [
        "deepseek-v4-pro",
        "gemini-3-flash-preview",
    ]
    assert targets[1].stage_label == "planner"
    assert targets[1].api_key_configured is True


def test_agent_hook_matches_gemini_client_params_base_url():
    settings = Settings(
        llm_api_key="ds-key",
        planner_llm_provider="gemini",
        planner_llm_model="gemini-3-flash-preview",
        planner_llm_base_url="https://gemini.local/v1",
        planner_llm_api_key="gemini-key",
    )
    monitor = LlmHealthMonitor(settings)
    agent = SimpleNamespace(
        llm=SimpleNamespace(
            provider="gemini",
            model="gemini-3-flash-preview",
            base_url=None,
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

    monitor = LlmHealthMonitor(Settings(llm_api_key="key"))
    target = monitor.targets[0]

    monitor._now = lambda: failed_at  # type: ignore[method-assign]
    monitor.mark_failure(target, reason="empty structured output", stage="planner")
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
    assert "deepseek-v4-pro：正常" in report
    assert "Provider" not in report
    assert "阶段" not in report
    assert "不可用时间段：09:10~09:15（5分钟）" in report
    assert "今日总不可用：5分钟" in report
    assert "可用率：91.67%" in report


def test_unhealthy_report_uses_now_for_open_outage():
    start = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    failed_at = start + timedelta(minutes=45)
    end = start + timedelta(hours=1)

    monitor = LlmHealthMonitor(Settings(llm_api_key="key"))
    target = monitor.targets[0]

    monitor._now = lambda: failed_at  # type: ignore[method-assign]
    monitor.mark_failure(target, reason="timeout", stage="composer")
    monitor._now = lambda: end  # type: ignore[method-assign]

    report = monitor.format_daily_report(start, end)

    assert "deepseek-v4-pro：异常" in report
    assert "已不可用：15分钟" in report
    assert "不可用时间段：09:45~现在（15分钟）" in report
    assert "最后错误：timeout" in report


def test_unknown_report_does_not_claim_full_availability():
    start = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    report = LlmHealthMonitor(Settings(llm_api_key="key")).format_daily_report(
        start,
        end,
    )

    assert "检测结果：尚未完成首次健康检查" in report
    assert "可用率：100.00%" not in report
