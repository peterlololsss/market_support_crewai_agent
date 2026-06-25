from __future__ import annotations

from datetime import datetime, timezone

import pytest

from market_support_crewai_agent.runtime.context.models import ContextProjectionPolicy
from market_support_crewai_agent.runtime.context.pressure import (
    ContextPressureEstimator,
    ProjectionLimitError,
)
from market_support_crewai_agent.runtime.context.projection import ContextProjectionManager
from market_support_crewai_agent.runtime.domain.policy import compile_policy
from market_support_crewai_agent.runtime.state.conversation_store import ConversationMessage
from tests.helpers.planning import make_request


def test_pressure_estimate_drops_after_projection_and_warns_before_hard_limit():
    estimator = ContextPressureEstimator()
    policy = ContextProjectionPolicy(
        token_budget=1000,
        warning_threshold=0.4,
        hard_threshold=0.9,
    )

    pressure = estimator.estimate(
        raw_context={"history": "x" * 20000},
        projected_context={"summary": "x" * 2000},
        policy=policy,
    )

    assert pressure.estimated_tokens_before > pressure.estimated_tokens_after
    assert pressure.warning is True
    assert pressure.hard_blocked is False


def test_projection_limit_error_only_after_projected_context_exceeds_hard_limit():
    request = make_request(message="current" * 100)
    manager = ContextProjectionManager(
        ContextProjectionPolicy(token_budget=10, hard_threshold=0.5)
    )

    with pytest.raises(ProjectionLimitError):
        manager.project_for_stage(
            stage="planner_intent",
            request=request,
            policy=compile_policy(request),
            history=[
                ConversationMessage("assistant", "old" * 1000, datetime.now(timezone.utc))
            ],
        )
