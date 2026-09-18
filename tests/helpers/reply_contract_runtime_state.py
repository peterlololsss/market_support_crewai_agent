from __future__ import annotations

import pytest

from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)


@pytest.fixture(autouse=True)
def isolate_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import market_support_crewai_agent.runtime.turn as runtime_turn

    monkeypatch.setattr(
        runtime_turn,
        "get_reply_state_coordinator",
        lambda: ReplyStateTransactionCoordinatorV1(),
    )
    monkeypatch.setattr(runtime_turn, "_APP_ACTION_LEDGER", ActionLedger())
    monkeypatch.setenv("MARKET_AGENT_API_KEY", "integration-key")
    monkeypatch.setenv("MARKET_AGENT_DEPLOYMENT_TENANT_REF", "tenant:test")
