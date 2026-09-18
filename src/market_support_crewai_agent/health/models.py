from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from market_support_crewai_agent.runtime.prompts.provider_errors import (
    ProviderFailureCodeV1,
)

HealthTargetSlot = Literal["composer", "planner"]
HealthStatus = Literal["unknown", "healthy", "unhealthy"]


class HealthProbeResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    ok: bool


@dataclass(frozen=True, slots=True)
class LlmHealthTarget:
    target_slot: HealthTargetSlot
    htk1: str

    @property
    def key(self) -> str:
        return self.htk1


@dataclass(slots=True)
class OutageWindow:
    """Mutable outage state updated when a provider recovers."""

    started_at: datetime
    ended_at: datetime | None = None
    reason: ProviderFailureCodeV1 | Literal[""] = ""


@dataclass(slots=True)
class LlmHealthState:
    """Mutable monitor state owned by the long-lived process health loop."""

    status: HealthStatus = "unknown"
    healthy_since: datetime | None = None
    unhealthy_since: datetime | None = None
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    last_probe_at: datetime | None = None
    last_warning_at: datetime | None = None
    last_error: ProviderFailureCodeV1 | Literal[""] = ""
    outages: list[OutageWindow] = field(default_factory=list)

    def record_success(self, now: datetime) -> OutageWindow | None:
        previous_outage = self.open_outage()
        self.status = "healthy"
        self.last_success_at = now
        self.last_error = ""
        self.unhealthy_since = None
        if self.healthy_since is None or previous_outage is not None:
            self.healthy_since = now
        if previous_outage is not None and previous_outage.ended_at is None:
            previous_outage.ended_at = now
            return previous_outage
        return None

    def record_failure(
        self,
        now: datetime,
        *,
        reason: ProviderFailureCodeV1,
        warning_cooldown_seconds: float,
    ) -> bool:
        first_failure = self.status != "unhealthy"
        self.status = "unhealthy"
        self.last_failure_at = now
        self.last_error = reason
        self.healthy_since = None
        if first_failure:
            self.unhealthy_since = now
            self.outages.append(OutageWindow(started_at=now, reason=reason))
        warning_due = self.warning_due(now, warning_cooldown_seconds)
        if first_failure or warning_due:
            self.last_warning_at = now
            return True
        return False

    def open_outage(self) -> OutageWindow | None:
        if self.status == "unhealthy" and self.outages:
            return self.outages[-1]
        return None

    def warning_due(self, now: datetime, cooldown_seconds: float) -> bool:
        if self.last_warning_at is None:
            return True
        return (now - self.last_warning_at).total_seconds() >= cooldown_seconds
