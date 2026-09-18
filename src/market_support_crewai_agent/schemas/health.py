from typing import Literal

from market_support_crewai_agent.schemas.base import StrictModel


class HealthResponse(StrictModel):
    status: Literal["ok"]
    service: str
