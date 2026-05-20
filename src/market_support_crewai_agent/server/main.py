from __future__ import annotations

from fastapi import FastAPI, HTTPException

from market_support_crewai_agent.runtime.reply_agent import (
    AgentRuntimeError,
    build_reply,
)
from market_support_crewai_agent.schemas import (
    HealthResponse,
    ReplyRequest,
    ReplyResponse,
)


app = FastAPI(
    title="market-support-crewai-agent",
    version="0.1.0",
    description="External agent-brain API for typed reply and action decisions.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="market-support-crewai-agent")


@app.post(
    "/reply",
    response_model=ReplyResponse,
    response_model_exclude_none=True,
)
async def reply(request: ReplyRequest) -> ReplyResponse:
    try:
        return await build_reply(request)
    except AgentRuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

