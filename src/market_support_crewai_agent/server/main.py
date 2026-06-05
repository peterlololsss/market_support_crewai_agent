from __future__ import annotations

from hmac import compare_digest

from fastapi import Depends, FastAPI, Header, HTTPException

from market_support_crewai_agent.runtime.action_ledger import get_action_ledger
from market_support_crewai_agent.runtime.input_guardrails import (
    InputGuardrailError,
    validate_reply_request_input,
)
from market_support_crewai_agent.runtime.reply_agent import (
    AgentRuntimeError,
    build_reply,
)
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    ActionFeedbackResponse,
    HealthResponse,
    ReplyRequest,
    ReplyResponse,
)
from market_support_crewai_agent.settings import get_settings


app = FastAPI(
    title="market-support-crewai-agent",
    version="0.1.0",
    description="External agent-brain API for typed reply and action decisions.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="market-support-crewai-agent")


def require_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = get_settings().api_key
    if not expected:
        return None

    candidates = []
    if authorization and authorization.startswith("Bearer "):
        candidates.append(authorization[len("Bearer "):])
    if x_api_key:
        candidates.append(x_api_key)

    if any(compare_digest(candidate, expected) for candidate in candidates):
        return None
    raise HTTPException(status_code=401, detail="unauthorized")


@app.post(
    "/reply",
    response_model=ReplyResponse,
    response_model_exclude_none=True,
)
async def reply(
    request: ReplyRequest,
    _authorized: None = Depends(require_api_key),
) -> ReplyResponse:
    try:
        validate_reply_request_input(request, get_settings())
        return await build_reply(request)
    except InputGuardrailError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except AgentRuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post(
    "/actions/feedback",
    response_model=ActionFeedbackResponse,
)
async def action_feedback(
    request: ActionFeedbackRequest,
    _authorized: None = Depends(require_api_key),
) -> ActionFeedbackResponse:
    stored = get_action_ledger().record_feedback(request)
    return ActionFeedbackResponse(status="accepted", stored=stored)
