from __future__ import annotations

from typing import Annotated

from anyio.to_thread import run_sync
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import ValidationError

from market_support_crewai_agent.runtime.identity import (
    DeploymentIdentityError,
    feedback_state_key_v2,
    normalize_reply_request_v2,
    validate_deployment_identity,
)
from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
)
from market_support_crewai_agent.runtime.observability.direct_audit import (
    DirectAuditKeyError,
    decode_direct_audit_hmac_key,
)
from market_support_crewai_agent.runtime.service import (
    build_reply,
)
from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.coordinator_provider import (
    get_reply_state_coordinator,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    FeedbackReceiptReplayV1,
    PreparedFeedbackV1,
)
from market_support_crewai_agent.runtime.turn import (
    AgentRuntimeError,
)
from market_support_crewai_agent.runtime.validation.reply_validator import (
    ReplyContractError,
)
from market_support_crewai_agent.runtime.validation.request_input_guard import (
    InputGuardrailError,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.feedback import (
    ActionFeedbackRequestV2,
    ActionFeedbackResponse,
)
from market_support_crewai_agent.schemas.health import HealthResponse
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.server import adapter_compatibility, auth, lifespan
from market_support_crewai_agent.server.logging_config import configure_app_logging
from market_support_crewai_agent.settings import get_settings

__all__ = ("app",)


configure_app_logging()


app = FastAPI(
    title="market-support-crewai-agent",
    version="0.1.0",
    description="External agent-brain API for typed reply and action decisions.",
    lifespan=lifespan.app_lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="market-support-crewai-agent")


@app.post(
    "/reply",
    response_model=ReplyResponse,
    response_model_exclude_none=True,
)
async def reply(
    http_request: Request,
    _authorized: Annotated[None, Depends(auth.require_api_key)],
) -> ReplyResponse:
    try:
        settings = get_settings()
        body = await http_request.body()
        if len(body) > 65_536:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "request_body_too_large",
                    "message": "request_body_too_large",
                },
            )
        try:
            request = ReplyRequestV2.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_request_contract",
                    "message": "invalid_request_contract",
                },
            ) from exc
        validate_deployment_identity(
            request.identity.tenant_ref,
            settings.deployment_tenant_ref,
        )
        if request.identity.scene == "direct":
            if not settings.internal_dm_enabled:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "internal_dm_disabled",
                        "message": "internal_dm_disabled",
                    },
                )
            try:
                _ = decode_direct_audit_hmac_key(settings.direct_audit_hmac_key or "")
            except DirectAuditKeyError as exc:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "internal_dm_audit_unavailable",
                        "message": "internal_dm_audit_unavailable",
                    },
                ) from exc
            if not settings.adapter_api_key or not settings.adapter_api_key.strip():
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "internal_dm_adapter_auth_required",
                        "message": "internal_dm_adapter_auth_required",
                    },
                )
            try:
                _ = await run_sync(
                    adapter_compatibility.compatibility_client().assert_scene_compatible,
                    "direct",
                    request.identity.tenant_ref,
                )
            except AdapterClientError as exc:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "internal_dm_adapter_incompatible",
                        "message": "internal_dm_adapter_incompatible",
                    },
                ) from exc
        if settings.agent_input_max_message_chars is not None and (
            len(request.message) > settings.agent_input_max_message_chars
        ):
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "request_body_too_large",
                    "message": "request_body_too_large",
                },
            )
        envelope = normalize_reply_request_v2(
            request,
            adapter_namespace=settings.adapter_caller_namespace,
        )
        return await build_reply(envelope)
    except DeploymentIdentityError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.code},
        ) from exc
    except InputGuardrailError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ReplyContractError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except AgentRuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except CoordinatorError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": exc.code},
        ) from exc


@app.post(
    "/actions/feedback",
    response_model=ActionFeedbackResponse,
)
async def action_feedback(
    http_request: Request,
    _authorized: Annotated[None, Depends(auth.require_api_key)],
) -> ActionFeedbackResponse:
    settings = get_settings()
    body = await http_request.body()
    if len(body) > 65_536:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "request_body_too_large",
                "message": "request_body_too_large",
            },
        )
    try:
        request_v2 = ActionFeedbackRequestV2.model_validate_json(
            body,
            context={
                "configured_secret_values": tuple(
                    secret
                    for secret in (
                        settings.api_key,
                        settings.adapter_api_key,
                        settings.llm_api_key,
                        settings.planner_llm_api_key,
                        settings.feishu_app_secret,
                    )
                    if secret
                )
            },
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_feedback_contract",
                "message": "invalid_feedback_contract",
            },
        ) from exc
    try:
        validate_deployment_identity(
            request_v2.identity.tenant_ref,
            settings.deployment_tenant_ref,
        )
    except DeploymentIdentityError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.code},
        ) from exc
    state_key = feedback_state_key_v2(
        request_v2,
        adapter_namespace=settings.adapter_caller_namespace,
    )
    try:
        coordinator = get_reply_state_coordinator()
        preparation = coordinator.prepare_feedback(state_key, request_v2)
        match preparation:
            case FeedbackReceiptReplayV1(stored=stored):
                return ActionFeedbackResponse(status="accepted", stored=stored)
            case PreparedFeedbackV1():
                result = coordinator.commit_feedback(preparation)
                return ActionFeedbackResponse(status="accepted", stored=result.stored)
    except CoordinatorError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": exc.code},
        ) from exc
