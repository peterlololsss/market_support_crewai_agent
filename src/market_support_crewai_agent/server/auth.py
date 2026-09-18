from __future__ import annotations

from hmac import compare_digest
from typing import Annotated

from fastapi import Header, HTTPException

from market_support_crewai_agent.settings import get_settings


def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    expected = get_settings().api_key
    if not expected:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "v2_adapter_auth_required",
                "message": "v2_adapter_auth_required",
            },
        )

    candidates: list[str] = []
    if authorization and authorization.startswith("Bearer "):
        candidates.append(authorization[len("Bearer ") :])
    if x_api_key:
        candidates.append(x_api_key)

    if any(compare_digest(candidate, expected) for candidate in candidates):
        return
    raise HTTPException(status_code=401, detail="unauthorized")
