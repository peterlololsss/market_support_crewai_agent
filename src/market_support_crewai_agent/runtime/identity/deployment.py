from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DeploymentIdentityCode = Literal[
    "deployment_identity_unavailable",
    "deployment_tenant_mismatch",
]


@dataclass(frozen=True, slots=True)
class DeploymentIdentityError(Exception):
    status_code: Literal[403, 503]
    code: DeploymentIdentityCode

    def __str__(self) -> str:
        return self.code


def validate_deployment_identity(
    request_tenant_ref: str,
    configured_tenant_ref: str | None,
) -> None:
    if not configured_tenant_ref:
        raise DeploymentIdentityError(
            status_code=503,
            code="deployment_identity_unavailable",
        )
    if request_tenant_ref != configured_tenant_ref:
        raise DeploymentIdentityError(
            status_code=403,
            code="deployment_tenant_mismatch",
        )
