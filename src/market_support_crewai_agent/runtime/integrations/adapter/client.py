from __future__ import annotations

import asyncio
import json
import threading
from _thread import LockType
from typing import Literal

from market_support_crewai_agent.runtime.integrations.adapter.capability_validation import (
    adapter_capability_errors,
    adapter_tenant_errors,
    canonical_deployment_tenant_ref,
    scene_compatibility_errors,
)
from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
    AdapterTransport,
)
from market_support_crewai_agent.schemas.adapter import (
    AdapterReportScopeRequest,
    AdapterReportScopeResult,
    AdapterResolveBatchRequest,
    AdapterResolveBatchResult,
    AdapterResolveRequest,
    AdapterResolveResult,
)
from market_support_crewai_agent.schemas.adapter_metadata import (
    AdapterCapabilities,
    AdapterMetrics,
)
from market_support_crewai_agent.settings import get_settings
from market_support_crewai_agent.settings_model import Settings


class AdapterResolveClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.timeout: float = self.settings.adapter_timeout_seconds
        self.api_key: str | None = self.settings.adapter_api_key
        self._transport: AdapterTransport = AdapterTransport.create(
            self.settings.adapter_base_url,
            self.timeout,
            self.api_key,
        )
        self.base_url: str = self._transport.endpoint.base_url
        self._ready_capabilities: AdapterCapabilities | None = None
        self._ready_lock: LockType = threading.Lock()

    def capabilities(self) -> AdapterCapabilities:
        raw = self._get_json("/adapter/capabilities")
        try:
            return AdapterCapabilities.model_validate_json(raw)
        except ValueError:
            raise AdapterClientError(
                "adapter capabilities returned an invalid contract"
            ) from None

    def assert_ready(self) -> AdapterCapabilities:
        if self._ready_capabilities is not None:
            return self._ready_capabilities

        with self._ready_lock:
            if self._ready_capabilities is not None:
                return self._ready_capabilities
            capabilities = self.capabilities()
            errors = adapter_capability_errors(capabilities)
            if errors:
                raise AdapterClientError(
                    "adapter capabilities mismatch: {}".format("; ".join(errors))
                )
            self._ready_capabilities = capabilities
            return capabilities

    def assert_scene_compatible(
        self,
        scene: Literal["direct", "group"],
        deployment_tenant_ref: str,
    ) -> AdapterCapabilities:
        if self.api_key is None or not self.api_key.strip():
            raise AdapterClientError("adapter authentication is required")
        try:
            canonical_tenant_ref = canonical_deployment_tenant_ref(
                deployment_tenant_ref
            )
        except ValueError as exc:
            raise AdapterClientError(
                "adapter scene compatibility rejected invalid deployment tenant"
            ) from exc

        capabilities = self.assert_ready()
        errors = scene_compatibility_errors(
            capabilities,
            scene,
            canonical_tenant_ref,
        )
        if errors:
            raise AdapterClientError(
                "adapter scene compatibility mismatch: {}".format("; ".join(errors))
            )
        return capabilities

    def metrics(self) -> AdapterMetrics:
        raw = self._get_json("/adapter/metrics")
        try:
            return AdapterMetrics.model_validate_json(raw)
        except ValueError:
            raise AdapterClientError(
                "adapter metrics returned an invalid contract"
            ) from None

    def resolve(self, request: AdapterResolveRequest) -> AdapterResolveResult:
        raw = self._post_json(
            "/adapter/resolve",
            request.model_dump(mode="json", exclude_none=True),
        )
        try:
            return AdapterResolveResult.model_validate_json(raw)
        except ValueError:
            raise AdapterClientError(
                "adapter resolve returned an invalid contract"
            ) from None

    def resolve_many(
        self,
        requests: list[AdapterResolveRequest],
    ) -> list[AdapterResolveResult]:
        batch_request = AdapterResolveBatchRequest(requests=requests)
        raw = self._post_json(
            "/adapter/resolve/batch",
            batch_request.model_dump(mode="json", exclude_none=True),
        )
        try:
            return AdapterResolveBatchResult.model_validate_json(raw).results
        except ValueError:
            raise AdapterClientError(
                "adapter batch resolve returned an invalid contract"
            ) from None

    def report_scope(
        self, request: AdapterReportScopeRequest
    ) -> AdapterReportScopeResult:
        raw = self._post_json(
            "/adapter/report-scope",
            request.model_dump(mode="json", exclude_none=True),
        )
        try:
            return AdapterReportScopeResult.model_validate_json(raw)
        except ValueError:
            raise AdapterClientError(
                "adapter report-scope returned an invalid contract"
            ) from None

    def _get_json(self, path: str) -> str:
        return self._request_json("GET", path)

    def _post_json(self, path: str, payload: dict[str, object]) -> str:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return self._request_json("POST", path, body=body)

    def _request_json(
        self,
        method: Literal["GET", "POST"],
        path: str,
        body: bytes | None = None,
    ) -> str:
        return self._transport.request_json(method, path, body)

    async def resolve_async(
        self,
        request: AdapterResolveRequest,
    ) -> AdapterResolveResult:
        return await asyncio.to_thread(self.resolve, request)

    async def resolve_many_async(
        self,
        requests: list[AdapterResolveRequest],
    ) -> list[AdapterResolveResult]:
        return await asyncio.to_thread(self.resolve_many, requests)

    async def report_scope_async(
        self,
        request: AdapterReportScopeRequest,
    ) -> AdapterReportScopeResult:
        return await asyncio.to_thread(self.report_scope, request)

    async def capabilities_async(self) -> AdapterCapabilities:
        return await asyncio.to_thread(self.capabilities)

    async def assert_ready_async(self) -> AdapterCapabilities:
        return await asyncio.to_thread(self.assert_ready)

    async def assert_ready_for_tenant_async(
        self,
        tenant_ref: str,
    ) -> AdapterCapabilities:
        try:
            canonical_tenant_ref = canonical_deployment_tenant_ref(tenant_ref)
        except ValueError as exc:
            raise AdapterClientError(
                "adapter readiness rejected invalid deployment tenant"
            ) from exc
        capabilities = await self.assert_ready_async()
        errors = adapter_tenant_errors(capabilities, canonical_tenant_ref)
        if errors:
            raise AdapterClientError(
                "adapter capabilities mismatch: {}".format("; ".join(errors))
            )
        return capabilities

    async def metrics_async(self) -> AdapterMetrics:
        return await asyncio.to_thread(self.metrics)
