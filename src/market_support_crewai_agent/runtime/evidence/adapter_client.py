from __future__ import annotations

import asyncio
import json
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from market_support_crewai_agent.runtime.domain.capabilities import adapter_resolve_types
from market_support_crewai_agent.schemas import (
    AdapterCapabilities,
    AdapterMetrics,
    AdapterReportScopeRequest,
    AdapterReportScopeResult,
    AdapterResolveBatchRequest,
    AdapterResolveBatchResult,
    AdapterResolveRequest,
    AdapterResolveResult,
    OutboundTargetKind,
    OutboundTargetResolveResult,
)
from market_support_crewai_agent.settings import Settings, get_settings


REQUIRED_ADAPTER_SERVICE = "xiaoyan-wecom-market-agent-adapter"
REQUIRED_RESOLVE_CONTRACT_VERSION = "adapter-resolve"
REQUIRED_BATCH_CONTRACT_VERSION = "adapter-resolve-batch"
REQUIRED_ACTION_CONTRACT_VERSION = "adapter-action"
REQUIRED_RESOLVE_TYPES = adapter_resolve_types()
REQUIRED_STATUSES = {
    "resolved",
    "missing",
    "ambiguous",
    "forbidden",
    "temporarily_unavailable",
}
REQUIRED_ENDPOINTS = {
    "health": "/health",
    "capabilities": "/adapter/capabilities",
    "metrics": "/adapter/metrics",
    "resolve": "/adapter/resolve",
    "batch_resolve": "/adapter/resolve/batch",
}
OPTIONAL_ENDPOINTS = {
    "report_scope": "/adapter/report-scope",
}
REQUIRED_MIN_BATCH_REQUESTS = len(REQUIRED_RESOLVE_TYPES)


class AdapterClientError(RuntimeError):
    """Raised when the WeCom adapter resolve API cannot return a valid result."""


class AdapterResolveClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.adapter_base_url.rstrip("/")
        self.timeout = self.settings.adapter_timeout_seconds
        self.api_key = self.settings.adapter_api_key
        self._ready_capabilities: AdapterCapabilities | None = None
        self._ready_lock = threading.Lock()

    def capabilities(self) -> AdapterCapabilities:
        raw = self._get_json("/adapter/capabilities")
        try:
            return AdapterCapabilities.model_validate_json(raw)
        except ValueError as exc:
            raise AdapterClientError("adapter capabilities returned an invalid contract") from exc

    def assert_ready(self) -> AdapterCapabilities:
        if self._ready_capabilities is not None:
            return self._ready_capabilities

        with self._ready_lock:
            if self._ready_capabilities is not None:
                return self._ready_capabilities
            capabilities = self.capabilities()
            errors = _adapter_capability_errors(capabilities)
            if errors:
                raise AdapterClientError(
                    "adapter capabilities mismatch: {}".format("; ".join(errors))
                )
            self._ready_capabilities = capabilities
            return capabilities

    def metrics(self) -> AdapterMetrics:
        raw = self._get_json("/adapter/metrics")
        try:
            return AdapterMetrics.model_validate_json(raw)
        except ValueError as exc:
            raise AdapterClientError("adapter metrics returned an invalid contract") from exc

    def resolve(self, request: AdapterResolveRequest) -> AdapterResolveResult:
        raw = self._post_json(
            "/adapter/resolve",
            request.model_dump(mode="json", exclude_none=True),
        )
        try:
            return AdapterResolveResult.model_validate_json(raw)
        except ValueError as exc:
            raise AdapterClientError("adapter resolve returned an invalid contract") from exc

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
        except ValueError as exc:
            raise AdapterClientError("adapter batch resolve returned an invalid contract") from exc

    def resolve_outbound_target(
        self,
        target_kind: OutboundTargetKind,
        target_name: str,
    ) -> OutboundTargetResolveResult:
        raw = self._post_json(
            "/adapter/resolve",
            {
                "resolve_type": "outbound_message_target",
                "target_kind": target_kind,
                "target_name": target_name,
            },
        )
        try:
            return OutboundTargetResolveResult.model_validate_json(raw)
        except ValueError as exc:
            raise AdapterClientError(
                "adapter outbound target resolve returned an invalid contract"
            ) from exc

    def report_scope(self, request: AdapterReportScopeRequest) -> AdapterReportScopeResult:
        raw = self._post_json(
            "/adapter/report-scope",
            request.model_dump(mode="json", exclude_none=True),
        )
        try:
            return AdapterReportScopeResult.model_validate_json(raw)
        except ValueError as exc:
            raise AdapterClientError("adapter report-scope returned an invalid contract") from exc

    def _get_json(self, path: str) -> str:
        return self._request_json("GET", path)

    def _post_json(self, path: str, payload: dict) -> str:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return self._request_json("POST", path, body=body)

    def _request_json(self, method: str, path: str, body: bytes | None = None) -> str:
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        http_request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AdapterClientError(
                f"adapter request returned HTTP {exc.code}: {detail}"
            ) from exc
        except TimeoutError as exc:
            raise AdapterClientError(f"adapter request timed out: {exc}") from exc
        except URLError as exc:
            raise AdapterClientError(f"adapter request failed: {exc}") from exc

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

    async def resolve_outbound_target_async(
        self,
        target_kind: OutboundTargetKind,
        target_name: str,
    ) -> OutboundTargetResolveResult:
        return await asyncio.to_thread(
            self.resolve_outbound_target,
            target_kind,
            target_name,
        )

    async def report_scope_async(
        self,
        request: AdapterReportScopeRequest,
    ) -> AdapterReportScopeResult:
        return await asyncio.to_thread(self.report_scope, request)

    async def capabilities_async(self) -> AdapterCapabilities:
        return await asyncio.to_thread(self.capabilities)

    async def assert_ready_async(self) -> AdapterCapabilities:
        return await asyncio.to_thread(self.assert_ready)

    async def metrics_async(self) -> AdapterMetrics:
        return await asyncio.to_thread(self.metrics)


def _adapter_capability_errors(capabilities: AdapterCapabilities) -> list[str]:
    errors: list[str] = []
    if capabilities.service != REQUIRED_ADAPTER_SERVICE:
        errors.append("service={}".format(capabilities.service))
    if capabilities.contract_version != REQUIRED_RESOLVE_CONTRACT_VERSION:
        errors.append("contract_version={}".format(capabilities.contract_version))
    if capabilities.batch_contract_version != REQUIRED_BATCH_CONTRACT_VERSION:
        errors.append("batch_contract_version={}".format(capabilities.batch_contract_version))
    if capabilities.action_contract_version != REQUIRED_ACTION_CONTRACT_VERSION:
        errors.append("action_contract_version={}".format(capabilities.action_contract_version))
    for name, path in REQUIRED_ENDPOINTS.items():
        actual = getattr(capabilities.endpoints, name)
        if actual != path:
            errors.append("endpoints.{}={}".format(name, actual))
    for name, path in OPTIONAL_ENDPOINTS.items():
        actual = getattr(capabilities.endpoints, name, None)
        if actual is not None and actual != path:
            errors.append("endpoints.{}={}".format(name, actual))
    missing_types = sorted(REQUIRED_RESOLVE_TYPES - set(capabilities.resolve_types))
    if missing_types:
        errors.append("missing resolve_types={}".format(",".join(missing_types)))
    missing_statuses = sorted(REQUIRED_STATUSES - set(capabilities.statuses))
    if missing_statuses:
        errors.append("missing statuses={}".format(",".join(missing_statuses)))
    if capabilities.max_batch_requests < REQUIRED_MIN_BATCH_REQUESTS:
        errors.append("max_batch_requests={}".format(capabilities.max_batch_requests))
    if capabilities.max_request_body_bytes <= 0:
        errors.append("max_request_body_bytes={}".format(capabilities.max_request_body_bytes))
    return errors
