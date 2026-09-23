from __future__ import annotations

from typing import final

from typing_extensions import override

from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    normalize_reply_request_v2,
)
from market_support_crewai_agent.runtime.integrations.adapter.client import (
    AdapterResolveClient,
)
from market_support_crewai_agent.runtime.integrations.adapter.transport import (
    AdapterClientError,
)
from market_support_crewai_agent.schemas.adapter import (
    AdapterResolveRequest,
    AdapterResolveResult,
    AvailableArtifact,
)
from market_support_crewai_agent.schemas.adapter_metadata import (
    AdapterCapabilities,
    AdapterCapabilityEndpoints,
)
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.type_ids import AdapterResolveType
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_payload


def make_preflight_request(
    *,
    message: str = "hello",
    dist_channel_name: str = "test channel",
    available_artifacts: list[AvailableArtifact] | None = None,
) -> KernelReplyRequestV1:
    artifacts = available_artifacts or [
        AvailableArtifact(type="material_pack", options=[]),
        AvailableArtifact(type="weekly_report"),
        AvailableArtifact(type="monthly_report"),
    ]
    request = ReplyRequestV2.model_validate(
        make_v2_payload(
            message,
            business_scope={
                "kind": "distribution",
                "dist_channel_name": dist_channel_name,
                "channel_type": "bank",
                "available_artifacts": [
                    artifact.model_dump(mode="json") for artifact in artifacts
                ],
            },
        )
    )
    return normalize_reply_request_v2(
        request,
        adapter_namespace="assistant-wecom",
    ).request


@final
class FakeAdapterClient(AdapterResolveClient):
    def __init__(
        self,
        failures: set[AdapterResolveType] | None = None,
        omissions: set[AdapterResolveType] | None = None,
        readiness_error: str = "",
        deployment_tenant_ref: str | None = None,
    ) -> None:
        super().__init__(Settings(llm_api_key="test-key"))
        self.failures = failures or set()
        self.omissions = omissions or set()
        self.readiness_error = readiness_error
        self.ready_calls = 0
        self.requests: list[AdapterResolveRequest] = []
        self.capabilities_value = AdapterCapabilities(
            service="assistant-wecom-market-agent-adapter",
            contract_version="adapter-resolve",
            batch_contract_version="adapter-resolve-batch",
            action_contract_version="adapter-action",
            endpoints=AdapterCapabilityEndpoints(
                health="/health",
                capabilities="/adapter/capabilities",
                metrics="/adapter/metrics",
                resolve="/adapter/resolve",
                batch_resolve="/adapter/resolve/batch",
            ),
            resolve_types=[
                "material_pack",
                "weekly_report",
                "monthly_report",
                "sales_mention",
            ],
            statuses=["resolved"],
            max_batch_requests=4,
            max_request_body_bytes=1024,
            deployment_tenant_ref=deployment_tenant_ref,
        )
        self._ready_capabilities = self.capabilities_value

    @override
    async def assert_ready_async(self) -> AdapterCapabilities:
        self.ready_calls += 1
        if self.readiness_error:
            raise AdapterClientError(self.readiness_error)
        return self.capabilities_value

    @override
    async def resolve_many_async(
        self,
        requests: list[AdapterResolveRequest],
    ) -> list[AdapterResolveResult]:
        self.requests.extend(requests)
        failure = next(
            (
                request.resolve_type
                for request in requests
                if request.resolve_type in self.failures
            ),
            None,
        )
        if failure is not None:
            raise AdapterClientError(f"{failure} unavailable")
        return [
            self._resolve(request)
            for request in requests
            if request.resolve_type not in self.omissions
        ]

    @staticmethod
    def _resolve(request: AdapterResolveRequest) -> AdapterResolveResult:
        return AdapterResolveResult(
            contract_version="adapter-resolve",
            resolve_type=request.resolve_type,
            status="resolved",
            display_name=request.dist_name,
            reason_code="ok",
            candidates=[],
            channel_type="bank",
            available_artifacts=[
                AvailableArtifact(type="material_pack", options=["指增"]),
                AvailableArtifact(type="weekly_report"),
                AvailableArtifact(type="monthly_report"),
            ],
            resolved_at=1,
            resolve_ref=f"{request.resolve_type}:ref",
            material_pack_option=request.material_pack_option,
            period=("20260529" if request.resolve_type == "weekly_report" else None),
        )
