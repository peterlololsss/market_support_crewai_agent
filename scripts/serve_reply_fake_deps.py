from __future__ import annotations

import base64
from _thread import LockType
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum, unique
from hmac import compare_digest
from ipaddress import ip_address
from threading import Lock, Thread
from time import monotonic, sleep
from types import SimpleNamespace
from typing import Annotated, ClassVar, Final, Literal, Never, Protocol, assert_never

import typer
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
from typing_extensions import override

from market_support_crewai_agent.runtime.context.stage_inputs import (
    KnowledgeComposerPromptInputV1,
    PlannerPromptInputV1,
    SmalltalkComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge import (
    InternalCompanyKnowledgeGatewayV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayDocumentContextV1,
    GatewayStaticContextV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
    kernel_channel_type,
)
from market_support_crewai_agent.runtime.integrations.adapter.client import (
    AdapterResolveClient,
)
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightService,
)
from market_support_crewai_agent.runtime.integrations.document_mcp.cache import (
    DocumentMcpCacheAuthorityV1,
)
from market_support_crewai_agent.runtime.planning import PlanSpec
from market_support_crewai_agent.runtime.planning.plan_spec import (
    AnswerabilityPolicy,
    DistributionPlanDomainScopeV2,
    PlanSpecEvidenceContractV1,
    PlanStep,
    PlanUnit,
    UnscopedPlanDomainScopeV2,
)
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
)
from market_support_crewai_agent.runtime.policy.capabilities.definitions import (
    CapabilityManifestIdV2,
)
from market_support_crewai_agent.runtime.policy.capabilities.evidence_vocabulary import (
    EvidenceArtifactTypeV2,
)
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.coordinator_state import (
    EMPTY_COORDINATOR_ROOT,
    CoordinatorStateRootV1,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.schemas.adapter import (
    AdapterResolveBatchRequest,
    AdapterResolveBatchResult,
    AdapterResolveRequest,
    AdapterResolveResult,
    AvailableArtifact,
)
from market_support_crewai_agent.schemas.adapter_metadata import (
    AdapterCapabilities,
    AdapterCapabilityAuth,
    AdapterCapabilityEndpoints,
)
from market_support_crewai_agent.schemas.base import StrictModel
from market_support_crewai_agent.schemas.conversation import (
    validate_canonical_tenant_ref,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from market_support_crewai_agent.server import (
    adapter_compatibility,
    auth,
    lifespan,
)
from market_support_crewai_agent.server import main as production
from market_support_crewai_agent.settings_model import Settings


@unique
class StartupMode(StrEnum):
    COMPATIBLE = "compatible"
    TENANT_UNCONFIGURED = "tenant-unconfigured"
    DM_DISABLED = "dm-disabled"
    ADAPTER_MISSING_FIELDS = "adapter-missing-fields"
    ADAPTER_WRONG_SCENE = "adapter-wrong-scene"
    ADAPTER_WRONG_TENANT = "adapter-wrong-tenant"
    ADAPTER_WRONG_VERSION = "adapter-wrong-version"


@unique
class BooleanArg(StrEnum):
    TRUE = "true"
    FALSE = "false"


@dataclass(frozen=True, slots=True)
class ServeConfigError(ValueError):
    message: str

    @override
    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class ServeHarnessError(RuntimeError):
    message: str

    @override
    def __str__(self) -> str:
        return self.message


@unique
class Scenario(StrEnum):
    DIRECT_KNOWLEDGE = "direct-knowledge"
    GROUP_SAFE = "group-safe"
    GROUP_ACTION = "group-action"


class ServeConfig(StrictModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    host: str
    port: int = Field(ge=1024, le=65535)
    adapter_port: int = Field(ge=1024, le=65535)
    api_key: str = Field(min_length=1)
    adapter_api_key: str = Field(min_length=1)
    tenant_ref: str
    internal_dm_enabled: bool
    mode: StartupMode

    @field_validator("host")
    @classmethod
    def loopback_only(cls, value: str) -> str:
        if not ip_address(value).is_loopback:
            raise ServeConfigError("host must be a loopback IP address")
        return value

    @field_validator("api_key", "adapter_api_key")
    @classmethod
    def nonblank_key(cls, value: str) -> str:
        if not value.strip():
            raise ServeConfigError("fixture keys must be nonblank")
        return value

    @field_validator("tenant_ref")
    @classmethod
    def canonical_tenant(cls, value: str) -> str:
        return validate_canonical_tenant_ref(value)

    @model_validator(mode="after")
    def distinct_ports(self) -> ServeConfig:
        if self.port == self.adapter_port:
            raise ServeConfigError("agent and adapter ports must differ")
        return self


_SCENARIOS: Final = {
    "req:http-direct-knowledge": Scenario.DIRECT_KNOWLEDGE,
    "req:http-direct-injection": Scenario.DIRECT_KNOWLEDGE,
    "req:http-direct-issued": Scenario.DIRECT_KNOWLEDGE,
    "req:http-group-safe": Scenario.GROUP_SAFE,
    "req:http-group-action": Scenario.GROUP_ACTION,
    "req:http-isolation-alice-seed": Scenario.GROUP_SAFE,
    "req:http-isolation-alice-followup": Scenario.GROUP_SAFE,
    "req:http-isolation-bob-first": Scenario.GROUP_SAFE,
}
_CURRENT_ENVELOPE: ContextVar[VerifiedRequestEnvelopeV1 | None] = ContextVar(
    "scene_http_envelope", default=None
)
_PLAN_FIXTURES: Final[
    dict[Scenario, tuple[CapabilityManifestIdV2, AnswerabilityPolicy, str | None]]
] = {
    Scenario.DIRECT_KNOWLEDGE: (
        "answer_internal_company_knowledge",
        "answer",
        "company profile",
    ),
    Scenario.GROUP_SAFE: ("general.smalltalk", "smalltalk", None),
    Scenario.GROUP_ACTION: ("weekly_report.send", "send", None),
}
_ALL_EVIDENCE_ARTIFACT_TYPES: Final[tuple[EvidenceArtifactTypeV2, ...]] = (
    "material_pack",
    "weekly_report",
    "monthly_report",
    "document_context",
    "adapter_context",
    "history",
    "user_upload",
    "unknown",
)


class CounterStore:
    def __init__(self) -> None:
        self._lock: LockType = Lock()
        self._values: dict[str, int] = {
            "planner_count": 0,
            "composer_count": 0,
            "knowledge_count": 0,
            "adapter_capabilities_count": 0,
            "adapter_resolve_count": 0,
            "send_spy_count": 0,
        }
        self._history_counts: dict[str, int] = {}

    def increment(self, name: str) -> None:
        with self._lock:
            self._values[name] += 1

    def set_history_count(self, request_id: str, count: int) -> None:
        with self._lock:
            self._history_counts[request_id] = count

    def history_count(self, request_id: str) -> int:
        with self._lock:
            return self._history_counts.get(request_id, 0)

    def snapshot(
        self, coordinator: ReplyStateTransactionCoordinatorV1
    ) -> dict[str, int]:
        with self._lock:
            values = dict(self._values)
        root = _published_root(coordinator)
        values.update(
            root_revision=coordinator.root_revision(),
            issued_count=len(root.issued_records),
            state_key_count=len(root.state_revisions),
            feedback_count=len(root.feedback_receipts),
        )
        return values


def _published_root(
    coordinator: ReplyStateTransactionCoordinatorV1,
) -> CoordinatorStateRootV1:
    journal = coordinator.last_journal()
    if journal is None:
        return EMPTY_COORDINATOR_ROOT
    if journal.outcome == "committed":
        return journal.candidate_root
    return journal.prior_root


class PlannerAgentLike(Protocol):
    async def kickoff_async(self, prompt: str, response_format: type[StrictModel]): ...


class FakeDocumentProvider:
    def __init__(self, counters: CounterStore) -> None:
        self._counters: CounterStore = counters

    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
        cache_authority: DocumentMcpCacheAuthorityV1 | None,
    ) -> tuple[GatewayDocumentContextV1, ...]:
        del request, evidence_query, cache_authority
        self._counters.increment("knowledge_count")
        return (
            GatewayDocumentContextV1(
                document_id="fixture-company-profile",
                title="fixture company profile",
                text="本企业内部市场支持服务已启用。",
            ),
        )


class EmptyStaticProvider:
    async def collect(
        self,
        *,
        request: KernelReplyRequestV1,
        evidence_query: str,
    ) -> tuple[GatewayStaticContextV1, ...]:
        del request, evidence_query
        return ()


def _composer_input_variant(
    input_value: ComposerPromptInputV1,
) -> ComposerPromptInputV1 | None:
    return input_value


def _startup_mode_variant(mode: StartupMode) -> StartupMode | None:
    return mode


def _reject_fixture_variant(value: None, label: str) -> Never:
    del value
    raise ServeHarnessError(f"fixture {label} variant is unsupported")


class FakeComposer:
    def __init__(self, counters: CounterStore) -> None:
        self._counters: CounterStore = counters

    async def compose(self, input_value: ComposerPromptInputV1) -> ComposerReplyOutput:
        self._counters.increment("composer_count")
        envelope = _required_envelope()
        match _composer_input_variant(input_value):
            case KnowledgeComposerPromptInputV1():
                text = "内部资料答复：本企业内部市场支持服务已启用。"
            case SmalltalkComposerPromptInputV1():
                history_count = self._counters.history_count(
                    envelope.request.request_id
                )
                text = f"安全答复，历史条数={history_count}。"
            case _ as unreachable:
                assert_never(_reject_fixture_variant(unreachable, "composer input"))
        return ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(kind="answer", text=text, mentions=[]),
        )


def _required_envelope() -> VerifiedRequestEnvelopeV1:
    envelope = _CURRENT_ENVELOPE.get()
    if envelope is None:
        raise ServeHarnessError("fixture request context is unavailable")
    return envelope


def _plan_for(envelope: VerifiedRequestEnvelopeV1) -> PlanSpec:
    scenario = _SCENARIOS.get(envelope.request.request_id)
    if scenario is None:
        raise ServeHarnessError("fixture request id is not registered")
    capability_id, answerability, query = _PLAN_FIXTURES[scenario]
    manifest = CAPABILITY_MANIFEST_REGISTRY.find(capability_id)
    if manifest is None:
        raise ServeHarnessError("fixture capability is unavailable")
    scope = business_scope_authority_v1(envelope.request.business_scope)
    if envelope.request.identity.scene == "direct":
        domain_scope = UnscopedPlanDomainScopeV2()
    else:
        domain_scope = DistributionPlanDomainScopeV2(
            business_scope_ref=scope.business_scope_ref,
            channel_kind=kernel_channel_type(envelope.request),
        )
    evidence = manifest.evidence_contract
    allowed = [str(item) for item in evidence.allowed_artifact_types]
    required = [str(item) for item in evidence.required_artifact_types]
    forbidden = [item for item in _ALL_EVIDENCE_ARTIFACT_TYPES if item not in allowed]
    return PlanSpec(
        plan_id=f"fixture-{scenario.value}",
        user_intent_summary=f"fixture {scenario.value}",
        plan_units=[
            PlanUnit(
                unit_id="fixture-unit",
                selected_capability_id=manifest.manifest_id,
                domain_scope=domain_scope,
                required_artifacts=required,
                allowed_artifacts=allowed,
                forbidden_artifacts=forbidden,
                required_tools=[],
                answerability_policy=answerability,
                output_schema_ref=f"{manifest.manifest_id}:output_schema",
                evidence_contract_ref=f"{manifest.manifest_id}:evidence_contract",
                evidence_contract=PlanSpecEvidenceContractV1(
                    required_fact_types=evidence.required_fact_types,
                    any_of_fact_types=evidence.any_of_fact_types,
                    allowed_source_types=evidence.allowed_source_types,
                    forbidden_source_types=evidence.forbidden_source_types,
                    min_facts=evidence.min_facts,
                ),
                steps=[
                    PlanStep(
                        step_id="fixture-step",
                        description=f"fixture {scenario.value}",
                        uses_artifacts=required,
                        required_artifacts=required,
                        allowed_artifacts=allowed,
                        forbidden_artifacts=forbidden,
                        required_tools=[],
                        evidence_query=query,
                    )
                ],
                acceptance_criteria=["satisfy fixture capability"],
                abstention_cases=[],
                risk_flags=[],
            )
        ],
        risk_flags=[],
    )


def _adapter_result(request: AdapterResolveRequest) -> AdapterResolveResult:
    return AdapterResolveResult(
        contract_version="adapter-resolve",
        resolve_type=request.resolve_type,
        status="resolved",
        display_name=request.dist_name,
        reason_code="ok",
        candidates=[],
        channel_type="non_bank",
        available_artifacts=[AvailableArtifact(type="weekly_report")],
        resolved_at=1,
        resolve_ref=f"{request.resolve_type}:fixture-ref",
        period="2026-W29" if request.resolve_type == "weekly_report" else None,
        report_date="2026-07-17" if request.resolve_type == "weekly_report" else None,
    )


def _capabilities(config: ServeConfig) -> AdapterCapabilities:
    scenes: list[Literal["direct", "group"]] | None = ["direct", "group"]
    reply_versions: list[str] | None = ["reply-request.v2"]
    feedback_versions: list[str] | None = ["action-feedback.v2"]
    identity_versions: list[str] | None = ["conversation-identity.v1"]
    tenant_ref: str | None = config.tenant_ref
    match _startup_mode_variant(config.mode):
        case StartupMode.ADAPTER_MISSING_FIELDS:
            scenes = reply_versions = feedback_versions = identity_versions = None
            tenant_ref = None
        case StartupMode.ADAPTER_WRONG_SCENE:
            scenes = ["group"]
        case StartupMode.ADAPTER_WRONG_TENANT:
            tenant_ref = "tenant:wrong"
        case StartupMode.ADAPTER_WRONG_VERSION:
            reply_versions = ["reply-request.v3"]
        case (
            StartupMode.COMPATIBLE
            | StartupMode.TENANT_UNCONFIGURED
            | StartupMode.DM_DISABLED
        ):
            pass
        case _ as unreachable:
            assert_never(_reject_fixture_variant(unreachable, "startup mode"))
    return AdapterCapabilities(
        service="xiaoyan-wecom-market-agent-adapter",
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
        statuses=[
            "resolved",
            "missing",
            "ambiguous",
            "forbidden",
            "temporarily_unavailable",
        ],
        max_batch_requests=16,
        max_request_body_bytes=65_536,
        auth=AdapterCapabilityAuth(
            header_schemes=["Bearer"],
            protected_endpoints=["/adapter/capabilities", "/adapter/resolve/batch"],
        ),
        supported_scenes=scenes,
        reply_request_contract_versions=reply_versions,
        action_feedback_contract_versions=feedback_versions,
        conversation_identity_contract_versions=identity_versions,
        deployment_tenant_ref=tenant_ref,
    )


def _build_adapter_app(config: ServeConfig, counters: CounterStore) -> FastAPI:
    app = FastAPI()

    def authorize(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = f"Bearer {config.adapter_api_key}"
        if authorization is None or not compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.middleware("http")
    async def send_spy(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        safe_posts = {"/adapter/resolve", "/adapter/resolve/batch"}
        if request.method != "GET" and request.url.path not in safe_posts:
            counters.increment("send_spy_count")
            raise ServeHarnessError("fake adapter execution is forbidden")
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/adapter/capabilities", response_model=AdapterCapabilities)
    async def capabilities(
        _auth: Annotated[None, Depends(authorize)] = None,
    ) -> AdapterCapabilities:
        counters.increment("adapter_capabilities_count")
        return _capabilities(config)

    @app.post("/adapter/resolve", response_model=AdapterResolveResult)
    async def resolve(
        request: AdapterResolveRequest,
        _auth: Annotated[None, Depends(authorize)] = None,
    ) -> AdapterResolveResult:
        counters.increment("adapter_resolve_count")
        return _adapter_result(request)

    @app.post("/adapter/resolve/batch", response_model=AdapterResolveBatchResult)
    async def resolve_batch(
        request: AdapterResolveBatchRequest,
        _auth: Annotated[None, Depends(authorize)] = None,
    ) -> AdapterResolveBatchResult:
        counters.increment("adapter_resolve_count")
        return AdapterResolveBatchResult(
            contract_version="adapter-resolve-batch",
            results=[_adapter_result(item) for item in request.requests],
        )

    _ = (send_spy, health, capabilities, resolve, resolve_batch)
    return app


def _settings(config: ServeConfig) -> Settings:
    audit_key = base64.urlsafe_b64encode(b"q" * 32).decode("ascii").rstrip("=")
    return Settings(
        api_key=config.api_key,
        deployment_tenant_ref=(
            None
            if config.mode is StartupMode.TENANT_UNCONFIGURED
            else config.tenant_ref
        ),
        internal_dm_enabled=(
            False
            if config.mode is StartupMode.DM_DISABLED
            else config.internal_dm_enabled
        ),
        llm_api_key="fixture-llm-key",
        planner_llm_api_key="fixture-planner-key",
        direct_audit_hmac_key=audit_key,
        adapter_base_url=f"http://{config.host}:{config.adapter_port}",
        adapter_api_key=config.adapter_api_key,
        adapter_timeout_seconds=2,
        group_recall_mode="off",
        doc_mcp_enabled=False,
        doc_mcp_cache_ttl_seconds=0,
        reply_alignment_verifier_enabled=False,
    )


def _install_production_app(
    config: ServeConfig,
    counters: CounterStore,
    coordinator: ReplyStateTransactionCoordinatorV1,
) -> None:
    settings = _settings(config)
    preflight = AdapterPreflightService(AdapterResolveClient(settings))
    runtime = CrewAIReplyRuntime(
        settings,
        preflight_service=preflight,
        coordinator=coordinator,
        internal_company_knowledge_gateway=InternalCompanyKnowledgeGatewayV1(
            document_provider=FakeDocumentProvider(counters),
            static_provider=EmptyStaticProvider(),
        ),
        v2_composer=FakeComposer(counters),
    )

    async def fake_planner(
        planner_agent: PlannerAgentLike | None,
        planner_program: PromptProgram,
        *,
        planner_input: PlannerPromptInputV1,
        timeout_seconds: float | None,
        retry_attempts: int,
        base_delay_seconds: float,
        journal: TurnLlmInvocationJournalV1 | None = None,
        settings: Settings | None = None,
    ) -> tuple[SimpleNamespace, list[dict[str, JsonValue]]]:
        del (
            planner_agent,
            planner_program,
            timeout_seconds,
            retry_attempts,
            base_delay_seconds,
            journal,
            settings,
        )
        envelope = _required_envelope()
        counters.increment("planner_count")
        counters.set_history_count(
            envelope.request.request_id, len(planner_input.history)
        )
        plan = _plan_for(envelope)
        return SimpleNamespace(pydantic=plan, raw=plan.model_dump_json()), []

    async def injected_build_reply(
        envelope: VerifiedRequestEnvelopeV1,
    ) -> ReplyResponse:
        token = _CURRENT_ENVELOPE.set(envelope)
        try:
            return await runtime.reply(envelope)
        finally:
            _CURRENT_ENVELOPE.reset(token)

    async def qa_headers(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        for name, value in counters.snapshot(coordinator).items():
            response.headers[f"X-QA-{name.replace('_', '-')}"] = str(value)
        return response

    from market_support_crewai_agent.runtime import planning_flow

    planning_flow.__dict__["run_planner_kickoff_with_retry"] = fake_planner
    production.__dict__["get_settings"] = lambda: settings
    production.__dict__["get_reply_state_coordinator"] = lambda: coordinator
    production.__dict__["build_reply"] = injected_build_reply
    auth.__dict__["get_settings"] = lambda: settings
    lifespan.__dict__["get_settings"] = lambda: settings
    lifespan.__dict__["get_reply_state_coordinator"] = lambda: coordinator
    adapter_compatibility.__dict__["get_settings"] = lambda: settings
    adapter_compatibility.clear_compatibility_client_cache_for_testing()
    _ = production.app.middleware("http")(qa_headers)


def _run_server(config: ServeConfig) -> None:
    counters = CounterStore()
    coordinator = ReplyStateTransactionCoordinatorV1()
    adapter_server = uvicorn.Server(
        uvicorn.Config(
            _build_adapter_app(config, counters),
            host=config.host,
            port=config.adapter_port,
            log_level="warning",
        )
    )
    adapter_thread = Thread(target=adapter_server.run, name="fixture-adapter")
    adapter_thread.start()
    deadline = monotonic() + 10
    while not adapter_server.started and adapter_thread.is_alive():
        if monotonic() >= deadline:
            break
        sleep(0.01)
    if not adapter_server.started:
        adapter_server.should_exit = True
        adapter_thread.join(timeout=5)
        raise ServeHarnessError("fake adapter failed to bind")
    _install_production_app(config, counters, coordinator)
    agent_server = uvicorn.Server(
        uvicorn.Config(
            production.app,
            host=config.host,
            port=config.port,
            log_level="warning",
        )
    )
    try:
        agent_server.run()
    finally:
        adapter_server.should_exit = True
        adapter_thread.join(timeout=10)
        adapter_compatibility.clear_compatibility_client_cache_for_testing()
    if adapter_thread.is_alive():
        raise ServeHarnessError("fake adapter did not stop")


def main(
    host: Annotated[str, typer.Option()],
    port: Annotated[int, typer.Option()],
    adapter_port: Annotated[int, typer.Option()],
    api_key: Annotated[str, typer.Option()],
    adapter_api_key: Annotated[str, typer.Option()],
    tenant_ref: Annotated[str, typer.Option()],
    internal_dm_enabled: Annotated[BooleanArg, typer.Option()],
    mode: Annotated[StartupMode, typer.Option()],
) -> None:
    config = ServeConfig(
        host=host,
        port=port,
        adapter_port=adapter_port,
        api_key=api_key,
        adapter_api_key=adapter_api_key,
        tenant_ref=tenant_ref,
        internal_dm_enabled=internal_dm_enabled is BooleanArg.TRUE,
        mode=mode,
    )
    _run_server(config)


if __name__ == "__main__":
    typer.run(main)
