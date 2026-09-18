from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from typing import Literal, NoReturn

import anyio
import pytest
from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.identity import (
    ConversationStateKey,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.audit_records import (
    AuditRecordV1,
    DirectAuditRecordV1,
    GroupAuditRecordV1,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse
from market_support_crewai_agent.server import auth
from market_support_crewai_agent.server import main as server_main
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_envelope, make_v2_payload
from tests.integration.runtime._todo14_direct_provider_fixtures import (
    install_direct_provider_http_fake,
)
from tests.integration.runtime._todo14_reachability_fixtures import (
    assert_reachability,
    install_registry_spy,
    install_runtime_settings,
    runtime_with_provider_fakes,
)


@dataclass(frozen=True, slots=True)
class _ForbiddenPlannerFactory:
    calls: list[str]

    def build_planner_agent(self) -> NoReturn:
        self.calls.append("planner")
        raise AssertionError("direct_scene_crewai_forbidden")


@dataclass(frozen=True, slots=True)
class _ForbiddenComposerFactory:
    calls: list[str]

    def build_composer_agent(
        self,
        stage: Literal["knowledge_composer", "smalltalk_composer"],
    ) -> NoReturn:
        self.calls.append(stage)
        raise AssertionError("direct_scene_crewai_forbidden")


@dataclass(frozen=True, slots=True)
class _ForbiddenAlignmentFactory:
    calls: list[str]

    def build_alignment_verifier_agent(self) -> NoReturn:
        self.calls.append("alignment")
        raise AssertionError("direct_scene_crewai_forbidden")


def _audit_record(
    coordinator: ReplyStateTransactionCoordinatorV1,
    state_key: ConversationStateKey,
    response_id: str,
) -> AuditRecordV1:
    journal = coordinator.last_journal()
    assert journal is not None
    return journal.candidate_root.audit_records[(state_key, response_id)]


def test_group_reply_uses_v2_programs_one_shared_journal_and_committed_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the real HTTP route and lifecycle with provider-bound planner/composer/verifier fakes.
    settings = Settings(
        api_key="secret",
        deployment_tenant_ref="tenant:test",
        llm_api_key="test-key",
        group_recall_mode="off",
        planner_transient_retry_attempts=0,
        reply_alignment_verifier_enabled=True,
        reply_alignment_max_replans=2,
        reply_alignment_max_evidence_refetches=2,
        reply_alignment_max_recomposes=2,
        reply_alignment_max_total_remediations=2,
    )
    payload = make_v2_payload("你好")
    envelope = make_v2_envelope("你好")
    events: list[str] = []
    journals: list[TurnLlmInvocationJournalV1] = []
    install_registry_spy(monkeypatch, events)
    coordinator = ReplyStateTransactionCoordinatorV1()
    runtime = runtime_with_provider_fakes(
        monkeypatch=monkeypatch,
        settings=settings,
        envelope=envelope,
        events=events,
        journals=journals,
    )
    runtime.coordinator = coordinator

    async def build_reply(request: VerifiedRequestEnvelopeV1):
        assert request == envelope
        return await runtime.reply(request)

    monkeypatch.setattr(server_main, "get_settings", lambda: settings)
    monkeypatch.setattr(auth, "get_settings", lambda: settings)
    monkeypatch.setattr(server_main, "build_reply", build_reply)

    # When: POST /reply runs through the actual route, lifecycle, dispatch, and commit path.
    response = TestClient(server_main.app).post(
        "/reply",
        json=payload,
        headers={"X-API-Key": "secret"},
    )

    # Then: packaged V2 rows resolve before all three dispatches and reach group audit.
    assert response.status_code == 200
    reply = ReplyResponse.model_validate_json(response.content)
    assert reply.reply.text == "你好，我在。"
    audit = _audit_record(
        coordinator,
        envelope.state_key,
        reply.response_id,
    )
    assert isinstance(audit, GroupAuditRecordV1)
    assert_reachability(
        events=events,
        journals=journals,
        scene="group",
        record=audit,
    )
    required_fields = {
        "program_version",
        "scene_key",
        "scene_contract_ref",
        "model_family",
        "provider_id",
        "transport_id",
        "input_schema_version",
        "output_schema_version",
        "osh1",
        "poh1",
        "hph1",
        "prh1",
        "input_digest",
        "output_digest",
    }
    journal_rows = journals[0].rows
    assert journals[0].max_rows == settings.maximum_llm_invocation_rows
    for journal_row, persisted_row in zip(
        journal_rows,
        audit.programs,
        strict=True,
    ):
        persisted = asdict(persisted_row)
        journal_projection = {
            field: getattr(journal_row, field) for field in required_fields
        }
        assert required_fields <= persisted.keys()
        assert {field: persisted[field] for field in required_fields} == (
            journal_projection
        )


def test_direct_reply_lifecycle_projects_the_same_journal_into_redacted_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the direct lifecycle seam (the public direct gate remains Todo15-owned).
    audit_key = base64.urlsafe_b64encode(b"d" * 32).decode("ascii").rstrip("=")
    settings = Settings(
        llm_api_key="test-key",
        direct_audit_hmac_key=audit_key,
        group_recall_mode="off",
        planner_transient_retry_attempts=0,
        reply_alignment_verifier_enabled=True,
    )
    envelope = make_v2_envelope(
        "你好",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:todo14-direct",
            "direct_thread_ref": "direct:todo14-thread",
            "principal_ref": "principal:todo14",
        },
        presentation={"contract_version": "direct-presentation.v1"},
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    )
    events: list[str] = []
    journals: list[TurnLlmInvocationJournalV1] = []
    requests: list[str] = []
    install_registry_spy(monkeypatch, events)
    install_runtime_settings(monkeypatch, settings)
    install_direct_provider_http_fake(
        monkeypatch,
        envelope=envelope,
        requests=requests,
        events=events,
        journals=journals,
    )
    coordinator = ReplyStateTransactionCoordinatorV1()
    runtime = CrewAIReplyRuntime(
        settings,
        coordinator=coordinator,
    )
    crewai_calls: list[str] = []
    runtime.planner_agent_factory = _ForbiddenPlannerFactory(crewai_calls)
    runtime.composer_agent_factory = _ForbiddenComposerFactory(crewai_calls)
    runtime.alignment_agent_factory = _ForbiddenAlignmentFactory(crewai_calls)

    # When: the direct shared-kernel lifecycle dispatches and commits.
    response = anyio.run(runtime.reply, envelope)

    # Then: the same gap-free journal reaches the redacted direct audit record.
    audit = _audit_record(coordinator, envelope.state_key, response.response_id)
    assert isinstance(audit, DirectAuditRecordV1)
    assert_reachability(
        events=events,
        journals=journals,
        scene="direct",
        record=audit,
    )
    assert journals[0].max_rows == settings.maximum_llm_invocation_rows
    assert audit.redaction_flags
    assert "principal:todo14" not in repr(audit)
    assert crewai_calls == []
    assert len(requests) == 3
    assert all("principal:todo14" not in request for request in requests)
    assert all(
        program.transport_id == "openai_chat_completions" for program in audit.programs
    )
