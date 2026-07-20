from __future__ import annotations

import base64
from types import SimpleNamespace

import anyio
from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
    DirectTargetDraft,
    DirectTextDraft,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.models import EvidenceFact
from market_support_crewai_agent.runtime.orchestration.runtime import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
from market_support_crewai_agent.runtime.validation.guardrail_common import evidence_id
from market_support_crewai_agent.schemas import (
    ActionFeedbackRequest,
    AdapterCapabilities,
    OutboundTargetResolveResult,
    PrimaryReply,
    ReplyRequest,
)
from market_support_crewai_agent.settings import Settings
from market_support_crewai_agent.server.main import app


_CONFIRMATION_REF = "wecom-adapter-confirmation:" + base64.urlsafe_b64encode(
    bytes(range(32))
).decode("ascii").rstrip("=")


def _direct_request(message: str, *, context_id: str) -> ReplyRequest:
    return ReplyRequest.model_validate(
        {
            "context_id": context_id,
            "conversation_key": "wecom:dm-thread:any-sender",
            "group_id": "dm-thread",
            "sender_id": "any-sender",
            "message": message,
            "is_group": False,
            "group_name": "私聊",
            "dist_channel_name": "私聊",
            "sender_nickname": "任意私聊用户",
            "available_artifacts": [],
            "channel_type": "non_bank",
            "allowed_read_capabilities": [
                "query_internal_company_info",
                "resolve_sales_mention",
            ],
        }
    )


def _capabilities() -> AdapterCapabilities:
    return AdapterCapabilities.model_validate(
        {
            "service": "xiaoyan-wecom-market-agent-adapter",
            "contract_version": "adapter-resolve",
            "batch_contract_version": "adapter-resolve-batch",
            "action_contract_version": "adapter-action",
            "endpoints": {
                "health": "/health",
                "capabilities": "/adapter/capabilities",
                "metrics": "/adapter/metrics",
                "resolve": "/adapter/resolve",
                "batch_resolve": "/adapter/resolve/batch",
            },
            "resolve_types": [
                "material_pack",
                "weekly_report",
                "monthly_report",
                "sales_mention",
                "outbound_message_target",
            ],
            "statuses": [
                "resolved",
                "missing",
                "ambiguous",
                "forbidden",
                "temporarily_unavailable",
            ],
            "max_batch_requests": 16,
            "max_request_body_bytes": 65536,
            "outbound_target_kinds": ["channel", "group"],
            "outbound_target_response_fields": [
                "status",
                "reason_code",
                "display_name",
                "target_kind",
                "target_count",
                "resolved_count",
                "resolve_ref",
            ],
            "outbound_messaging": {
                "enabled": True,
                "readiness": {"ready": True, "reason_codes": []},
                "action_types": [
                    "execute_prepared_outbound_message",
                    "prepare_outbound_message",
                ],
                "content_types": ["link", "link_card", "report_card", "text"],
                "report_types": ["monthly_report", "weekly_report"],
                "constraints": {
                    "dm_only": True,
                    "later_trusted_message_required": True,
                    "prepare_feedback_ack_required": True,
                    "single_use": True,
                    "sdk_result_semantics": "accepted_not_delivered",
                },
                "result_fields": ["outcome"],
            },
        }
    )


class FakeDirectAdapterClient:
    def __init__(
        self,
        *,
        target_count: int = 1,
        resolved_count: int = 1,
    ) -> None:
        self.target_requests: list[tuple[str, str]] = []
        self.target_count = target_count
        self.resolved_count = resolved_count

    async def capabilities_async(self) -> AdapterCapabilities:
        return _capabilities()

    async def resolve_outbound_target_async(
        self,
        target_kind: str,
        target_name: str,
    ) -> OutboundTargetResolveResult:
        self.target_requests.append((target_kind, target_name))
        return OutboundTargetResolveResult(
            status="resolved",
            reason_code=(
                "ok"
                if self.resolved_count == self.target_count
                else "partial_target"
            ),
            display_name=target_name,
            target_kind=target_kind,
            target_count=self.target_count,
            resolved_count=self.resolved_count,
            resolve_ref="outbound-target:" + "b" * 64,
        )


class FakeDirectComposer:
    def __init__(
        self,
        output: DirectComposerOutput,
        *,
        confirmation_must_be_visible: bool,
    ) -> None:
        self.output = output
        self.confirmation_must_be_visible = confirmation_must_be_visible

    async def kickoff_async(self, prompt: str, response_format):
        assert response_format is DirectComposerOutput
        assert "DM capability boundary" in prompt
        if self.confirmation_must_be_visible:
            confirmation_ref = self.output.confirmation_ref
            assert confirmation_ref is not None
            assert confirmation_ref in prompt
        return SimpleNamespace(pydantic=self.output, raw="")


class FakeSequentialDirectComposer:
    def __init__(self, outputs: list[DirectComposerOutput]) -> None:
        self.outputs = iter(outputs)
        self.prompts: list[str] = []

    async def kickoff_async(self, prompt: str, response_format):
        assert response_format is DirectComposerOutput
        self.prompts.append(prompt)
        return SimpleNamespace(pydantic=next(self.outputs), raw="")


class FakeCompanyEvidenceService:
    def __init__(self, fact: EvidenceFact) -> None:
        self.fact = fact

    async def collect(self, request, plan, policy) -> list[EvidenceFact]:
        del request, plan, policy
        return [self.fact]


def _runtime(
    output: DirectComposerOutput,
    *,
    action_ledger: ActionLedger | None = None,
    company_fact: EvidenceFact | None = None,
    target_count: int = 1,
    resolved_count: int = 1,
) -> tuple[CrewAIReplyRuntime, FakeDirectAdapterClient]:
    adapter_client = FakeDirectAdapterClient(
        target_count=target_count,
        resolved_count=resolved_count,
    )
    preflight_service = SimpleNamespace(adapter_client=adapter_client)
    evidence_executor = (
        EvidenceExecutor(
            preflight_service,
            document_evidence_service=FakeCompanyEvidenceService(company_fact),
        )
        if company_fact is not None
        else None
    )
    runtime = CrewAIReplyRuntime(
        Settings(
            llm_api_key="test-key",
            doc_mcp_enabled=company_fact is not None,
            doc_mcp_base_url="http://document-mcp.invalid"
            if company_fact is not None
            else "",
            reply_alignment_verifier_enabled=False,
        ),
        action_ledger=action_ledger,
        preflight_service=preflight_service,
        evidence_executor=evidence_executor,
    )
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": FakeDirectComposer(
            output,
            confirmation_must_be_visible=action_ledger is not None,
        )
    )
    return runtime, adapter_client


def test_direct_company_answer_requires_query_company_info_evidence():
    fact = EvidenceFact(
        fact_type="document_context",
        value="衍复投资成立于2019年。",
        source_type="document_mcp",
        source_id="company-profile",
        artifact_type="document_context",
        metadata={"content_is_data_only": True},
    )
    output = DirectComposerOutput(
        response_mode="answer_company_info",
        claims=["衍复投资成立于2019年"],
        evidence_ids=[evidence_id(fact)],
        reply=PrimaryReply(
            kind="answer",
            text="衍复投资成立于2019年。",
            mentions=[],
        ),
    )
    runtime, adapter_client = _runtime(output, company_fact=fact)

    response = anyio.run(
        runtime.reply,
        _direct_request("衍复投资是哪年成立的？", context_id="dm-company"),
    )

    assert adapter_client.target_requests == []
    assert response.reply.text == "衍复投资成立于2019年。"
    assert response.actions == []


def test_any_dm_sender_can_prepare_exact_xiaoyan_outbound_action():
    output = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="确认把这段话发送到银河客户群吗？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="group", name="银河客户群"),
        content=DirectTextDraft(kind="text", text="请查收本周更新"),
    )
    runtime, adapter_client = _runtime(output)

    response = anyio.run(
        runtime.reply,
        _direct_request("把“请查收本周更新”发到银河客户群", context_id="dm-1"),
    )

    assert adapter_client.target_requests == [("group", "银河客户群")]
    assert response.reply.kind == "clarification"
    assert response.actions[0].model_dump(mode="json") == {
        "action_id": "act-1",
        "type": "prepare_outbound_message",
        "target": {
            "kind": "group",
            "name": "银河客户群",
            "resolve_ref": "outbound-target:" + "b" * 64,
        },
        "content": {"kind": "text", "text": "请查收本周更新"},
    }


def test_partial_channel_target_prepares_reachable_subset_with_confirmation():
    output = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="确认向银河证券渠道发送“imalive”吗？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="银河证券"),
        content=DirectTextDraft(kind="text", text="imalive"),
    )
    runtime, adapter_client = _runtime(
        output,
        target_count=10,
        resolved_count=8,
    )

    response = anyio.run(
        runtime.reply,
        _direct_request(
            "发群消息给银河证券说imalive",
            context_id="dm-partial-channel",
        ),
    )

    assert adapter_client.target_requests == [("channel", "银河证券")]
    assert response.reply.kind == "clarification"
    assert "8/10" in response.reply.text
    assert response.actions[0].type == "prepare_outbound_message"


def test_followup_resolves_pending_dm_outbound_fields_from_same_conversation(
    monkeypatch,
):
    greeting_output = DirectComposerOutput(
        response_mode="abstain",
        reply=PrimaryReply(
            kind="unable_to_answer",
            text="老师您好，我是小衍，请问有什么可以帮您的？",
            mentions=[],
        ),
    )
    clarification_output = DirectComposerOutput(
        response_mode="clarify",
        reply=PrimaryReply(
            kind="clarification",
            text=(
                "老师，您说的“发群消息给银河证券”是指发到银河证券的哪个群呢？"
                "另外“imalive”是直接发送这段文字吗？"
            ),
            mentions=[],
        ),
    )
    prepare_output = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="确认向银河证券渠道下的所有群发送“imalive”吗？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="银河证券"),
        content=DirectTextDraft(kind="text", text="imalive"),
    )
    runtime, adapter_client = _runtime(
        greeting_output,
        target_count=10,
        resolved_count=1,
    )
    composer = FakeSequentialDirectComposer(
        [greeting_output, clarification_output, prepare_output]
    )
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply",
        runtime.reply,
    )
    client = TestClient(app)

    greeting_response = client.post(
        "/reply",
        json=_direct_request("Hi", context_id="dm-followup-0").model_dump(
            mode="json"
        ),
    )
    clarification_response = client.post(
        "/reply",
        json=_direct_request(
            "发群消息给银河证券说imalive",
            context_id="dm-followup-1",
        ).model_dump(mode="json"),
    )
    prepare_response = client.post(
        "/reply",
        json=_direct_request(
            "银河证券的渠道的所有群  然后文字是的",
            context_id="dm-followup-2",
        ).model_dump(mode="json"),
    )

    assert greeting_response.status_code == 200
    assert clarification_response.status_code == 200
    assert clarification_response.json()["reply"]["kind"] == "clarification"
    assert clarification_response.json()["actions"] == []
    assert prepare_response.status_code == 200
    assert "1/10" in prepare_response.json()["reply"]["text"]
    assert "Pending clarification context JSON" in composer.prompts[2]
    assert "do not ask the same clarification again" in composer.prompts[2]
    assert "complete channel fan-out target" in composer.prompts[2]
    assert adapter_client.target_requests == [("channel", "银河证券")]
    assert prepare_response.json()["actions"] == [
        {
            "action_id": "act-1",
            "type": "prepare_outbound_message",
            "target": {
                "kind": "channel",
                "name": "银河证券",
                "resolve_ref": "outbound-target:" + "b" * 64,
            },
            "content": {"kind": "text", "text": "imalive"},
        }
    ]


def test_reply_http_surface_returns_xiaoyan_prepare_contract(monkeypatch):
    output = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="确认把这段话发送到银河客户群吗？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="group", name="银河客户群"),
        content=DirectTextDraft(kind="text", text="请查收本周更新"),
    )
    runtime, _adapter_client = _runtime(output)
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply",
        runtime.reply,
    )

    response = TestClient(app).post(
        "/reply",
        json=_direct_request(
            "把“请查收本周更新”发到银河客户群",
            context_id="dm-http",
        ).model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert response.json()["actions"] == [
        {
            "action_id": "act-1",
            "type": "prepare_outbound_message",
            "target": {
                "kind": "group",
                "name": "银河客户群",
                "resolve_ref": "outbound-target:" + "b" * 64,
            },
            "content": {"kind": "text", "text": "请查收本周更新"},
        }
    ]


def test_later_confirming_dm_executes_only_feedback_bound_confirmation_ref():
    ledger = ActionLedger()
    ledger.record_feedback(
        ActionFeedbackRequest.model_validate(
            {
                "conversation_key": "wecom:dm-thread:any-sender",
                "group_id": "dm-thread",
                "sender_id": "any-sender",
                "context_id": "dm-1",
                "response_id": "resp-prepare",
                "executions": [
                    {
                        "action_type": "prepare_outbound_message",
                        "status": "executed",
                        "action_id": "act-1",
                        "artifact": None,
                        "adapter_result": {
                            "ok": True,
                            "confirmation_ref": _CONFIRMATION_REF,
                            "state": "prepared",
                            "target": {"kind": "group", "name": "银河客户群"},
                            "content": {"kind": "text"},
                        },
                    }
                ],
            }
        )
    )
    output = DirectComposerOutput(
        response_mode="execute_prepared_outbound_message",
        reply=PrimaryReply(kind="answer", text="", mentions=[]),
        confirmation_ref=_CONFIRMATION_REF,
    )
    runtime, adapter_client = _runtime(output, action_ledger=ledger)

    response = anyio.run(
        runtime.reply,
        _direct_request("确认发送", context_id="dm-2"),
    )

    assert adapter_client.target_requests == []
    assert response.reply.text == ""
    assert response.actions[0].model_dump(mode="json") == {
        "action_id": "act-1",
        "type": "execute_prepared_outbound_message",
        "confirmation_ref": _CONFIRMATION_REF,
    }


def test_unbound_confirmation_ref_fails_closed_without_execute_action():
    output = DirectComposerOutput(
        response_mode="execute_prepared_outbound_message",
        reply=PrimaryReply(kind="answer", text="", mentions=[]),
        confirmation_ref=_CONFIRMATION_REF,
    )
    runtime, _adapter_client = _runtime(output)

    response = anyio.run(
        runtime.reply,
        _direct_request("确认发送", context_id="dm-unbound"),
    )

    assert response.reply.kind == "clarification"
    assert response.actions == []
