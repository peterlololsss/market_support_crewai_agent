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
from market_support_crewai_agent.runtime.evidence.adapter_client import (
    AdapterClientError,
)
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
_LATEST_CONFIRMATION_REF = "wecom-adapter-confirmation:" + base64.urlsafe_b64encode(
    bytes(reversed(range(32)))
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
        available: bool = True,
    ) -> None:
        self.target_requests: list[tuple[str, str]] = []
        self.target_count = target_count
        self.resolved_count = resolved_count
        self.available: bool = available

    async def capabilities_async(self) -> AdapterCapabilities:
        if not self.available:
            raise AdapterClientError("adapter unavailable")
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
                "ok" if self.resolved_count == self.target_count else "partial_target"
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


class FakeInvalidFollowupDirectComposer:
    def __init__(
        self,
        first_output: DirectComposerOutput,
        recovered_output: DirectComposerOutput,
    ) -> None:
        self.first_output = first_output
        self.recovered_output = recovered_output
        self.calls = 0

    async def kickoff_async(self, prompt: str, response_format):
        del prompt
        assert response_format is DirectComposerOutput
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(pydantic=self.first_output, raw="")
        if self.calls >= 4:
            return SimpleNamespace(pydantic=self.recovered_output, raw="")
        return SimpleNamespace(
            pydantic=DirectComposerOutput.model_validate(
                {
                    "response_mode": "answer_company_info",
                    "reply": {
                        "kind": "clarification",
                        "text": "请确认发送内容。",
                        "mentions": [],
                    },
                }
            ),
            raw="",
        )


class FakeCompanyEvidenceService:
    def __init__(self, fact: EvidenceFact) -> None:
        self.fact = fact
        self.calls = 0

    async def collect(self, request, plan, policy) -> list[EvidenceFact]:
        del request, plan, policy
        self.calls += 1
        return [self.fact]


def _runtime(
    output: DirectComposerOutput,
    *,
    action_ledger: ActionLedger | None = None,
    company_fact: EvidenceFact | None = None,
    target_count: int = 1,
    resolved_count: int = 1,
    adapter_available: bool = True,
) -> tuple[CrewAIReplyRuntime, FakeDirectAdapterClient]:
    adapter_client = FakeDirectAdapterClient(
        target_count=target_count,
        resolved_count=resolved_count,
        available=adapter_available,
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
    knowledge_request = DirectComposerOutput(
        response_mode="request_company_info",
        reply=PrimaryReply(
            kind="unable_to_answer",
            text="我需要查询内部资料后再回答。",
            mentions=[],
        ),
    )
    answer = DirectComposerOutput(
        response_mode="answer_company_info",
        claims=["衍复投资成立于2019年"],
        evidence_ids=[evidence_id(fact)],
        reply=PrimaryReply(
            kind="answer",
            text="衍复投资成立于2019年。",
            mentions=[],
        ),
    )
    runtime, adapter_client = _runtime(answer, company_fact=fact)
    composer = FakeSequentialDirectComposer([knowledge_request, answer])
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )

    response = anyio.run(
        runtime.reply,
        _direct_request("衍复投资是哪年成立的？", context_id="dm-company"),
    )

    assert adapter_client.target_requests == []
    assert response.reply.text == "衍复投资成立于2019年。"
    assert response.actions == []
    evidence_service = runtime.evidence_executor.document_evidence_service
    assert isinstance(evidence_service, FakeCompanyEvidenceService)
    assert evidence_service.calls == 1
    assert len(composer.prompts) == 2
    assert "衍复投资成立于2019年。" not in composer.prompts[0]
    assert "衍复投资成立于2019年。" in composer.prompts[1]


def test_direct_outbound_does_not_preload_company_documents():
    fact = EvidenceFact(
        fact_type="document_context",
        value="与本次外发无关的内部文档。",
        source_type="document_mcp",
        source_id="unrelated-doc",
        artifact_type="document_context",
        metadata={"content_is_data_only": True},
    )
    output = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="确认发送吗？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="兴业证券"),
        content=DirectTextDraft(kind="text", text="公告测试"),
    )
    runtime, _adapter_client = _runtime(output, company_fact=fact)

    response = anyio.run(
        runtime.reply,
        _direct_request(
            "发渠道消息给兴业证券说公告测试",
            context_id="dm-lazy-doc-outbound",
        ),
    )

    assert response.actions[0].type == "prepare_outbound_message"
    evidence_service = runtime.evidence_executor.document_evidence_service
    assert isinstance(evidence_service, FakeCompanyEvidenceService)
    assert evidence_service.calls == 0


def test_direct_generic_abstention_does_not_request_company_documents():
    fact = EvidenceFact(
        fact_type="document_context",
        value="与本次请求无关的内部文档。",
        source_type="document_mcp",
        source_id="unrelated-doc",
        artifact_type="document_context",
        metadata={"content_is_data_only": True},
    )
    output = DirectComposerOutput(
        response_mode="abstain",
        reply=PrimaryReply(
            kind="unable_to_answer",
            text="这个请求不在当前能力范围内。",
            mentions=[],
        ),
    )
    runtime, _adapter_client = _runtime(output, company_fact=fact)
    composer = FakeSequentialDirectComposer([output])
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )

    response = anyio.run(
        runtime.reply,
        _direct_request("帮我查询天气", context_id="dm-generic-abstain"),
    )

    assert response.reply.kind == "unable_to_answer"
    evidence_service = runtime.evidence_executor.document_evidence_service
    assert isinstance(evidence_service, FakeCompanyEvidenceService)
    assert evidence_service.calls == 0
    assert len(composer.prompts) == 1


def test_direct_smalltalk_answers_without_evidence_or_actions():
    output = DirectComposerOutput(
        response_mode="smalltalk",
        reply=PrimaryReply(
            kind="answer",
            text="老师您好，我是小衍，请问有什么可以帮您的？",
            mentions=[],
        ),
    )
    runtime, adapter_client = _runtime(output)

    response = anyio.run(
        runtime.reply,
        _direct_request("Hi", context_id="dm-smalltalk"),
    )

    assert adapter_client.target_requests == []
    assert response.reply.kind == "answer"
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


def test_unavailable_adapter_explains_that_dm_send_was_not_attempted():
    output = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="确认发送吗？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="group", name="测试群"),
        content=DirectTextDraft(kind="text", text="测试公共"),
    )
    runtime, adapter_client = _runtime(output, adapter_available=False)

    response = anyio.run(
        runtime.reply,
        _direct_request(
            "发群消息给测试群说测试公共",
            context_id="dm-adapter-unavailable",
        ),
    )

    assert adapter_client.target_requests == []
    assert response.reply.text == (
        "企微适配器当前不可用，无法验证发送权限和目标，因此本次没有发送。请稍后重试。"
    )
    assert response.actions == []


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
        response_mode="smalltalk",
        reply=PrimaryReply(
            kind="answer",
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
        json=_direct_request("Hi", context_id="dm-followup-0").model_dump(mode="json"),
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
    assert greeting_response.json()["reply"]["kind"] == "answer"
    assert clarification_response.status_code == 200
    assert clarification_response.json()["reply"]["kind"] == "clarification"
    assert clarification_response.json()["actions"] == []
    assert prepare_response.status_code == 200
    assert "1/10" in prepare_response.json()["reply"]["text"]
    followup_prompt = composer.prompts[2]
    pending_position = followup_prompt.rfind("Pending clarification context JSON")
    assert pending_position > followup_prompt.rfind("</prompt_fragment>")
    assert pending_position < followup_prompt.rfind("Current user message")
    assert "Resolve the current message as a reply to the assistant question" in followup_prompt
    assert "complete channel fan-out target" in followup_prompt
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


def test_dm_outbound_draft_survives_content_and_target_kind_clarifications(
    monkeypatch,
):
    target_only = DirectComposerOutput(
        response_mode="clarify",
        reply=PrimaryReply(
            kind="clarification",
            text="请问要发送什么内容？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind=None, name="兴业银行"),
    )
    content_only = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="请确认是发送到渠道还是单个群？",
            mentions=[],
        ),
        content=DirectTextDraft(kind="text", text="公告测试"),
    )
    channel_only = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="确认向兴业银行渠道发送“公告测试”吗？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="兴业银行"),
    )
    runtime, adapter_client = _runtime(target_only)
    composer = FakeSequentialDirectComposer(
        [target_only, content_only, channel_only]
    )
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply",
        runtime.reply,
    )
    client = TestClient(app)

    content_question = client.post(
        "/reply",
        json=_direct_request(
            "发消息给兴业银行",
            context_id="dm-draft-1",
        ).model_dump(mode="json"),
    )
    target_kind_question = client.post(
        "/reply",
        json=_direct_request(
            "公告测试",
            context_id="dm-draft-2",
        ).model_dump(mode="json"),
    )
    prepare_response = client.post(
        "/reply",
        json=_direct_request(
            "渠道",
            context_id="dm-draft-3",
        ).model_dump(mode="json"),
    )

    assert content_question.status_code == 200
    assert content_question.json()["reply"]["text"] == "请问要发送什么内容？"
    assert target_kind_question.status_code == 200
    assert "同名群和渠道" in target_kind_question.json()["reply"]["text"]
    assert target_kind_question.json()["actions"] == []
    assert prepare_response.status_code == 200
    assert adapter_client.target_requests == [
        ("channel", "兴业银行"),
        ("group", "兴业银行"),
        ("channel", "兴业银行"),
    ]
    assert prepare_response.json()["actions"] == [
        {
            "action_id": "act-1",
            "type": "prepare_outbound_message",
            "target": {
                "kind": "channel",
                "name": "兴业银行",
                "resolve_ref": "outbound-target:" + "b" * 64,
            },
            "content": {"kind": "text", "text": "公告测试"},
        }
    ]
    assert '"pending_outbound_draft"' in composer.prompts[1]
    assert '"unresolved_fields": [' in composer.prompts[1]
    assert '"content"' in composer.prompts[1]


def test_dm_cancellation_does_not_become_pending_outbound_content(
    monkeypatch,
):
    target_only = DirectComposerOutput(
        response_mode="clarify",
        reply=PrimaryReply(
            kind="clarification",
            text="请问要发送什么内容？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="兴业银行"),
    )
    cancellation = DirectComposerOutput(
        response_mode="smalltalk",
        reply=PrimaryReply(
            kind="answer",
            text="好的，本次不发送。",
            mentions=[],
        ),
    )
    runtime, adapter_client = _runtime(target_only)
    composer = FakeSequentialDirectComposer([target_only, cancellation])
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply",
        runtime.reply,
    )
    client = TestClient(app)

    content_question = client.post(
        "/reply",
        json=_direct_request(
            "发渠道消息给兴业银行",
            context_id="dm-content-slot-1",
        ).model_dump(mode="json"),
    )
    cancellation_response = client.post(
        "/reply",
        json=_direct_request(
            "先不发了",
            context_id="dm-content-slot-2",
        ).model_dump(mode="json"),
    )

    assert content_question.status_code == 200
    assert content_question.json()["reply"]["text"] == "请问要发送什么内容？"
    assert cancellation_response.status_code == 200
    assert cancellation_response.json()["reply"]["text"] == "好的，本次不发送。"
    assert cancellation_response.json()["actions"] == []
    assert adapter_client.target_requests == []
    assert "ExecutionPlan JSON" not in composer.prompts[0]
    assert "IntentGate JSON" not in composer.prompts[0]


def test_dm_topic_switch_answers_company_question_and_clears_pending_outbound(
    monkeypatch,
):
    fact = EvidenceFact(
        fact_type="document_context",
        value="衍复投资是一家量化投资公司。",
        source_type="document_mcp",
        source_id="company-profile",
        artifact_type="document_context",
        metadata={"content_is_data_only": True},
    )
    target_only = DirectComposerOutput(
        response_mode="clarify",
        reply=PrimaryReply(
            kind="clarification",
            text="请问要发送什么内容？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="兴业银行"),
    )
    knowledge_request = DirectComposerOutput(
        response_mode="request_company_info",
        reply=PrimaryReply(
            kind="unable_to_answer",
            text="我需要查询内部资料后再回答。",
            mentions=[],
        ),
    )
    knowledge_answer = DirectComposerOutput(
        response_mode="answer_company_info",
        claims=["衍复投资是一家量化投资公司"],
        evidence_ids=[evidence_id(fact)],
        reply=PrimaryReply(
            kind="answer",
            text="衍复投资是一家量化投资公司。",
            mentions=[],
        ),
    )
    content_without_target = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="请确认发送目标。",
            mentions=[],
        ),
        content=DirectTextDraft(kind="text", text="公告测试"),
    )
    runtime, adapter_client = _runtime(target_only, company_fact=fact)
    composer = FakeSequentialDirectComposer(
        [target_only, knowledge_request, knowledge_answer, content_without_target]
    )
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply",
        runtime.reply,
    )
    client = TestClient(app)

    content_question = client.post(
        "/reply",
        json=_direct_request(
            "发渠道消息给兴业证券",
            context_id="dm-abstain-content-slot-1",
        ).model_dump(mode="json"),
    )
    topic_switch_response = client.post(
        "/reply",
        json=_direct_request(
            "先不发了，衍复是什么公司？",
            context_id="dm-abstain-content-slot-2",
        ).model_dump(mode="json"),
    )
    later_content_response = client.post(
        "/reply",
        json=_direct_request(
            "公告测试",
            context_id="dm-abstain-content-slot-3",
        ).model_dump(mode="json"),
    )

    assert content_question.status_code == 200
    assert content_question.json()["reply"]["text"] == "请问要发送什么内容？"
    assert topic_switch_response.status_code == 200
    assert topic_switch_response.json()["reply"]["text"] == (
        "衍复投资是一家量化投资公司。"
    )
    assert topic_switch_response.json()["actions"] == []
    assert later_content_response.status_code == 200
    assert later_content_response.json()["actions"] == []
    assert adapter_client.target_requests == []
    assert '"status": "none"' in composer.prompts[3]


def test_dm_unreachable_channel_can_be_replaced_without_reentering_content(
    monkeypatch,
):
    target_only = DirectComposerOutput(
        response_mode="clarify",
        reply=PrimaryReply(
            kind="clarification",
            text="请问要发送什么内容？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="兴业银行"),
    )
    content_only = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="请确认是否按当前目标准备发送？",
            mentions=[],
        ),
        content=DirectTextDraft(kind="text", text="渠道测试"),
    )
    replacement_target = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="请确认是否发送到兴业证券渠道？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="兴业证券"),
    )
    runtime, adapter_client = _runtime(target_only)
    composer = FakeSequentialDirectComposer(
        [target_only, content_only, replacement_target]
    )
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )

    async def resolve_target(
        target_kind: str,
        target_name: str,
    ) -> OutboundTargetResolveResult:
        adapter_client.target_requests.append((target_kind, target_name))
        if target_name == "兴业证券" and target_kind == "channel":
            return OutboundTargetResolveResult(
                status="resolved",
                reason_code="ok",
                display_name=target_name,
                target_kind="channel",
                target_count=2,
                resolved_count=2,
                resolve_ref="outbound-target:" + "b" * 64,
            )
        recognized_channel = target_name == "兴业银行" and target_kind == "channel"
        return OutboundTargetResolveResult(
            status="missing",
            reason_code="target_incomplete" if recognized_channel else "target_not_found",
            display_name=target_name,
            target_kind=target_kind,
            target_count=11 if recognized_channel else 0,
            resolved_count=0,
            resolve_ref="",
        )

    adapter_client.resolve_outbound_target_async = resolve_target  # type: ignore[method-assign]
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply",
        runtime.reply,
    )
    client = TestClient(app)

    content_question = client.post(
        "/reply",
        json=_direct_request(
            "发渠道消息给兴业银行",
            context_id="dm-replace-target-1",
        ).model_dump(mode="json"),
    )
    unavailable = client.post(
        "/reply",
        json=_direct_request(
            "渠道测试",
            context_id="dm-replace-target-2",
        ).model_dump(mode="json"),
    )
    replacement = client.post(
        "/reply",
        json=_direct_request(
            "那兴业证券呢",
            context_id="dm-replace-target-3",
        ).model_dump(mode="json"),
    )

    assert content_question.status_code == 200
    assert unavailable.status_code == 200
    assert "已识别渠道" in unavailable.json()["reply"]["text"]
    assert unavailable.json()["actions"] == []
    assert replacement.status_code == 200
    assert adapter_client.target_requests == [
        ("channel", "兴业银行"),
        ("group", "兴业银行"),
        ("channel", "兴业证券"),
    ]
    assert replacement.json()["actions"] == [
        {
            "action_id": "act-1",
            "type": "prepare_outbound_message",
            "target": {
                "kind": "channel",
                "name": "兴业证券",
                "resolve_ref": "outbound-target:" + "b" * 64,
            },
            "content": {"kind": "text", "text": "渠道测试"},
        }
    ]
    assert '"pending_outbound_draft"' in composer.prompts[2]
    assert "渠道测试" in composer.prompts[2]


def test_dm_invalid_followup_clarifies_without_guessing_and_preserves_pending(
    monkeypatch,
):
    target_only = DirectComposerOutput(
        response_mode="clarify",
        reply=PrimaryReply(
            kind="clarification",
            text="请问要发送什么内容？",
            mentions=[],
        ),
        target=DirectTargetDraft(
            kind="group",
            name="XYYH-衍复投资沟通交流测试",
        ),
    )
    recovered_content = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(
            kind="clarification",
            text="请确认是否按当前目标准备发送？",
            mentions=[],
        ),
        content=DirectTextDraft(kind="text", text="公告测试"),
    )
    runtime, adapter_client = _runtime(target_only)
    composer = FakeInvalidFollowupDirectComposer(target_only, recovered_content)
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply",
        runtime.reply,
    )
    client = TestClient(app)

    content_question = client.post(
        "/reply",
        json=_direct_request(
            "发消息给群XYYH-衍复投资沟通交流测试",
            context_id="dm-invalid-output-1",
        ).model_dump(mode="json"),
    )
    prepare_response = client.post(
        "/reply",
        json=_direct_request(
            "公告测试",
            context_id="dm-invalid-output-2",
        ).model_dump(mode="json"),
    )
    recovered_response = client.post(
        "/reply",
        json=_direct_request(
            "公告测试",
            context_id="dm-invalid-output-3",
        ).model_dump(mode="json"),
    )

    assert content_question.status_code == 200
    assert prepare_response.status_code == 200
    assert prepare_response.json()["reply"]["kind"] == "clarification"
    assert prepare_response.json()["actions"] == []
    assert composer.calls == 4
    assert adapter_client.target_requests == [
        ("group", "XYYH-衍复投资沟通交流测试")
    ]
    assert recovered_response.json()["actions"][0]["content"] == {
        "kind": "text",
        "text": "公告测试",
    }


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


def test_dm_correction_misclassified_as_answer_is_recomposed_before_later_send(
    monkeypatch,
):
    ledger = ActionLedger()
    original_text = (
        "衍复的策略依然是在严格对标指数进行风控约束的前提下持续追求超额收益，"
        "并非依赖行情的风格暴露。同时策略模型高中低的全频次覆盖，能够有效应对"
        "市场行情波动，帮助投资人在长期获得与预期收益相符的稳健投资收益，提升投资体验。"
    )
    original = DirectComposerOutput(
        response_mode="prepare_outbound_message",
        reply=PrimaryReply(kind="clarification", text="请确认是否发送？", mentions=[]),
        target=DirectTargetDraft(kind="channel", name="银河证券"),
        content=DirectTextDraft(kind="text", text=original_text),
    )
    misclassified_correction = DirectComposerOutput(
        response_mode="smalltalk",
        pending_confirmation_resolution="correct",
        reply=PrimaryReply(
            kind="answer",
            text="好的，那原文就是“你好”。",
            mentions=[],
        ),
    )
    corrected = original.model_copy(
        update={
            "pending_confirmation_resolution": "correct",
            "content": DirectTextDraft(kind="text", text="你好"),
        }
    )
    confirm = DirectComposerOutput(
        response_mode="execute_prepared_outbound_message",
        pending_confirmation_resolution="confirm",
        reply=PrimaryReply(kind="answer", text="", mentions=[]),
        confirmation_ref=_CONFIRMATION_REF,
    )
    runtime, _adapter_client = _runtime(original, action_ledger=ledger)
    composer = FakeSequentialDirectComposer(
        [original, misclassified_correction, corrected, confirm]
    )
    runtime._build_agent = (  # type: ignore[method-assign]
        lambda stage="knowledge_composer": composer
    )
    monkeypatch.setattr(
        "market_support_crewai_agent.server.main.build_reply",
        runtime.reply,
    )
    client = TestClient(app)

    first_response = client.post(
        "/reply",
        json=_direct_request(
            f"在银河证券渠道发送{original_text}",
            context_id="dm-correct-1",
        ).model_dump(mode="json"),
    )
    assert first_response.status_code == 200
    first = first_response.json()
    ledger.record_feedback(
        ActionFeedbackRequest(
            conversation_key="wecom:dm-thread:any-sender",
            group_id="dm-thread",
            sender_id="any-sender",
            response_id=first["response_id"],
            executions=[
                {
                    "action_type": "prepare_outbound_message",
                    "status": "executed",
                    "action_id": "act-1",
                    "adapter_result": {"confirmation_ref": _CONFIRMATION_REF},
                }
            ],
        )
    )
    second_response = client.post(
        "/reply",
        json=_direct_request(
            "不，原文是你好",
            context_id="dm-correct-2",
        ).model_dump(mode="json"),
    )
    assert second_response.status_code == 200
    second = second_response.json()
    ledger.record_feedback(
        ActionFeedbackRequest(
            conversation_key="wecom:dm-thread:any-sender",
            group_id="dm-thread",
            sender_id="any-sender",
            response_id=second["response_id"],
            executions=[
                {
                    "action_type": "prepare_outbound_message",
                    "status": "executed",
                    "action_id": "act-1",
                    "adapter_result": {"confirmation_ref": _LATEST_CONFIRMATION_REF},
                }
            ],
        )
    )

    response = client.post(
        "/reply",
        json=_direct_request("发送", context_id="dm-correct-3").model_dump(mode="json"),
    )

    assert response.status_code == 200
    assert second["actions"][0]["content"] == {
        "kind": "text",
        "text": "你好",
    }
    assert '"status": "awaiting_user_answer"' in composer.prompts[2]
    assert '"pending_confirmation"' in composer.prompts[2]
    assert "你好" in composer.prompts[2]
    assert "pending_confirmation_resolution_mode_mismatch" in composer.prompts[2]
    assert len(composer.prompts) == 4
    action = response.json()["actions"][0]
    assert action["type"] == "execute_prepared_outbound_message"
    assert action["confirmation_ref"] == _LATEST_CONFIRMATION_REF


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
