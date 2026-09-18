from __future__ import annotations

import anyio
import pytest

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_manifest_v2,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.recall.document_qa_corpus import (
    parse_qa_entries,
)
from market_support_crewai_agent.runtime.recall.question_models import (
    QuestionRecallEntry,
    QuestionRecallMatch,
)
from market_support_crewai_agent.runtime.recall.question_recall_analyzer import (
    approved_static_entries,
    document_entry,
    search_question_recall,
)
from market_support_crewai_agent.runtime.recall.service import QuestionRecallService
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_requests import make_v2_envelope

_INTERNAL_COMPANY_READ_GRANTS = {
    "contract_version": "principal-grants.v1",
    "read_capabilities": ["query_internal_company_info"],
    "outbound_actions": [],
    "mention_types": [],
}


def _internal_company_request(message: str) -> KernelReplyRequestV1:
    return make_v2_envelope(message, grants=_INTERNAL_COMPANY_READ_GRANTS).request


def _verified_policy(request: KernelReplyRequestV1) -> PolicyManifestV2:
    return compile_policy_manifest_v2(
        request,
        business_scope_authority_v1(request.business_scope),
        policy_ledger_summary_v1((), 0),
    )


def _collect_approved_static_match(
    service: QuestionRecallService,
    request: KernelReplyRequestV1,
    policy: PolicyManifestV2,
) -> QuestionRecallMatch:
    return anyio.run(service.collect, request, policy).match


def test_recall_hit_when_width_case_and_finance_alias_match() -> None:
    request = _internal_company_request("衍复投资ＵＲＬ是什么？")
    policy = _verified_policy(request)
    service = QuestionRecallService(
        Settings(
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local",
        )
    )

    match = _collect_approved_static_match(service, request, policy)

    assert match.decision == "recall_hit"
    assert match.status == "matched"
    assert match.confidence >= 0.72
    assert match.candidates[0].entry_id == "company_basic_contact"
    assert "company_basic_contact" in match.candidates[0].canonical_id
    assert "normalizer_version" in match.trace
    assert "lexicon_version" in match.trace
    assert match.source_type == "approved_static_knowledge"
    assert "网址" in match.evidence_text


def test_recall_fail_open_when_ambiguous_or_low_coverage() -> None:
    request = _internal_company_request("这个怎么看")
    policy = _verified_policy(request)
    service = QuestionRecallService(
        Settings(
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local",
        )
    )

    match = _collect_approved_static_match(service, request, policy)

    assert match.decision == "fail_open"
    assert match.confidence < 0.72
    assert match.allow_planner_document_context_fallback is True


def test_recall_fail_open_when_cjk_terms_are_scrambled() -> None:
    match = search_question_recall(
        "网址投资衍复是么什",
        approved_static_entries(),
    )

    assert match.decision == "fail_open"


def test_recall_fail_open_when_message_contains_blocked_compliance_wording() -> None:
    request = _internal_company_request("保本无风险，衍复投资ＵＲＬ是什么？")
    policy = _verified_policy(request)
    service = QuestionRecallService(
        Settings(
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local",
        )
    )

    match = _collect_approved_static_match(service, request, policy)

    assert match.decision == "fail_open"
    assert match.reason_code == "compliance_risky_message"


@pytest.mark.parametrize(
    "message",
    [
        "产品预计收益多少？衍复投资ＵＲＬ是什么？",
        "最低收益多少，衍复投资ＵＲＬ是什么？",
        "给我私人微信，衍复投资ＵＲＬ是什么？",
        "其他管理人和你们比怎么样？衍复投资ＵＲＬ是什么？",
        "发我一个四级估值表吧，衍复投资网址是什么？",
        "保 本 无 风 险，衍复投资URL是什么？",
        "产品到期能有多少收益？衍复投资ＵＲＬ是什么？",
        "能保证回报吗？衍复投资ＵＲＬ是什么？",
        "加你微信了，通过一下，衍复投资ＵＲＬ是什么？",
        "赎回费可以免了吗？衍复投资ＵＲＬ是什么？",
        "周末可以一起看电影吗？衍复投资ＵＲＬ是什么？",
        "不是合格投资人可以买吗？衍复投资ＵＲＬ是什么？",
        "100万以下可以买吗？衍复投资ＵＲＬ是什么？",
    ],
)
def test_recall_fail_open_for_compliance_taxonomy_risks(message: str) -> None:
    request = _internal_company_request(message)
    policy = _verified_policy(request)
    service = QuestionRecallService(
        Settings(
            doc_mcp_enabled=True,
            doc_mcp_base_url="http://doc-mcp.local",
        )
    )

    match = _collect_approved_static_match(service, request, policy)

    assert match.decision == "fail_open"
    assert match.reason_code == "compliance_risky_message"


def test_document_mcp_recall_candidates_do_not_shortcut_planner() -> None:
    parsed_entry = QuestionRecallEntry(
        entry_id="qa:test",
        canonical_id="document_mcp.qa:test",
        doc_id="document",
        title="document",
        question="衍复投资网址是什么？",
        answer="网址：http://example.test",
        source_type="document_mcp",
        surfaces=("衍复投资网址是什么？",),
    )

    match = search_question_recall("衍复投资网址是什么？", (parsed_entry,))

    assert match.decision != "recall_hit"


def test_recall_fail_open_when_answer_claims_adapter_execution() -> None:
    entry = QuestionRecallEntry(
        entry_id="unsafe_answer",
        canonical_id="approved_static_knowledge.unsafe_answer",
        doc_id="unsafe_answer",
        title="不安全已发送话术",
        question="材料发了吗？",
        answer="我已经发送给您了。",
        source_type="approved_static_knowledge",
        surfaces=("材料发了吗？",),
    )

    match = search_question_recall("材料发了吗？", (entry,))

    assert match.decision == "fail_open"


@pytest.mark.parametrize(
    "answer",
    [
        "我已经发送给您了。",
        "我已经同步给您了。",
        "材料已发您。",
        "请查收。",
        "发送完成。",
        "发好了。",
        "The report has been sent.",
        "Sent successfully.",
    ],
)
def test_recall_fail_open_for_adapter_execution_claim_variants(answer: str) -> None:
    entry = QuestionRecallEntry(
        entry_id="unsafe_answer",
        canonical_id="approved_static_knowledge.unsafe_answer",
        doc_id="unsafe_answer",
        title="不安全执行状态话术",
        question="材料发了吗？",
        answer=answer,
        source_type="approved_static_knowledge",
        surfaces=("材料发了吗？",),
    )

    match = search_question_recall("材料发了吗？", (entry,))

    assert match.decision == "fail_open"


def test_recall_sanitizes_document_metadata_before_prompt_exposure() -> None:
    parsed = parse_qa_entries(
        [
            {
                "id": "file:///Users/ivan/secret.md",
                "title": "secret token=abc",
                "content": "Q：测试问题\nA：测试答案",
            }
        ]
    )

    entry = document_entry(parsed[0])

    assert "file://" not in entry.doc_id
    assert "/Users/" not in entry.doc_id
    assert "secret" not in entry.title.lower()
    assert "token" not in entry.title.lower()
