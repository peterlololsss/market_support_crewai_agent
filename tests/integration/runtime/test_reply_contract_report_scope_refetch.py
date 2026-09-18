from __future__ import annotations

import asyncio
import os
from typing import override

os.environ["CREWAI_MAX_RETRY_LIMIT"] = "0"

from market_support_crewai_agent.runtime.context.stage_inputs import (
    SanitizedAlignmentVerifierInputV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_commands import (
    DocumentMcpCacheConfigV1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    CanonicalEvidenceExecutionResultV1,
)
from market_support_crewai_agent.runtime.evidence.executor import EvidenceExecutor
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.planning.models import ExecutionPlanV2
from market_support_crewai_agent.runtime.policy.manifest import PolicyManifestV2
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerPromptInputV1,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.conversation_store import (
    ConversationStore,
)
from market_support_crewai_agent.runtime.validation.alignment_refetch import (
    AlignmentRefetchRequestV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.reply_contract_agents import install_fake_planner
from tests.helpers.reply_contract_evidence import (
    report_scope_products_fact,
    v2_evidence,
)
from tests.helpers.reply_contract_plan_fixtures import make_support_plan_spec
from tests.helpers.reply_contract_preflight import EmptyPreflightService
from tests.helpers.reply_contract_requests import make_v2_envelope

pytest_plugins: tuple[str, ...] = ("tests.helpers.reply_contract_runtime_state",)


def test_alignment_verifier_refetches_report_scope_products_from_typed_refetch():
    envelope = make_v2_envelope(
        "周报里有哪些产品？",
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": [
                "resolve_material_pack",
                "resolve_weekly_report",
                "resolve_sales_mention",
                "query_internal_company_info",
                "query_weekly_report_product_list",
            ],
            "outbound_actions": [
                "send_material_pack",
                "send_weekly_report",
                "send_monthly_report",
            ],
            "mention_types": ["sales"],
        },
    )
    runtime = CrewAIReplyRuntime(
        Settings(llm_api_key="test-key"),
        conversation_store=ConversationStore(),
        preflight_service=EmptyPreflightService(),
    )
    install_fake_planner(
        runtime,
        make_support_plan_spec(
            request=envelope.request,
            user_need="answer weekly report product list",
            artifact_kind="knowledge_answer",
            action_intent="answer",
            requested_capabilities=["weekly_report"],
            evidence_query=None,
            ambiguity_slots=[],
        ),
    )
    queries: list[str | None] = []

    class ReportScopeEvidenceExecutor(EvidenceExecutor):
        @override
        async def execute_v2(
            self,
            request: KernelReplyRequestV1,
            plan: ExecutionPlanV2,
            policy: PolicyManifestV2,
            *,
            scope_authority: BusinessScopeAuthorityV1,
            state_key_ref: str | None = None,
            document_cache_config: DocumentMcpCacheConfigV1 | None = None,
            alignment_refetch_request: AlignmentRefetchRequestV1 | None = None,
        ) -> CanonicalEvidenceExecutionResultV1:
            del state_key_ref, document_cache_config
            refetch = alignment_refetch_request
            query = (
                refetch.refined_evidence_query
                if refetch is not None
                else plan.units[0].evidence_query
            )
            queries.append(query)
            facts = (
                (report_scope_products_fact(plan),)
                if query == "report_scope_products"
                else ()
            )
            return v2_evidence(request, plan, policy, scope_authority, facts=facts)

    class FakeComposer:
        async def compose(
            self, input_value: ComposerPromptInputV1
        ) -> ComposerReplyOutput:
            rendered = input_value.model_dump_json()
            text = (
                "weekly report products: Product1, Product2"
                if "Product1" in rendered
                else "weekly report product scope unavailable"
            )
            return ComposerReplyOutput(
                response_mode="answer", reply=PrimaryReply(kind="answer", text=text)
            )

    class TypedRefetchThenPassVerifier:
        def __init__(self) -> None:
            self.calls: int = 0

        async def verify(
            self, input_value: SanitizedAlignmentVerifierInputV1
        ) -> ReplyAlignmentVerdict:
            del input_value
            self.calls += 1
            if self.calls == 1:
                return ReplyAlignmentVerdict(
                    aligned=False,
                    safe_to_return=False,
                    failure_code="missing_answer",
                    remediation="refetch_report_scope",
                    refined_evidence_query="report_scope_products",
                )
            return ReplyAlignmentVerdict(
                aligned=True, safe_to_return=True, confidence=0.9
            )

    runtime.evidence_executor = ReportScopeEvidenceExecutor(EmptyPreflightService())
    runtime.v2_composer = FakeComposer()
    runtime.alignment_verifier = TypedRefetchThenPassVerifier()

    response = asyncio.run(runtime.reply(envelope))

    assert queries == [None, "report_scope_products"]
    assert response.reply.text == "weekly report products: Product1, Product2"
