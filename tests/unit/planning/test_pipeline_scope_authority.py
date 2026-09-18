from __future__ import annotations

from collections.abc import Callable
from inspect import Parameter, signature

import pytest

from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import (
    KernelReplyRequestV1,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.pipeline import build_candidate_response
from market_support_crewai_agent.runtime.policy.manifest import (
    PolicyManifestV2,
    compile_policy_authority_core_v1,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.policy.ontology import DomainContextV1Builder
from market_support_crewai_agent.runtime.prompts.context import IntentGateResult
from tests.helpers.reply_contract_requests import make_v2_envelope
from tests.integration.runtime._recall_pipeline_fixtures import (
    document_candidate,
    static_hit,
)
from tests.integration.runtime._recall_pipeline_runtime_fixtures import (
    DocumentCollector,
    RecallPipelineRuntime,
    StaticCollector,
)


class _PlanCaptureRuntime(RecallPipelineRuntime):
    def __init__(self, policy: PolicyManifestV2) -> None:
        super().__init__(
            StaticCollector(static_hit(policy, with_payload=False)),
            DocumentCollector(document_candidate()),
        )


def _policy_for(
    request: KernelReplyRequestV1,
) -> tuple[PolicyManifestV2, BusinessScopeAuthorityV1]:
    scope_authority = business_scope_authority_v1(request.business_scope)
    core = compile_policy_authority_core_v1(request, scope_authority)
    return (
        PolicyManifestV2.from_core(core, policy_ledger_summary_v1((), 0)),
        scope_authority,
    )


def _direct_t0_envelope() -> VerifiedRequestEnvelopeV1:
    return make_v2_envelope(
        "t0",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:pipeline",
            "direct_thread_ref": "direct:pipeline",
            "principal_ref": "principal:pipeline",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "pipeline",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    )


def _group_weekly_envelope() -> VerifiedRequestEnvelopeV1:
    return make_v2_envelope("请发周报")


def _group_t0_envelope() -> VerifiedRequestEnvelopeV1:
    return make_v2_envelope("t0")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("envelope_factory", "expected_origin", "expected_manifest"),
    (
        (_group_t0_envelope, "input_policy", "sales.handoff"),
        (_direct_t0_envelope, "input_policy", "general.handoff"),
        (_group_weekly_envelope, "direct_send", "weekly_report.send"),
    ),
)
async def test_active_v2_pipeline_reaches_deterministic_plan_origins(
    envelope_factory: Callable[[], VerifiedRequestEnvelopeV1],
    expected_origin: str,
    expected_manifest: str,
) -> None:
    envelope = envelope_factory()
    request = envelope.request
    policy, scope_authority = _policy_for(request)
    runtime = _PlanCaptureRuntime(policy)

    result = await build_candidate_response(
        runtime,
        request=request,
        domain_context=DomainContextV1Builder().build(
            request,
            scope_authority=scope_authority,
        ),
        policy=policy,
        model_family="generic",
        intent_gate=IntentGateResult(artifact_hint="unclear"),
        history=[],
        action_history=[],
        prompt_programs=[],
        llm_executions=[],
        scope_authority=scope_authority,
        state_key_ref=envelope.state_key_ref,
    )

    assert result.plan is runtime.plans[0]
    assert result.plan.contract_version == "execution-plan.v2"
    assert result.plan.origin == expected_origin
    assert result.plan.units[0].manifest_ref.manifest_id == expected_manifest


@pytest.mark.anyio
async def test_active_v2_pipeline_does_not_default_missing_scope_authority() -> None:
    envelope = make_v2_envelope("t0")
    request = envelope.request
    _policy, _scope_authority = _policy_for(request)
    scope_parameter = signature(build_candidate_response).parameters["scope_authority"]

    assert scope_parameter.kind is Parameter.KEYWORD_ONLY
    assert "=" not in str(scope_parameter)
