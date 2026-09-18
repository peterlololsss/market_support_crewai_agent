from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from market_support_crewai_agent.runtime.decisions.business_fact_models import (
    UnitBusinessFactsV1,
)
from market_support_crewai_agent.runtime.evidence.grounding import (
    ExecutionUnitGroundingV1,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_admission import (
    static_fact,
)
from market_support_crewai_agent.runtime.evidence.internal_company_knowledge_models import (
    GatewayStaticContextV1,
    RegisteredMediaBindingV1,
)
from market_support_crewai_agent.runtime.evidence.scope_authority import (
    BusinessScopeAuthorityV1,
    business_scope_authority_v1,
)
from market_support_crewai_agent.runtime.identity import KernelReplyRequestV1
from market_support_crewai_agent.runtime.integrations.adapter.preflight import (
    AdapterPreflightSnapshot,
)
from market_support_crewai_agent.runtime.planning.models import (
    ComplianceDecisionV1,
    DistributionExecutionDomainScopeV2,
    ExecutionPlanUnitV2,
    ExecutionPlanV2,
    UnscopedExecutionDomainScopeV2,
)
from market_support_crewai_agent.runtime.planning.plan_hash import execution_plan_id_v2
from market_support_crewai_agent.runtime.policy.capabilities import (
    CAPABILITY_MANIFEST_REGISTRY,
    ManifestRefV1,
)
from market_support_crewai_agent.runtime.policy.manifest import (
    compile_policy_manifest_v2,
    policy_ledger_summary_v1,
)
from market_support_crewai_agent.runtime.recall.approved_static_knowledge import (
    APPROVED_STATIC_MANIFEST_REF,
)
from market_support_crewai_agent.runtime.rendering.v2_composer import (
    ComposerInvocationV1,
    V2Composer,
)
from market_support_crewai_agent.runtime.validation.locator_safety import (
    LocatorSafetyClassifierV1,
)
from tests.helpers.reply_contract_requests import make_v2_envelope

_STATIC_ROWS = {
    "company_shareholders": (
        "%%company_shareholders.png%%",
        "company_shareholders_chart",
    ),
    "company_historical_aum": (
        "%%company_historical_aum.png%%",
        "company_historical_aum_chart",
    ),
}


@dataclass(frozen=True, slots=True)
class ComposerScenario:
    request: KernelReplyRequestV1
    authority: BusinessScopeAuthorityV1
    plan: ExecutionPlanV2
    groundings: tuple[ExecutionUnitGroundingV1, ...]
    invocation: ComposerInvocationV1


@dataclass(frozen=True, slots=True)
class ComposerRuntimeStub:
    v2_composer: V2Composer | None


def composer_scenario(
    mode: Literal["knowledge_answer", "smalltalk"],
    *,
    unit_count: int = 1,
    direct: bool = False,
) -> ComposerScenario:
    request = _request(direct=direct)
    authority = business_scope_authority_v1(request.business_scope)
    plan = _plan(mode, unit_count, authority)
    groundings = tuple(_grounding(unit) for unit in plan.units)
    policy = compile_policy_manifest_v2(
        request,
        authority,
        policy_ledger_summary_v1(
            recent_artifact_types=(),
            recent_executed_count=0,
        ),
    )
    invocation = ComposerInvocationV1(
        request=request,
        policy=policy,
        scope_authority=authority,
        plan=plan,
        preflight=AdapterPreflightSnapshot.empty(),
        groundings=groundings,
        media_bindings=(),
        locator_safety=LocatorSafetyClassifierV1(
            internal_origins=frozenset(),
            secrets=(),
        ),
        now=datetime(2026, 7, 19, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    return ComposerScenario(
        request=request,
        authority=authority,
        plan=plan,
        groundings=groundings,
        invocation=invocation,
    )


def with_static_facts(
    scenario: ComposerScenario,
    entry_ids: tuple[str, ...],
) -> ComposerScenario:
    if len(entry_ids) != len(scenario.plan.units):
        raise AssertionError("one static entry is required per plan unit")
    groundings: list[ExecutionUnitGroundingV1] = []
    bindings: list[RegisteredMediaBindingV1] = []
    for unit, grounding, entry_id in zip(
        scenario.plan.units,
        scenario.groundings,
        entry_ids,
        strict=True,
    ):
        marker, asset_id = _STATIC_ROWS[entry_id]
        fact, unit_bindings = static_fact(
            GatewayStaticContextV1(
                entry_id=entry_id,
                manifest_ref=APPROVED_STATIC_MANIFEST_REF,
                text=marker,
                selected_asset_ids=(asset_id,),
            ),
            unit,
        )
        if fact is None:
            raise AssertionError("static fixture must produce one admitted fact")
        groundings.append(
            grounding.model_copy(
                update={
                    "allowed_evidence_ids": (fact.evidence_id,),
                    "allowed_evidence": (fact,),
                    "business_facts": UnitBusinessFactsV1(evidence_fact_count=1),
                }
            )
        )
        bindings.extend(unit_bindings)
    grounded = tuple(groundings)
    invocation = replace(
        scenario.invocation,
        groundings=grounded,
        media_bindings=tuple(bindings),
    )
    return replace(scenario, groundings=grounded, invocation=invocation)


def _request(*, direct: bool) -> KernelReplyRequestV1:
    if not direct:
        return make_v2_envelope("请介绍公司的投研策略").request
    return make_v2_envelope(
        "请介绍公司的投研策略",
        identity={
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "direct",
            "tenant_ref": "tenant:test",
            "direct_thread_ref": "direct:thread-1",
            "principal_ref": "principal:sender-1",
        },
        presentation={
            "contract_version": "direct-presentation.v1",
            "principal_name": "test user",
        },
        business_scope={"kind": "unscoped"},
        grants={
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["query_internal_company_info"],
            "outbound_actions": [],
            "mention_types": [],
        },
    ).request


def _plan(
    mode: Literal["knowledge_answer", "smalltalk"],
    unit_count: int,
    authority: BusinessScopeAuthorityV1,
) -> ExecutionPlanV2:
    manifest_id = (
        "answer_internal_company_knowledge"
        if mode == "knowledge_answer"
        else "general.smalltalk"
    )
    manifest = CAPABILITY_MANIFEST_REGISTRY.get(manifest_id)
    manifest_ref = ManifestRefV1(
        manifest_id=manifest.manifest_id,
        manifest_version=manifest.manifest_version,
    )
    scope = _execution_scope(authority)
    units = tuple(
        ExecutionPlanUnitV2(
            unit_id=f"unit-{index}",
            manifest_ref=manifest_ref,
            answerability_policy=(
                "answer" if mode == "knowledge_answer" else "smalltalk"
            ),
            artifact_kind=mode,
            scope=scope,
            evidence_query=(
                f"介绍策略-{index}" if mode == "knowledge_answer" else None
            ),
        )
        for index in range(1, unit_count + 1)
    )
    draft = ExecutionPlanV2.model_construct(
        execution_plan_id="epl1:" + "0" * 64,
        origin="planner",
        user_need="回答用户问题",
        artifact_kind=mode,
        response_mode=mode,
        compliance=ComplianceDecisionV1(
            is_compliant=True,
            reason_code="compliant_product_request",
        ),
        units=units,
        selected_manifest_refs=(manifest_ref,),
        adapter_resolves=(),
        action_intents=(),
        guardrail_decisions=(),
        confidence=0.9,
        plan_spec=None,
    )
    return draft.model_copy(update={"execution_plan_id": execution_plan_id_v2(draft)})


def _execution_scope(
    authority: BusinessScopeAuthorityV1,
) -> DistributionExecutionDomainScopeV2 | UnscopedExecutionDomainScopeV2:
    if authority.scope.kind == "distribution":
        return DistributionExecutionDomainScopeV2(
            business_scope_ref=authority.business_scope_ref,
            channel_kind=authority.scope.channel_type,
        )
    return UnscopedExecutionDomainScopeV2()


def _grounding(unit: ExecutionPlanUnitV2) -> ExecutionUnitGroundingV1:
    return ExecutionUnitGroundingV1(
        unit_id=unit.unit_id,
        manifest_ref=unit.manifest_ref,
        answerability=unit.answerability_policy,
        scope=unit.scope,
        evidence_query=unit.evidence_query,
        action_intents=unit.action_intents,
        business_facts=UnitBusinessFactsV1(evidence_fact_count=0),
    )
