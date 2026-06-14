from __future__ import annotations

from market_support_crewai_agent.runtime.capabilities import (
    capability_by_name,
)
from market_support_crewai_agent.runtime.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.runtime.prompt_assembler import (
    PromptProgram,
    assemble_prompt_program,
)
from market_support_crewai_agent.runtime.prompt_context import (
    IntentGateResult,
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.prompt_profiles import (
    ModelFamily,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings


_SEND_KEYWORDS = ("发", "发送", "转发", "send")
_MATERIAL_KEYWORDS = ("材料", "材料包", "推介材料", "material")
_WEEKLY_KEYWORDS = ("周报", "weekly")
_MONTHLY_KEYWORDS = ("月报", "monthly")
_HUMAN_SUPPORT_KEYWORDS = (
    "销售",
    "客户经理",
    "人工",
    "帮我问",
    "问一下销售",
    "支持同事",
)
_BLOCKED_KEYWORDS = (
    "保本",
    "保证收益",
    "稳赚",
    "预期收益",
    "目标收益",
    "合同",
    "交易文件",
    "认购文件",
)
_KNOWLEDGE_KEYWORDS = (
    "怎么样",
    "有没有",
    "为什么",
    "回撤",
    "规模",
    "费率",
    "净值",
    "报告里有没有",
    "刚发的",
    "介绍",
    "是什么",
)


def model_family_from_settings(settings: Settings) -> ModelFamily:
    model = settings.llm_model.lower()
    if "deepseek-v4-pro" in model or "ds-v4pro" in model or "v4-pro" in model:
        return "ds_v4pro"
    if "deepseek" in model:
        return "deepseek"
    if "gpt" in model:
        return "gpt"
    if "claude" in model:
        return "claude"
    return "generic"


def route_intent(
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    policy: PolicyManifest,
) -> IntentGateResult:
    del policy
    text = request.message.lower()
    matched: list[str] = []

    blocked_matches = _matches(text, _BLOCKED_KEYWORDS)
    if blocked_matches:
        return IntentGateResult(
            artifact_hint="refusal",
            side_effect_hint=False,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="blocked",
            matched_keywords=_limit(blocked_matches),
            confidence=0.95,
        )

    send = _has_any(text, _SEND_KEYWORDS)
    side_effect_artifacts: list[str] = []
    if send and _has_any(text, _WEEKLY_KEYWORDS):
        side_effect_artifacts.append("weekly_report")
        matched.extend(_matches(text, _WEEKLY_KEYWORDS))
    if send and _has_any(text, _MONTHLY_KEYWORDS):
        side_effect_artifacts.append("monthly_report")
        matched.extend(_matches(text, _MONTHLY_KEYWORDS))
    if send and _has_any(text, _MATERIAL_KEYWORDS):
        side_effect_artifacts.append("material_pack")
        matched.extend(_matches(text, _MATERIAL_KEYWORDS))

    if len(side_effect_artifacts) > 1:
        return IntentGateResult(
            artifact_hint="unclear",
            side_effect_hint=True,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(matched),
            confidence=0.85,
        )
    if side_effect_artifacts:
        return IntentGateResult(
            artifact_hint=side_effect_artifacts[0],  # type: ignore[arg-type]
            side_effect_hint=True,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(matched),
            confidence=0.9,
        )

    human_matches = _matches(text, _HUMAN_SUPPORT_KEYWORDS)
    if human_matches:
        return IntentGateResult(
            artifact_hint="human_support",
            side_effect_hint=False,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(human_matches),
            confidence=0.8,
        )

    knowledge_matches = _matches(text, _KNOWLEDGE_KEYWORDS)
    if knowledge_matches:
        return IntentGateResult(
            artifact_hint="knowledge_answer",
            side_effect_hint=False,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(knowledge_matches),
            confidence=0.75,
        )

    return IntentGateResult(
        artifact_hint="unclear",
        side_effect_hint=False,
        named_strategy_count=_named_strategy_count(request, canonical_context),
        compliance_hint="unknown",
        matched_keywords=[],
        confidence=0.2,
    )


def select_prompt_program(ctx: PromptAssemblyContext) -> PromptProgram:
    profile = prompt_profile_by_stage(ctx.stage, ctx.model_family)
    fragment_ids = (
        _planner_fragments(ctx)
        if ctx.stage == "planner_intent"
        else _knowledge_composer_fragments(ctx)
    )
    return assemble_prompt_program(ctx, profile, tuple(fragment_ids))


def _planner_fragments(ctx: PromptAssemblyContext) -> list[str]:
    gate = ctx.intent_gate or route_intent(
        ctx.request,
        ctx.canonical_context,
        ctx.policy,
    )
    fragment_ids = [
        "base.planner_intent",
        _model_fragment(ctx.model_family),
        "output.intent_frame_schema",
        "compliance.reason_codes",
    ]
    artifact = gate.artifact_hint
    if artifact == "material_pack":
        fragment_ids.extend(_capability_fragments("material_pack"))
    elif artifact == "weekly_report":
        fragment_ids.extend(_capability_fragments("weekly_report"))
    elif artifact == "monthly_report":
        fragment_ids.extend(_capability_fragments("monthly_report"))
    elif artifact == "knowledge_answer":
        fragment_ids.extend(_capability_fragments("document_context"))
    elif artifact == "human_support":
        fragment_ids.extend(_capability_fragments("sales_mention"))
    elif artifact == "unclear" and gate.side_effect_hint:
        fragment_ids.append("examples.multi_artifact_clarification")

    if artifact == "refusal" or gate.compliance_hint in {"blocked", "risky"}:
        fragment_ids.append("compliance.refusal_examples")

    if ctx.request.channel_type == "bank" and artifact == "material_pack":
        fragment_ids.append("channel.bank_material_rules")

    return _dedupe(fragment_ids)


def _knowledge_composer_fragments(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.knowledge_composer",
        _model_fragment(ctx.model_family),
        "output.reply_response_no_actions",
        "evidence.document_grounding",
        "style.wecom_concise_zh",
    ]


def _capability_fragments(capability_name: str) -> list[str]:
    capability = capability_by_name(capability_name)
    if capability is None:
        return []
    return list(capability.planner_fragment_ids)


def _model_fragment(model_family: ModelFamily) -> str:
    if model_family in {"ds_v4pro", "deepseek"}:
        return "model.ds_v4pro.structured"
    return "model.generic.structured"


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return bool(_matches(text, keywords))


def _matches(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword.lower() in text]


def _limit(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= 20:
            break
    return output


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _named_strategy_count(
    request: ReplyRequest,
    canonical_context: CanonicalContext,
) -> int:
    if canonical_context.strategy_status == "resolved":
        return 1
    if canonical_context.strategy_status == "ambiguous":
        return len(canonical_context.strategy_candidates)
    text = request.message.lower()
    return sum(
        1
        for strategy in request.available_strategies
        if strategy and strategy.lower() in text
    )
