from __future__ import annotations

from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.state.conversation_store import ConversationMessage
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.llm.prompt_assembler import (
    PromptProgram,
    assemble_prompt_program,
)
from market_support_crewai_agent.runtime.llm.prompt_context import (
    IntentGateResult,
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.llm.prompt_profiles import (
    ModelFamily,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings


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
    history: list[ConversationMessage] | None = None,
) -> IntentGateResult:
    """Legacy audit hint only.

    The planner LLM is the semantic router. This function must not infer artifact
    kind, compliance status, or side-effect intent from message substrings.
    """
    del policy, history
    return IntentGateResult(
        artifact_hint="unclear",
        side_effect_hint=False,
        named_strategy_count=_named_strategy_count(request, canonical_context),
        compliance_hint="unknown",
        matched_keywords=[],
        confidence=0.0,
    )


def select_prompt_program(ctx: PromptAssemblyContext) -> PromptProgram:
    profile = prompt_profile_by_stage(ctx.stage, ctx.model_family)
    if ctx.stage == "planner_intent":
        fragment_ids = _planner_fragments(ctx)
    elif ctx.stage == "knowledge_composer":
        fragment_ids = _knowledge_composer_fragments(ctx)
    elif ctx.stage == "smalltalk_composer":
        fragment_ids = _smalltalk_composer_fragments(ctx)
    elif ctx.stage == "alignment_verifier":
        fragment_ids = _alignment_verifier_fragments(ctx)
    else:
        raise ValueError(f"unsupported prompt stage: {ctx.stage}")
    return assemble_prompt_program(ctx, profile, tuple(fragment_ids))


def _planner_fragments(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.planner_intent",
        _model_fragment(ctx.model_family),
        "planner.intent_taxonomy",
        "output.intent_frame_schema",
        "compliance.reason_codes",
    ]


def _knowledge_composer_fragments(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.knowledge_composer",
        _model_fragment(ctx.model_family),
        "output.reply_response_no_actions",
        "evidence.document_grounding",
        "style.wecom_concise_zh",
    ]


def _smalltalk_composer_fragments(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.smalltalk_composer",
        _model_fragment(ctx.model_family),
        "output.reply_response_no_actions",
        "style.wecom_concise_zh",
    ]


def _alignment_verifier_fragments(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.alignment_verifier",
        _model_fragment(ctx.model_family),
        "output.reply_alignment_verdict_schema",
    ]


def _model_fragment(model_family: ModelFamily) -> str:
    if model_family in {"ds_v4pro", "deepseek"}:
        return "model.ds_v4pro.structured"
    return "model.generic.structured"


def _named_strategy_count(
    request: ReplyRequest,
    canonical_context: CanonicalContext,
) -> int:
    if canonical_context.strategy_status == "resolved":
        return 1
    if canonical_context.strategy_status == "ambiguous":
        return len(canonical_context.strategy_candidates)
    text = _user_visible_text(request.message).lower()
    return sum(
        1
        for strategy in request.available_strategies
        if strategy and strategy.lower() in text
    )


def _user_visible_text(message: str) -> str:
    lines = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("[adapter_") and stripped.endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()
