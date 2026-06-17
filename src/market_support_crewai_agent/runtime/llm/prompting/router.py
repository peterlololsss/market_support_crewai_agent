from __future__ import annotations

from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.state.conversation_store import ConversationMessage
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.runtime.llm.prompting.assembler import (
    PromptAssembler,
    PromptProgram,
)
from market_support_crewai_agent.runtime.llm.prompting.context import (
    IntentGateResult,
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.llm.prompting.profiles import (
    ModelFamily,
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
    """Audit hint only.

    The planner LLM is the semantic router. This function must not infer artifact
    kind, compliance status, or side-effect intent from message substrings.
    """
    del policy, history
    return IntentGateResult(
        artifact_hint="unclear",
        side_effect_hint=False,
        named_strategy_count=_named_strategy_count(request, canonical_context),
        compliance_hint="unknown",
        confidence=0.0,
    )


def select_prompt_program(ctx: PromptAssemblyContext) -> PromptProgram:
    if ctx.stage == "planner_intent":
        return PromptAssembler().assemblePlannerPrompt(ctx)
    if ctx.stage in {"knowledge_composer", "smalltalk_composer"}:
        return PromptAssembler().assembleAgentPrompt(ctx)
    elif ctx.stage == "alignment_verifier":
        return PromptAssembler().assembleVerifierPrompt(ctx)
    raise ValueError(f"unsupported prompt stage: {ctx.stage}")


def planner_fragment_ids(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.planner_intent",
        _model_fragment(ctx.model_family),
        "planner.intent_taxonomy",
        "output.plan_spec_schema",
        "compliance.reason_codes",
    ]


def knowledge_composer_fragment_ids(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.knowledge_composer",
        _model_fragment(ctx.model_family),
        "output.reply_response_no_actions",
        "evidence.document_grounding",
        "style.wecom_concise_zh",
    ]


def smalltalk_composer_fragment_ids(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.smalltalk_composer",
        _model_fragment(ctx.model_family),
        "output.reply_response_no_actions",
        "style.wecom_concise_zh",
    ]


def verifier_fragment_ids(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.alignment_verifier",
        _model_fragment(ctx.model_family),
        "output.reply_alignment_verdict_schema",
    ]


def agent_fragment_ids(ctx: PromptAssemblyContext) -> list[str]:
    if ctx.stage == "knowledge_composer":
        return knowledge_composer_fragment_ids(ctx)
    if ctx.stage == "smalltalk_composer":
        return smalltalk_composer_fragment_ids(ctx)
    raise ValueError(f"unsupported agent prompt stage: {ctx.stage}")


def _model_fragment(model_family: ModelFamily) -> str:
    if model_family in {"ds_v4pro", "deepseek"}:
        return "model.ds_v4pro.structured"
    return "model.generic.structured"


def _named_strategy_count(
    request: ReplyRequest,
    canonical_context: CanonicalContext,
) -> int:
    del request
    if canonical_context.strategy_status == "resolved":
        return 1
    if canonical_context.strategy_status == "ambiguous":
        return len(canonical_context.strategy_candidates)
    return 0
