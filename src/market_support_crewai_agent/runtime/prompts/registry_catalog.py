from __future__ import annotations

from typing import Final, TypeAlias

from market_support_crewai_agent.runtime.prompts.agent_specs import (
    PROMPT_AGENT_SPECS as PROMPT_AGENT_SPECS,
)
from market_support_crewai_agent.runtime.prompts.profiles import (
    PromptScene,
    PromptStage,
    UserFacingPromptStage,
)
from market_support_crewai_agent.runtime.prompts.registry_models import (
    PresentationRuleId,
    PromptFragment,
    PromptLayer,
)

PromptFragmentRow: TypeAlias = tuple[str, PromptStage, PromptLayer, int, str]


PROMPT_FRAGMENT_PACKAGE: Final = "market_support_crewai_agent.runtime.prompts.fragments"
PROMPT_LAYER_ORDER: Final[tuple[PromptLayer, ...]] = (
    "stable",
    "domain",
    "runtime",
    "task",
    "ephemeral",
)

_USER_FACING_STAGES: Final[tuple[UserFacingPromptStage, ...]] = (
    "planner_intent",
    "knowledge_composer",
    "smalltalk_composer",
    "alignment_verifier",
)
_PROMPT_SCENES: Final[tuple[PromptScene, ...]] = ("group", "direct")
_PRECEDENCE_STAGES: Final[tuple[PromptStage, ...]] = (
    *_USER_FACING_STAGES,
    "approved_knowledge_selector",
    "document_product_selector",
    "llm_health_probe",
)

_FRAGMENT_ROWS: Final[tuple[PromptFragmentRow, ...]] = (
    (
        "base.planner_intent",
        "planner_intent",
        "stable",
        10,
        "base/planner_intent_base.md",
    ),
    (
        "base.knowledge_composer",
        "knowledge_composer",
        "stable",
        10,
        "base/knowledge_composer_base.md",
    ),
    (
        "base.smalltalk_composer",
        "smalltalk_composer",
        "stable",
        10,
        "base/smalltalk_composer_base.md",
    ),
    (
        "base.alignment_verifier",
        "alignment_verifier",
        "stable",
        10,
        "base/alignment_verifier_base.md",
    ),
    *(
        (
            "model.ds_v4pro.structured",
            stage,
            "stable",
            20,
            "model/ds_v4pro_structured.md",
        )
        for stage in _USER_FACING_STAGES
    ),
    *(
        (
            "model.generic.structured",
            stage,
            "stable",
            20,
            "model/generic_structured.md",
        )
        for stage in _USER_FACING_STAGES
    ),
    *(
        (
            "instruction.registered_over_untrusted_data.v1",
            stage,
            "stable",
            25,
            "guardrail/registered_over_untrusted_data.md",
        )
        for stage in _PRECEDENCE_STAGES
    ),
    (
        "planner.intent_taxonomy",
        "planner_intent",
        "domain",
        35,
        "planner/intent_taxonomy.md",
    ),
    (
        "compliance.reason_codes",
        "planner_intent",
        "domain",
        40,
        "compliance/reason_codes.md",
    ),
    (
        "evidence.document_grounding",
        "knowledge_composer",
        "domain",
        100,
        "evidence/document_grounding.md",
    ),
    (
        "style.wecom_concise_zh",
        "knowledge_composer",
        "stable",
        110,
        "style/wecom_concise_zh.md",
    ),
    (
        "style.wecom_concise_zh",
        "smalltalk_composer",
        "stable",
        110,
        "style/wecom_concise_zh.md",
    ),
    (
        "output.plan_spec_schema",
        "planner_intent",
        "task",
        30,
        "output/plan_spec_schema.md",
    ),
    (
        "output.reply_response_no_actions",
        "knowledge_composer",
        "task",
        30,
        "output/reply_response_no_actions.md",
    ),
    (
        "output.reply_response_no_actions",
        "smalltalk_composer",
        "task",
        30,
        "output/reply_response_no_actions.md",
    ),
    (
        "output.reply_alignment_verdict_schema",
        "alignment_verifier",
        "task",
        30,
        "output/reply_alignment_verdict_schema.md",
    ),
    (
        "canonicalization.document_product_selector",
        "document_product_selector",
        "task",
        10,
        "canonicalization/document_product_selector.md",
    ),
    (
        "canonicalization.approved_knowledge_selector",
        "approved_knowledge_selector",
        "task",
        10,
        "canonicalization/approved_knowledge_selector.md",
    ),
)

_SCENE_FRAGMENT_ROWS: Final[tuple[PromptFragmentRow, ...]] = tuple(
    (
        f"scene.wecom_{scene}.{stage}.v1",
        stage,
        "stable",
        120,
        f"scene/wecom_{scene}/{stage}.md",
    )
    for stage in _USER_FACING_STAGES
    for scene in _PROMPT_SCENES
)

PROMPT_FRAGMENTS: Final[tuple[PromptFragment, ...]] = tuple(
    PromptFragment(
        id=fragment_id,
        stage=stage,
        layer=layer,
        priority=priority,
        template_name=template_name,
        required=True,
    )
    for fragment_id, stage, layer, priority, template_name in (
        *_FRAGMENT_ROWS,
        *_SCENE_FRAGMENT_ROWS,
    )
)

_GROUP_RULES: Final[tuple[PresentationRuleId, ...]] = (
    "address_group_audience",
    "use_optional_principal_name",
    "allow_current_conversation_label",
)
_DIRECT_RULES: Final[tuple[PresentationRuleId, ...]] = (
    "address_individual",
    "use_optional_principal_name",
    "forbid_group_addressing",
)
