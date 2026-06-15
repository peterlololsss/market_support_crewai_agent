from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from string import Template

from market_support_crewai_agent.runtime.llm.prompt_profiles import PromptStage


@dataclass(frozen=True)
class PromptFragment:
    id: str
    stage: PromptStage
    priority: int
    template_name: str
    required: bool = False
    conflict_tags: frozenset[str] = frozenset()
    token_budget_hint: int | None = None


PROMPT_FRAGMENT_PACKAGE = "market_support_crewai_agent.runtime.llm.prompts.fragments"


PROMPT_FRAGMENTS: tuple[PromptFragment, ...] = (
    PromptFragment(
        id="base.planner_intent",
        stage="planner_intent",
        priority=10,
        template_name="base/planner_intent_base.md",
        required=True,
    ),
    PromptFragment(
        id="base.knowledge_composer",
        stage="knowledge_composer",
        priority=10,
        template_name="base/knowledge_composer_base.md",
        required=True,
    ),
    PromptFragment(
        id="base.smalltalk_composer",
        stage="smalltalk_composer",
        priority=10,
        template_name="base/smalltalk_composer_base.md",
        required=True,
    ),
    PromptFragment(
        id="base.alignment_verifier",
        stage="alignment_verifier",
        priority=10,
        template_name="base/alignment_verifier_base.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="planner_intent",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="knowledge_composer",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="smalltalk_composer",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="alignment_verifier",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="planner_intent",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="knowledge_composer",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="smalltalk_composer",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="alignment_verifier",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="output.intent_frame_schema",
        stage="planner_intent",
        priority=30,
        template_name="output/intent_frame_schema.md",
        required=True,
    ),
    PromptFragment(
        id="planner.intent_taxonomy",
        stage="planner_intent",
        priority=35,
        template_name="planner/intent_taxonomy.md",
        required=True,
    ),
    PromptFragment(
        id="output.reply_response_no_actions",
        stage="knowledge_composer",
        priority=30,
        template_name="output/reply_response_no_actions.md",
        required=True,
    ),
    PromptFragment(
        id="output.reply_response_no_actions",
        stage="smalltalk_composer",
        priority=30,
        template_name="output/reply_response_no_actions.md",
        required=True,
    ),
    PromptFragment(
        id="output.reply_alignment_verdict_schema",
        stage="alignment_verifier",
        priority=30,
        template_name="output/reply_alignment_verdict_schema.md",
        required=True,
    ),
    PromptFragment(
        id="compliance.reason_codes",
        stage="planner_intent",
        priority=40,
        template_name="compliance/reason_codes.md",
        required=True,
    ),
    PromptFragment(
        id="compliance.refusal_examples",
        stage="planner_intent",
        priority=80,
        template_name="compliance/refusal_examples.md",
    ),
    PromptFragment(
        id="capability.material_pack",
        stage="planner_intent",
        priority=100,
        template_name="capability/material_pack.md",
        conflict_tags=frozenset({"artifact_capability"}),
    ),
    PromptFragment(
        id="capability.weekly_report",
        stage="planner_intent",
        priority=100,
        template_name="capability/weekly_report.md",
        conflict_tags=frozenset({"artifact_capability"}),
    ),
    PromptFragment(
        id="capability.monthly_report",
        stage="planner_intent",
        priority=100,
        template_name="capability/monthly_report.md",
        conflict_tags=frozenset({"artifact_capability"}),
    ),
    PromptFragment(
        id="capability.document_context",
        stage="planner_intent",
        priority=100,
        template_name="capability/document_context.md",
    ),
    PromptFragment(
        id="capability.sales_handoff",
        stage="planner_intent",
        priority=100,
        template_name="capability/sales_handoff.md",
    ),
    PromptFragment(
        id="channel.bank_material_rules",
        stage="planner_intent",
        priority=120,
        template_name="channel/bank_material_rules.md",
    ),
    PromptFragment(
        id="examples.material_pack",
        stage="planner_intent",
        priority=200,
        template_name="examples/material_pack.md",
    ),
    PromptFragment(
        id="examples.report_scope",
        stage="planner_intent",
        priority=200,
        template_name="examples/report_scope.md",
    ),
    PromptFragment(
        id="examples.knowledge_answer",
        stage="planner_intent",
        priority=200,
        template_name="examples/knowledge_answer.md",
    ),
    PromptFragment(
        id="examples.handoff",
        stage="planner_intent",
        priority=200,
        template_name="examples/handoff.md",
    ),
    PromptFragment(
        id="examples.smalltalk",
        stage="planner_intent",
        priority=200,
        template_name="examples/smalltalk.md",
    ),
    PromptFragment(
        id="examples.multi_artifact_clarification",
        stage="planner_intent",
        priority=200,
        template_name="examples/multi_artifact_clarification.md",
    ),
    PromptFragment(
        id="evidence.document_grounding",
        stage="knowledge_composer",
        priority=100,
        template_name="evidence/document_grounding.md",
        required=True,
    ),
    PromptFragment(
        id="style.wecom_concise_zh",
        stage="knowledge_composer",
        priority=110,
        template_name="style/wecom_concise_zh.md",
        required=True,
    ),
    PromptFragment(
        id="style.wecom_concise_zh",
        stage="smalltalk_composer",
        priority=110,
        template_name="style/wecom_concise_zh.md",
        required=True,
    ),
)


def fragment_by_id(fragment_id: str, stage: PromptStage | None = None) -> PromptFragment:
    matches = [
        fragment
        for fragment in PROMPT_FRAGMENTS
        if fragment.id == fragment_id and (stage is None or fragment.stage == stage)
    ]
    if not matches:
        raise ValueError(f"Unknown prompt fragment: {fragment_id}")
    if len(matches) > 1 and stage is None:
        raise ValueError(f"Prompt fragment id needs stage disambiguation: {fragment_id}")
    return matches[0]


def render_prompt_fragment(
    fragment_id: str,
    stage: PromptStage,
    **context: str,
) -> str:
    fragment = fragment_by_id(fragment_id, stage)
    return Template(load_prompt_fragment_text(fragment)).safe_substitute(context).strip()


def load_prompt_fragment_text(fragment: PromptFragment) -> str:
    if (
        fragment.template_name.startswith("/")
        or ".." in fragment.template_name.split("/")
        or not fragment.template_name.endswith(".md")
    ):
        raise ValueError(f"Invalid prompt fragment template: {fragment.template_name}")
    resource = files(PROMPT_FRAGMENT_PACKAGE).joinpath(fragment.template_name)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown prompt fragment template: {fragment.template_name}") from exc
