from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from string import Template
from typing import Literal

from market_support_crewai_agent.runtime.llm.prompting.profiles import PromptStage

PromptLayer = Literal["stable", "domain", "runtime", "task", "ephemeral"]


@dataclass(frozen=True)
class PromptFragment:
    id: str
    stage: PromptStage
    layer: PromptLayer
    priority: int
    template_name: str
    required: bool = False
    conflict_tags: frozenset[str] = frozenset()
    token_budget_hint: int | None = None


@dataclass(frozen=True)
class PromptAgentSpec:
    id: str
    role: str
    goal: str
    backstory: str


PROMPT_FRAGMENT_PACKAGE = "market_support_crewai_agent.runtime.llm.prompts.fragments"

PROMPT_LAYER_ORDER: tuple[PromptLayer, ...] = (
    "stable",
    "domain",
    "runtime",
    "task",
    "ephemeral",
)

PROMPT_FRAGMENTS: tuple[PromptFragment, ...] = (
    PromptFragment(
        id="base.planner_intent",
        stage="planner_intent",
        layer="stable",
        priority=10,
        template_name="base/planner_intent_base.md",
        required=True,
    ),
    PromptFragment(
        id="base.knowledge_composer",
        stage="knowledge_composer",
        layer="stable",
        priority=10,
        template_name="base/knowledge_composer_base.md",
        required=True,
    ),
    PromptFragment(
        id="base.smalltalk_composer",
        stage="smalltalk_composer",
        layer="stable",
        priority=10,
        template_name="base/smalltalk_composer_base.md",
        required=True,
    ),
    PromptFragment(
        id="base.direct_composer",
        stage="direct_composer",
        layer="stable",
        priority=10,
        template_name="base/direct_composer_base.md",
        required=True,
    ),
    PromptFragment(
        id="base.alignment_verifier",
        stage="alignment_verifier",
        layer="stable",
        priority=10,
        template_name="base/alignment_verifier_base.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="planner_intent",
        layer="stable",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="knowledge_composer",
        layer="stable",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="smalltalk_composer",
        layer="stable",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="direct_composer",
        layer="stable",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.ds_v4pro.structured",
        stage="alignment_verifier",
        layer="stable",
        priority=20,
        template_name="model/ds_v4pro_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="planner_intent",
        layer="stable",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="knowledge_composer",
        layer="stable",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="smalltalk_composer",
        layer="stable",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="direct_composer",
        layer="stable",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="model.generic.structured",
        stage="alignment_verifier",
        layer="stable",
        priority=20,
        template_name="model/generic_structured.md",
        required=True,
    ),
    PromptFragment(
        id="planner.intent_taxonomy",
        stage="planner_intent",
        layer="domain",
        priority=35,
        template_name="planner/intent_taxonomy.md",
        required=True,
    ),
    PromptFragment(
        id="compliance.reason_codes",
        stage="planner_intent",
        layer="domain",
        priority=40,
        template_name="compliance/reason_codes.md",
        required=True,
    ),
    PromptFragment(
        id="evidence.document_grounding",
        stage="knowledge_composer",
        layer="domain",
        priority=100,
        template_name="evidence/document_grounding.md",
        required=True,
    ),
    PromptFragment(
        id="style.wecom_concise_zh",
        stage="knowledge_composer",
        layer="stable",
        priority=110,
        template_name="style/wecom_concise_zh.md",
        required=True,
    ),
    PromptFragment(
        id="style.wecom_concise_zh",
        stage="smalltalk_composer",
        layer="stable",
        priority=110,
        template_name="style/wecom_concise_zh.md",
        required=True,
    ),
    PromptFragment(
        id="style.wecom_concise_zh",
        stage="direct_composer",
        layer="stable",
        priority=110,
        template_name="style/wecom_concise_zh.md",
        required=True,
    ),
    PromptFragment(
        id="output.plan_spec_schema",
        stage="planner_intent",
        layer="task",
        priority=30,
        template_name="output/plan_spec_schema.md",
        required=True,
    ),
    PromptFragment(
        id="output.reply_response_no_actions",
        stage="knowledge_composer",
        layer="task",
        priority=30,
        template_name="output/reply_response_no_actions.md",
        required=True,
    ),
    PromptFragment(
        id="output.reply_response_no_actions",
        stage="smalltalk_composer",
        layer="task",
        priority=30,
        template_name="output/reply_response_no_actions.md",
        required=True,
    ),
    PromptFragment(
        id="output.direct_composer_schema",
        stage="direct_composer",
        layer="task",
        priority=30,
        template_name="output/direct_composer_schema.md",
        required=True,
    ),
    PromptFragment(
        id="output.reply_alignment_verdict_schema",
        stage="alignment_verifier",
        layer="task",
        priority=30,
        template_name="output/reply_alignment_verdict_schema.md",
        required=True,
    ),
    PromptFragment(
        id="canonicalization.document_product_selector",
        stage="document_product_selector",
        layer="task",
        priority=10,
        template_name="canonicalization/document_product_selector.md",
        required=True,
    ),
    PromptFragment(
        id="canonicalization.approved_knowledge_selector",
        stage="approved_knowledge_selector",
        layer="task",
        priority=10,
        template_name="canonicalization/approved_knowledge_selector.md",
        required=True,
    ),
    PromptFragment(
        id="guardrail.image_alignment_verifier",
        stage="image_alignment_verifier",
        layer="task",
        priority=10,
        template_name="guardrail/image_alignment_verifier.md",
        required=True,
    ),
)

PROMPT_AGENT_SPECS: tuple[PromptAgentSpec, ...] = (
    PromptAgentSpec(
        id="agent.planner",
        role="Market Support Reply Planner",
        goal=(
            "Interpret Chinese sales/support requests, evaluate compliance, and "
            "return a bounded PlanSpec for the deterministic harness."
        ),
        backstory=(
            "You plan the support workflow for Shanghai Yanfu Investment. You do "
            "not call tools, send messages, or produce final business facts."
        ),
    ),
    PromptAgentSpec(
        id="agent.composer",
        role="Market Support Reply Composer",
        goal=(
            "Compose the final ReplyResponse from a validated plan and "
            "deterministic evidence for the external WeWork adapter."
        ),
        backstory=(
            "You are the external agent brain for a market support workflow. You "
            "use the validated plan and evidence facts."
        ),
    ),
    PromptAgentSpec(
        id="agent.direct_composer",
        role="WeCom Direct Message Composer",
        goal=(
            "Handle one DM using only grounded company information or the "
            "adapter-owned prepared outbound lifecycle."
        ),
        backstory=(
            "You serve individual WeCom messages. You never widen permissions, "
            "invent adapter references, or bypass confirmation."
        ),
    ),
    PromptAgentSpec(
        id="agent.alignment_verifier",
        role="Market Support Reply Alignment Verifier",
        goal=(
            "Judge whether the validated ReplyResponse semantically aligns with "
            "the current market support request."
        ),
        backstory=(
            "You are a bounded verifier. You return only a structured verdict and "
            "never call tools, send messages, or mutate actions."
        ),
    ),
    PromptAgentSpec(
        id="agent.document_product_selector",
        role="Closed-set Document MCP Selector",
        goal="Select only valid candidate document IDs for evidence retrieval.",
        backstory=(
            "You specialize in choosing evidence sources from a fixed catalog. "
            "You never compose customer replies, invent IDs, or call tools."
        ),
    ),
    PromptAgentSpec(
        id="agent.approved_knowledge_selector",
        role="Approved Static Knowledge Semantic Selector",
        goal="Select only closed-set approved knowledge IDs for the harness.",
        backstory=(
            "You choose from a small vetted catalog. You do not compose replies, "
            "invent IDs, or call tools."
        ),
    ),
    PromptAgentSpec(
        id="agent.image_alignment_verifier",
        role="Final Image Alignment Verifier",
        goal="Judge image-marker semantic alignment for a validated support reply.",
        backstory=(
            "You are a bounded verifier. You do not compose the reply and you do "
            "not choose image assets."
        ),
    ),
)


class PromptRegistry:
    def __init__(
        self,
        fragments: tuple[PromptFragment, ...] = PROMPT_FRAGMENTS,
        agent_specs: tuple[PromptAgentSpec, ...] = PROMPT_AGENT_SPECS,
    ) -> None:
        self._fragments = fragments
        self._agent_specs = {spec.id: spec for spec in agent_specs}

    def fragment_by_id(
        self,
        fragment_id: str,
        stage: PromptStage | None = None,
    ) -> PromptFragment:
        matches = [
            fragment
            for fragment in self._fragments
            if fragment.id == fragment_id and (stage is None or fragment.stage == stage)
        ]
        if not matches:
            raise ValueError(f"Unknown prompt fragment: {fragment_id}")
        if len(matches) > 1 and stage is None:
            raise ValueError(f"Prompt fragment id needs stage disambiguation: {fragment_id}")
        return matches[0]

    def fragments(self) -> tuple[PromptFragment, ...]:
        return self._fragments

    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(fragment.id for fragment in self._fragments))

    def agent_spec_by_id(self, agent_id: str) -> PromptAgentSpec:
        try:
            return self._agent_specs[agent_id]
        except KeyError as exc:
            raise ValueError(f"Unknown prompt agent spec: {agent_id}") from exc

    def agent_spec_ids(self) -> tuple[str, ...]:
        return tuple(self._agent_specs)


PROMPT_REGISTRY = PromptRegistry()


def fragment_by_id(fragment_id: str, stage: PromptStage | None = None) -> PromptFragment:
    return PROMPT_REGISTRY.fragment_by_id(fragment_id, stage)


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


def prompt_agent_spec_by_id(agent_id: str) -> PromptAgentSpec:
    return PROMPT_REGISTRY.agent_spec_by_id(agent_id)
