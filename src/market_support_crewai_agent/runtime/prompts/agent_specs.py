from __future__ import annotations

from typing import Final

from market_support_crewai_agent.runtime.prompts.registry_models import (
    PromptAgentSpec,
)

PROMPT_AGENT_SPECS: Final[tuple[PromptAgentSpec, ...]] = (
    PromptAgentSpec(
        "agent.planner",
        "Market Support Reply Planner",
        "Return a bounded PlanSpec for the deterministic harness.",
        "Interpret support requests without calling tools or producing final business facts.",
    ),
    PromptAgentSpec(
        "agent.composer",
        "Market Support Reply Composer",
        "Compose a validated reply from selected evidence.",
        "Use only the validated plan, directive, and admitted evidence.",
    ),
    PromptAgentSpec(
        "agent.alignment_verifier",
        "Market Support Reply Alignment Verifier",
        "Return only a bounded semantic alignment verdict.",
        "Never call tools, send messages, or mutate actions.",
    ),
    PromptAgentSpec(
        "agent.document_product_selector",
        "Closed-set Document MCP Selector",
        "Select only declared document candidate IDs.",
        "Never compose customer replies or invent IDs.",
    ),
    PromptAgentSpec(
        "agent.approved_knowledge_selector",
        "Approved Static Knowledge Semantic Selector",
        "Select only declared approved knowledge IDs.",
        "Never compose customer replies or invent IDs.",
    ),
    PromptAgentSpec(
        "agent.llm_health_probe",
        "LLM Health Probe",
        "Return only the registered health response object.",
        "Receive no request, identity, policy, evidence, or business data.",
    ),
)
