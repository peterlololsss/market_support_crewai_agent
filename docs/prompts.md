# Prompt Registry And Assembly

Last updated: 2026-06-17.

Prompts are registered in `src/market_support_crewai_agent/runtime/llm/prompting/registry.py` and assembled by `PromptAssembler` in `src/market_support_crewai_agent/runtime/llm/prompting/assembler.py`.

## Layers

Prompt assembly emits layers in this order:

```text
stable -> domain -> runtime -> task -> ephemeral
```

- `stable`: agent identity, generic behavior, generic structured-output discipline, and style.
- `domain`: domain ontology, policy allowlists, capability registry summaries, and source/evidence rules.
- `runtime`: request metadata, current channel, material-pack routing options, available artifacts, recent turns, evidence facts, business facts, guardrail decisions, and answerability. This appears once as `Runtime Capability & Evidence Boundary JSON`.
- `task`: current user request, selected stage output schema, candidate response for verifier stages, and stage-specific task prompt.
- `ephemeral`: retry/self-correction state such as previous verifier verdicts. This layer is valid only for the current regeneration attempt.

Stable prompt content must not be mutated mid-request. Production runtime differences enter through `ModelVisibleContext` on `PromptAssemblyContext`; selector/guardrail prompts use their explicit JSON payloads.

## Section And Role Responsibilities

Planner prompts are intentionally split across multiple sections. Each section has
one job, so future fixes go to the right source instead of stuffing eval examples
into active instructions.

```text
base.planner_intent
  Planner identity, PlanSpec-only output boundary, source-of-truth hierarchy,
  request splitting, and deterministic harness ownership.

planner.intent_taxonomy
  Durable semantic categories only: send request, current metric/report evidence,
  evergreen document FAQ, report-scope question, material-pack collateral,
  handoff, clarification, refusal/abstention, smalltalk/no-reply.

CapabilityManifest in runtime/domain/capabilities/manifests.py
  Capability contract source: capability id, artifact/tool boundaries,
  EvidenceContract, abstention policy, verifier checks, and compact
  planner_guidance. This is where capability-specific selection boundaries live.

Capability registry JSON in Runtime Capability & Evidence Boundary JSON
  Runtime projection of manifest-derived capability contracts. It is the selectable
  allowlist for the current request and must omit examples_positive,
  examples_negative, and eval-question text.

ContextProjectionManager / ModelVisibleContext
  Runtime state projection: request metadata, policy allowlists, ledger/history
  context, runtime clock, guardrail decisions, evidence facts, business facts,
  and pressure decisions. It should expose facts and contracts, not new rules.

output.plan_spec_schema
  Output schema and field-level PlanSpec constraints. It should not carry
  domain routing examples.

compliance.reason_codes
  Allowed compliance reason-code vocabulary. It explains labels, not final
  legality; deterministic validators still enforce action legality.

composer and verifier fragments
  Composer fragments format grounded replies from BusinessFacts/evidence.
  Verifier fragments inspect an already proposed reply/action. They do not add
  planner routing policy.
```

Anti-overfitting rule: real Xiaoyan question-set strings belong in eval fixtures,
golden cases, or regression tests. Active prompt fragments and projected manifest
contracts should describe target behavior by category and schema.

## Failure Mode Covered By Prompts

The original architecture bug was source drift: a `材料包` product question could
be answered from `周报` or `月报` product evidence because the prompt only said to
"use evidence" broadly. Prompts now receive capability cards, evidence contracts,
guardrail decisions, and `AnswerabilityAssessment` in the runtime layer.

Prompt text may explain the boundary, but it is not the authority. The authority
is:

```text
CapabilityManifest -> EvidenceContract -> AnswerabilityGate -> direct guard functions
```

Weekly/monthly reports cannot satisfy material-pack evidence unless the selected
manifest explicitly allows fallback.

## Assembly APIs

Use these assembler methods rather than concatenating prompt strings in arbitrary modules:

```text
assemblePlannerPrompt(ctx)
assembleAgentPrompt(ctx)
assembleVerifierPrompt(ctx)
assembleCanonicalizationPrompt(prompt_id, stage=..., selector_input_json=...)
assembleGuardrailPrompt(prompt_id, stage=..., verifier_input_json=...)
```

Snake-case aliases exist for Python call sites.

## Prompt IDs

Registry fragment IDs:

```text
base.planner_intent
base.knowledge_composer
base.smalltalk_composer
base.direct_composer
base.action_feedback_composer
base.alignment_verifier
model.ds_v4pro.structured
model.generic.structured
planner.intent_taxonomy
compliance.reason_codes
evidence.document_grounding
style.wecom_concise_zh
output.plan_spec_schema
output.reply_response_no_actions
output.reply_alignment_verdict_schema
output.direct_composer_schema
canonicalization.document_product_selector
canonicalization.approved_knowledge_selector
guardrail.image_alignment_verifier
```

Registry agent descriptor IDs:

```text
agent.planner
agent.composer
agent.direct_composer
agent.alignment_verifier
agent.document_product_selector
agent.approved_knowledge_selector
agent.image_alignment_verifier
```

## Extension Rules

Capability additions should normally change:

```text
src/market_support_crewai_agent/runtime/domain/capabilities/manifests.py
```

Add or change prompt fragments only when the generic stage contract changes. Do not add capability-specific hierarchy, source precedence, artifact-routing, product-selection, report-scope, or channel-specific business rules to arbitrary code comments or prompt files.

Runtime capability and evidence state belongs in `PromptAssemblyContext` and appears once in the runtime boundary block. Evidence source rules should come from `CapabilityManifest`, `DomainContext`, `EvidenceContract`, `GuardrailDecision`, or deterministic validators before they become prompt prose.

## Context Is A Projection

`PromptAssemblyContext` should not be used in production without
`model_visible_context`. Future runtime state, source data, or evidence should
be added to `ContextProjectionManager`, not directly to prompt fragments or
ad hoc prompt rendering.

`ModelVisibleContext` is the model-call view of durable/current state:

```text
durable/current source-of-truth -> ContextProjectionManager -> ModelVisibleContext -> PromptAssembler
```

Block meanings:

```text
recent_verbatim        recent transcript as context-only
compacted_summary      deterministic older-history summary
large_result_preview   preview + reload_handle for large evidence/history
allowed_evidence       current-plan evidence selected by retrieval_source_guard/select_evidence_for_plan
context_only           helpful state that cannot ground claims by itself
disallowed_evidence    rejected source metadata with content redacted
app_state              request/canonical/domain/policy/plan/facts/guardrails
```

Conversation history is context only unless an evidence contract explicitly
allows history and deterministic guards accept its scope/provenance. Large
content is previewed with source/type/size metadata instead of arbitrary prompt
truncation. Context pressure metadata is attached to `PromptProgram` and audit
without logging full prompts or full payloads.

Add a prompt fragment only when a generic stage contract changes. Do not add a
`材料包`, `周报`, `月报`, `渠道`, `策略`, or `产品` rule directly to a stage prompt
when it can be represented as manifest metadata or domain context.

Snapshot tests live under `tests/snapshots/prompts/`. Update them intentionally with:

```bash
UPDATE_PROMPT_SNAPSHOTS=1 uv run --extra dev python -m pytest -q tests/unit/llm/test_prompt_snapshots.py
```

Run the registry lint before shipping prompt changes:

```bash
uv run python scripts/check_prompt_registry.py
```
