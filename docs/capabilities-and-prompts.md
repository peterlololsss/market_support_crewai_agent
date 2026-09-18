# Capabilities And Prompts

Last updated: 2026-07-20.

Capabilities define what the planner may ask the runtime to do. Prompts expose those capabilities and current facts to bounded LLM stages. Business rules belong in manifests, schemas, validators, and tests before prompt wording.

## Capability Registry

Capability metadata lives in:

```text
src/market_support_crewai_agent/runtime/policy/capabilities/manifests.py
```

A capability entry describes:

```text
id
runtime_capability
required_inputs
required_artifacts / allowed_artifacts / forbidden_artifacts
required_tools
output_schema
evidence_contract
abstention_policy
verifier_checks
examples_positive / examples_negative
```

The registry feeds planner cards, composer guidance, evidence contracts, and generic verifier checks. Do not scatter capability policy into unrelated prompt fragments, orchestration branches, or one-off verifier functions.

## PlanSpec Boundary

The planner emits `PlanSpec`, not a final answer. Each `plan_units[]` item selects a capability and declares the scope, tools, evidence expectations, output schema, and abstention cases for that unit.

`finalize_execution_plan_v2` converts `PlanSpec` into `ExecutionPlanV2`. The runtime validates the finalized plan before evidence execution. Capability-owned evidence contracts win over planner-supplied inline hints and cannot be loosened by model output.

## Scene Policy

The public wire uses `identity.scene="direct"` and `identity.scene="group"`. `internal_dm` and `external_group` are
operator-facing prose aliases only; they are not prompt-selected or accepted as wire values.

Internal DM has a deterministic ceiling:

- Business scope is exactly `{"kind":"unscoped"}`.
- Read grants are either empty or exactly `["query_internal_company_info"]`. The exact grant enables the unified
  internal-company-knowledge gateway; an empty list performs no internal knowledge provider or selector work.
- Approved-static text and Document MCP text may become post-plan evidence only when that grant and a validated knowledge
  plan are both present.
- Recall is fixed at `recall_mode="off"`.
- The scene has no actions, mentions, media, or adapter business resolves. Approved image markers, arbitrary image
  markers, media bindings, send/report/material proposals, and sales handoffs all fail deterministic validation.

External groups retain distribution scope, explicit principal grants, principal-scoped history/state, adapter
resolve/preflight, typed action proposals, sales-mention validation, and feedback/idempotency. The WeCom adapter remains
the final authority for validating, authorizing, executing, and recording the primary reply and proposed actions.

Prompt output cannot widen either scene. Scene policy comes from the request schema, policy manifest, evidence admission,
and final reply/action validators; planner and composer text remains a proposal inside those boundaries.

## Adding A Capability

Use this workflow when a new support behavior is needed:

1. Add or update the capability manifest.
2. Define required inputs, allowed artifacts, source types, and fallback behavior.
3. Add evidence execution support only through deterministic wrappers.
4. Update structural prompt-program tests if planner/composer-visible cards change.
5. Add focused tests for manifest validation, plan compilation, evidence boundaries, and reply/action validation.
6. Run the narrowest relevant eval or contract suite from `safety-and-evals.md`.

Avoid bespoke verifier code unless the generic manifest checks cannot express the invariant. If a new invariant is reusable, add it to the manifest/verifier model instead of a capability-specific branch.

## Prompt Assembly

Prompt assembly uses ordered layers:

```text
stable -> domain -> runtime -> task -> ephemeral
```

- `stable`: role, concise WeCom style, structured-output discipline.
- `domain`: capability cards, domain model, policy allowlists.
- `runtime`: the current request plus policy-scoped, canonical plan, evidence, grounding, business-fact, and guardrail views for that role.
- `task`: current stage schema and instruction.
- `ephemeral`: retry or alignment state for the current attempt only.

Production planner, composer, and verifier calls receive `PlannerPromptInputV1`, `KnowledgeComposerPromptInputV1` or `SmalltalkComposerPromptInputV1`, and `AlignmentVerifierPromptInputV1`, respectively. `DomainContextV1` remains deterministic runtime/evidence metadata and is not model authority. Do not feed raw request identity, transcript, ledger, policy, plan, or evidence objects directly into prompt rendering.

## Prompt Change Rules

- Keep long product, document, and report-scope lists out of default prompts.
- Prefer compact summaries, exact-match results, or explicit pagination.
- Use allowlists and canonical schema descriptions in active instructions.
- Keep obsolete rejected field names and historical anti-patterns out of routine prompts.
- Test registered IDs, layers, schemas, budgets, and observable behavior; do not bless rendered prompt prose snapshots.

## Prompt IDs

Registry coverage is enforced by `scripts/check_prompt_registry.py`. Keep these IDs documented when adding, removing, or renaming prompt fragments or agent specs.

Prompt fragments:

```text
base.planner_intent
base.knowledge_composer
base.smalltalk_composer
base.alignment_verifier
model.ds_v4pro.structured
model.generic.structured
instruction.registered_over_untrusted_data.v1
planner.intent_taxonomy
compliance.reason_codes
evidence.document_grounding
style.wecom_concise_zh
output.plan_spec_schema
output.reply_response_no_actions
output.reply_alignment_verdict_schema
canonicalization.document_product_selector
canonicalization.approved_knowledge_selector
scene.wecom_group.planner_intent.v1
scene.wecom_direct.planner_intent.v1
scene.wecom_group.knowledge_composer.v1
scene.wecom_direct.knowledge_composer.v1
scene.wecom_group.smalltalk_composer.v1
scene.wecom_direct.smalltalk_composer.v1
scene.wecom_group.alignment_verifier.v1
scene.wecom_direct.alignment_verifier.v1
```

Agent specs:

```text
agent.planner
agent.composer
agent.alignment_verifier
agent.document_product_selector
agent.approved_knowledge_selector
agent.llm_health_probe
```

## Related Code

```text
src/market_support_crewai_agent/runtime/policy/       capability and policy metadata
src/market_support_crewai_agent/runtime/planning/     PlanSpec and plan compilation
src/market_support_crewai_agent/runtime/prompts/      prompt registry, profiles, fragments, assembly
src/market_support_crewai_agent/runtime/context/      model-visible context projection
src/market_support_crewai_agent/runtime/validation/   plan/output/reply validators
```
