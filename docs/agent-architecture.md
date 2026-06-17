# Agent Architecture

Last updated: 2026-06-17.

This service is a deterministic support-reply harness around bounded LLM stages.
The LLM proposes intent and wording. The harness owns source scope, evidence,
policy, validation, audit, and outbound action safety.

## Original Failure Mode

The architecture review found one recurring bug class: a question about `材料包`
could be answered from unrelated `周报` or `月报` evidence because the old path
treated "some knowledge evidence exists" as enough to compose.

The current architecture prevents that by making answerability source-specific:

```text
材料包 product question -> material_pack.product_list -> material_pack evidence only
周报 performance question -> weekly_report.product_performance -> weekly_report evidence only
月报 performance question -> monthly_report.product_performance -> monthly_report evidence only
```

Weekly/monthly reports cannot satisfy `material_pack` evidence unless the
selected `CapabilityManifest.evidence_contract` explicitly allows that fallback.
The built-in material-pack manifests do not allow it.

## Current Runtime

```text
ReplyRequest
-> ConversationStore.get_recent + ActionLedger.recent_executed_for_conversation
-> DomainContextBuilder
-> CanonicalEntityResolver
-> compile_policy
-> ContextProjectionManager.project_for_stage
-> Planner LLM emits PlanSpec
-> compile_plan_spec -> ExecutionPlan
-> validate_execution_plan
-> EvidenceExecutor
-> EvidenceFact + BusinessFacts
-> AnswerabilityGate
-> ContextProjectionManager.project_for_stage
-> DecisionEngine / ResponseDirective
-> deterministic renderer or bounded composer
-> output_guard + validate_reply
-> ContextProjectionManager.project_for_stage when verifier runs
-> optional alignment verifier
-> audit + ReplyResponse
```

Only `ReplyResponse { reply, actions }` crosses the public `/reply` boundary.
The adapter still has final authority to execute `send_material_pack`,
`send_weekly_report`, and `send_monthly_report`.

## Capability Registry

Capabilities are declared in:

```text
src/market_support_crewai_agent/runtime/domain/capabilities/manifests.py
```

The registry entry is the source for planner cards, composer guidance, evidence
contracts, and generic verifier checks. It declares:

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

Add capability metadata here first. Do not scatter capability rules into prompt
fragments, orchestration branches, or verifier-specific functions.

## DomainContext

`DomainContextBuilder` builds the current business scope:

```text
渠道 DistributionChannel
策略 Strategy
产品 Product
材料包 / 周报 / 月报 Artifact
ArtifactScope(channel, strategy, product, time_range)
```

`CanonicalEntityResolver` resolves mentions only against explicit domain
entities, exact names/aliases, structured defaults, or bounded closed-set
selectors. Unknown or ambiguous 产品/策略 mentions stay unresolved or ambiguous;
the harness clarifies or abstains instead of nearest matching.

## PlanSpec

The planner emits `PlanSpec`, not a final answer. A `PlanSpec` selects one
capability and declares:

```text
selected_capability_id
domain_scope
required_artifacts / allowed_artifacts / forbidden_artifacts
required_tools
answerability_policy
output_schema_ref
evidence_contract_ref or inline evidence_contract
steps
abstention_cases
```

`compile_plan_spec` converts this to `ExecutionPlan`. The runtime validates the
compiled plan before executing evidence wrappers.

## EvidenceContract

`EvidenceContract` is the source boundary:

```text
required_fact_types / any_of_fact_types
allowed_source_types / forbidden_source_types
required_artifact_types / allowed_artifact_types
required_scope_match
allow_history
fallback_policy
provenance_required
```

Examples:

- `material_pack.product_list` requires `material_pack_product_list` from
  `adapter_material_pack_content` and forbids `adapter_report_scope`.
- `weekly_report.product_performance` uses `weekly_report` report evidence.
- `monthly_report.product_performance` uses `monthly_report` report evidence.
- History evidence is rejected unless `allow_history=true` and scope/provenance
  checks pass.

## Guard Functions

Use the phase-specific guard function directly:

```text
input_guard              SendScope and requested destination/scope checks
retrieval_source_guard   evidence-source, artifact, channel, strategy, history checks
execution_tool_guard     fixed-wrapper/tool and artifact-id checks
output_guard             composer evidence-id and source-scope checks
```

Guardrails return machine-readable reason codes. They do not repair unsafe model
output silently.

## AnswerabilityGate

`AnswerabilityGate` runs after evidence execution and before composition. It
decides whether the selected capability can answer from current evidence:

```text
answer   -> composer/renderer may answer
clarify  -> ask for missing 策略, 渠道, 产品, or artifact scope
abstain  -> state missing evidence and do not compose from another source
```

This is the direct fix for the material-pack/report source-mixing bug.

## Prompt Assembly Layers

Prompt assembly uses one ordered boundary:

```text
stable -> domain -> runtime -> task -> ephemeral
```

- `stable`: role, concise WeCom style, structured-output discipline.
- `domain`: capability cards, domain model, policy allowlists.
- `runtime`: current request, DomainContext, CanonicalContext, evidence facts,
  BusinessFacts, guardrails, answerability.
- `task`: current stage schema and instruction.
- `ephemeral`: retry/alignment state for the current attempt only.

Prompt snapshot tests lock this boundary. Capability changes should update
manifest cards and snapshots intentionally, not ad hoc prompt text.

## Context Is A Projection

A transcript records what happened. `ModelVisibleContext` decides what matters
for the current model call. Production planner, composer, and verifier prompts
must receive a `ModelVisibleContext` from
`ContextProjectionManager.project_for_stage(...)`; do not feed raw transcript,
ledger, or evidence directly to prompt rendering.

Projection block types:

```text
recent_verbatim        last few conversation turns, context-only
compacted_summary      deterministic summary of older turns, not user/assistant role text
large_result_preview   bounded preview with reload_handle for oversized content
allowed_evidence       evidence accepted by retrieval_source_guard/select_evidence_for_plan
context_only           useful state that is not claim evidence, including history
disallowed_evidence    rejected evidence metadata with content redacted
app_state              request metadata, canonical/domain/policy/plan/facts/guardrails
current_task           current user message and verifier candidate response
output_schema          stage output boundary when projected
ephemeral              retry/alignment attempt state
```

Conversation history is not claim evidence by default. Allowed evidence and
disallowed/context-only evidence are separated before prompt assembly, and
oversized evidence/history content enters prompts only as stable previews with
reload handles. Context pressure estimates are recorded on the prompt program
for audit; a hard over-budget projection fails before the model call.

## Migration Notes

When moving old behavior into the new architecture:

```text
prompt rule -> CapabilityManifest or EvidenceContract
keyword selector -> DomainContext entity or closed-set selector
custom verifier branch -> generic verifier primitive
LLM fact inference -> EvidenceFact / BusinessFacts
send text -> typed action proposal, adapter owns execution copy
history fact -> allow_history=true plus scope/provenance checks
```

Delete the old path once tests prove there is no active caller. A bridge needs a
published external boundary, an active-caller test, or an ADR.
