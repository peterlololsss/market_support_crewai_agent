# Capability Registry

The capability registry is the central metadata source for planner prompts, bounded
agent guidance, evidence contracts, and generic verifier checks.

Implementation files:

```text
src/market_support_crewai_agent/runtime/domain/capabilities/registry.py manifest schema and registry
src/market_support_crewai_agent/runtime/domain/plan_spec.py             planner/verifier PlanSpec schema
src/market_support_crewai_agent/runtime/domain/capabilities/manifests.py built-in manifests
src/market_support_crewai_agent/runtime/domain/capabilities/adapters.py planner/verifier adapters
src/market_support_crewai_agent/runtime/validation/capability_manifest_verifier.py generic checks
src/market_support_crewai_agent/runtime/validation/plan_spec_verifier.py generic PlanSpec/EvidenceContract checks
```

`CapabilitySpec` lookups in `runtime/domain/capabilities/__init__.py` describe the current
runtime policy, planning, adapter resolve, and action validation names. Capability
metadata should be added as `CapabilityManifest` entries, not as prompt-only rules.

## Manifest Fields

Each `CapabilityManifest` declares:

```text
id                    stable capability id, such as weekly_report.product_performance
version               manifest version
display_name          short human-readable label
description           concise capability purpose
capability_type       action, answer, summary, or handoff
domain_entities       involved entities, such as channel, strategy, product, artifact
required_inputs       runtime inputs that must be present
optional_inputs       useful but non-required runtime inputs
required_artifacts    artifact types that must ground the capability
allowed_artifacts     artifact types that may ground the capability
forbidden_artifacts   artifact types that must not ground the capability
required_tools        bounded wrappers or adapter commands required for evidence
output_schema         compact JSON-schema-like shape for the output being checked
evidence_contract     required evidence, source types, artifact types, scope matching, history/citation/provenance rules
abstention_policy     when the agent must abstain instead of composing
planner_guidance      compact planner-facing selection guidance; category-level, not eval-question prose
agent_guidance        bounded composer/agent guidance
verifier_checks       generic verifier primitive names
examples_positive     offline documentation or synthetic examples; not projected into planner prompts
examples_negative     offline documentation or synthetic contrasts; not projected into planner prompts
runtime_capability    current runtime capability name, when the manifest is gated by policy
```

`verifier_checks` must reference generic primitives only:

```text
output_schema
required_evidence_present
evidence_artifact_type_allowed
required_runtime_input_present
forbidden_source_not_used
abstention_correctness
```

Do not point a manifest at a capability-specific verifier function. Add a new
primitive only when the capability introduces a genuinely new kind of check that
cannot be represented with the existing primitives.

## Built-Ins

Built-in manifests live in `capabilities/manifests.py`. The initial manifest set includes:

```text
material_pack.send
weekly_report.send
monthly_report.send
sales.handoff
general.clarification
general.abstention
general.refusal
general.smalltalk
general.no_reply
material_pack.open_calendar
weekly_report.product_performance
monthly_report.product_performance
channel.strategy_summary
channel.product_summary
```

These are consumed through `CAPABILITY_MANIFEST_REGISTRY`. Planner prompt context
uses `planner_capability_cards(...)` and `ContextProjectionManager` projects the
result as compact `capability_contracts`. The projection includes id, type,
runtime capability, required/forbidden artifacts, required tools,
EvidenceContract fields, abstention guidance, verifier checks, and compact
planner guidance. It intentionally omits `examples_positive`,
`examples_negative`, and copied eval-question strings. Verifier code can use
`verify_capability_contracts(...)`, `verifier_manifest_contracts(...)`, or
`verify_plan_spec(...)` for the planner/verifier boundary.

## PlanSpec Boundary

The planner structured output is `PlanSpec`. It selects one capability manifest,
declares domain scope, artifacts, tools, answerability policy, output schema ref,
and an inline or referenced `EvidenceContract`. The runtime compiles `PlanSpec`
to `ExecutionPlan`; `PlanSpec` is the only planner contract accepted by the
runtime.

The generic PlanSpec verifier checks selected capability existence, required
artifact availability or valid abstention, forbidden step artifacts, output schema,
history restrictions, source/artifact restrictions, optional required scope match,
and evidence count/type/provenance requirements.

## Adding A Capability

1. Add a `CapabilityManifest` in `capabilities/manifests.py`.
2. Set `runtime_capability` when current policy should gate the manifest by an
   allowed runtime capability such as `weekly_report` or `document_context`.
3. Express evidence requirements in `evidence_contract` using evidence/fact types,
   allowed/disallowed source types, artifact types, `required_scope_match`,
   `minimum_evidence_count`, `allow_history`, fallback, citation, and provenance rules.
4. Choose only generic `verifier_checks`.
5. Keep `planner_guidance` category-level. Do not add every failed eval as a new
   sentence, and do not copy real eval questions into examples.
6. Add tests that register the manifest in a local `CapabilityRegistry`, call
   `planner_capability_cards(...)`, and validate success/failure with
   `verify_capability_contracts(...)`.

Adding a new capability should not require editing planner prompts or verifier
functions unless it needs a new primitive check or a new evidence wrapper/tool.
