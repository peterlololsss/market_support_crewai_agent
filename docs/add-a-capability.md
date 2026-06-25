# Add A Capability

Last updated: 2026-06-17.

Add capabilities manifest-first. If adding a capability requires custom planner
prompt branches or bespoke verifier code, stop and ask whether a new generic
primitive is actually needed.

## Checklist

1. Add a `CapabilityManifest`.
2. Declare required artifacts and allowed/forbidden artifacts.
3. Declare the output schema.
4. Declare the `EvidenceContract`.
5. Add positive/negative examples.
6. Add tests for registry cards, PlanSpec verification, answerability, and guards.
7. Add no bespoke verifier code unless a new generic primitive is needed.

## Complete Example

Example capability: answer "材料包里有哪些开放日产品？" from a material-pack open
calendar. It must not answer from `周报` or `月报`.

Add the manifest in:

```text
src/market_support_crewai_agent/runtime/domain/capabilities/manifests.py
```

```python
CapabilityManifest(
    id="material_pack.open_calendar",
    version="2026-06-17.1",
    display_name="Material pack open calendar",
    description="Answer 产品 opening-calendar questions from 材料包 content.",
    capability_type="answer",
    domain_entities=["channel", "strategy", "product", "artifact"],
    required_inputs=["request.dist_channel_name"],
    optional_inputs=["policy.material_pack_options"],
    required_artifacts=["material_pack"],
    allowed_artifacts=["material_pack"],
    forbidden_artifacts=["weekly_report", "monthly_report"],
    required_tools=["adapter_material_pack_content.open_calendar"],
    output_schema={
        "type": "object",
        "required": ["reply"],
        "properties": {
            "reply": {
                "type": "object",
                "required": ["kind", "text"],
                "properties": {
                    "kind": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
            "actions": {"type": "array"},
        },
    },
    evidence_contract=EvidenceContract(
        required_fact_types=["material_pack_open_calendar"],
        allowed_source_types=["adapter_material_pack_content"],
        forbidden_source_types=["adapter_report_scope", "document_mcp"],
        required_artifact_types=["material_pack"],
        allowed_artifact_types=["material_pack"],
        min_facts=1,
        fallback_policy="abstain",
    ),
    abstention_policy=AbstentionPolicy(
        requires_abstention_when_evidence_missing=True,
        abstention_reply_kinds=["unable_to_answer", "clarification"],
        guidance="If 材料包 open-calendar evidence is absent, abstain.",
    ),
    planner_guidance="Use for 材料包 产品开放日 or 申购日 questions.",
    agent_guidance="Answer only from material_pack_open_calendar evidence.",
    verifier_checks=[
        "output_schema",
        "required_runtime_input_present",
        "required_evidence_present",
        "evidence_artifact_type_allowed",
        "forbidden_source_not_used",
        "abstention_correctness",
    ],
    examples_positive=["材料包里哪些产品下周开放？"],
    examples_negative=["周报里产品表现怎么样？", "月报覆盖哪些产品？"],
    runtime_capability="material_pack",
)
```

## Required Artifacts

Use artifact names from the domain model:

```text
material_pack
weekly_report
monthly_report
document_context
adapter_context
history
```

For `材料包` answers, `周报` and `月报` are forbidden by default. Do not let reports
satisfy material-pack evidence unless the manifest explicitly allows fallback.

## Output Schema

Use the smallest schema needed by the capability. Most reply capabilities only
need:

```text
reply.kind
reply.text
actions
```

Action capabilities must include typed `actions`; answer capabilities should not
invent actions.

## Evidence Contract

Prefer exact facts and source types over prose:

```text
required_fact_types=["material_pack_open_calendar"]
allowed_source_types=["adapter_material_pack_content"]
required_artifact_types=["material_pack"]
required_scope_match=["channel_id", "strategy_name"]  # when needed
allow_history=False
```

Use `allow_history=true` only for capabilities that explicitly answer from
ledger/history and only with `required_scope_match` plus provenance checks.

## Examples

Examples should show capability boundaries:

```text
positive: "材料包里有哪些产品？"
negative: "周报里有哪些产品？"
negative: "月报里产品表现怎么样？"
```

Keep long product lists out of prompts. Use compact examples, explicit candidate
sets, or paginated evidence commands.

## Tests

Add the smallest deterministic tests:

```bash
uv run --extra dev python -m pytest -q tests/unit/domain/test_agent_behavior_eval_golden.py
uv run --extra dev python -m pytest -q tests/contract/test_plan_spec_contract.py
uv run --extra dev python -m pytest -q tests/unit/validation/test_answerability_gate.py
```

Test names should include the original bug when relevant, for example:

```text
test_regression_original_bug_material_pack_does_not_use_weekly_report_products
```

For a new source or primitive, add focused tests for:

```text
CapabilityRegistry planner card
PlanSpec validation
EvidenceContract source/artifact rejection
AnswerabilityGate abstention/clarification
direct guard output/evidence blocking
Prompt snapshot only if prompt text changed
```

## No Bespoke Verifier Code

Use generic verifier primitives:

```text
output_schema
required_evidence_present
evidence_artifact_type_allowed
required_runtime_input_present
forbidden_source_not_used
abstention_correctness
```

Add a new primitive only when the capability introduces a new class of check,
not because a new capability name exists.
