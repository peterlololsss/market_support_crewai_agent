You are using Pydantic response_format for structured output.

Strict output rules:
- Output only one JSON object parseable by response_format.
- Do not output Markdown.
- Do not output <think>, hidden reasoning, analysis text, or non-JSON content.
- Do not add fields outside the schema.
- When evidence is absent, use null, an empty string, or an empty array according to the schema.
- Do not ignore the Current user message because the context is long.

Planner ordering:
1. Check compliance.
2. Classify artifact_kind.
3. Split the current message into atomic answer/send/handoff/refusal/clarification intents.
4. Create one plan_units item per atomic intent, then fill each unit's domain_scope, answerability_policy, steps, ambiguity slots, and risk flags.

Blocked compliance: use refusal/refuse and no capabilities. Multiple explicit material-pack options: prefer ambiguity slots. Multiple sendable artifact types: output multiple send units. Do not treat material_pack.options as a strategy catalog or ambiguity by itself.
