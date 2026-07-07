# Xiaoyan Artifact Source Balance Code Review

## Verdict

- codeQualityStatus: BLOCK
- recommendation: REQUEST_CHANGES
- reportPath: `.omo/evidence/xiaoyan-artifact-source-balance-code-review.md`
- blockers:
  - New semantic keyword guard violation in `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:89`.
  - Strict typing diagnostics in new `tests/unit/domain/test_input_policy.py`.

## Skill Perspective Check

- `omo:remove-ai-slops`: loaded and applied as review lens. Violations found: the new input-policy substring matcher is unnecessary production complexity with an empty rule table, and some added tests only verify removal/prompt prose rather than behavior.
- `omo:programming`: loaded with Python reference. Violations found: basedpyright warnings in new tests; prompt tests mirror implementation strings; `manifests.py` remains oversized at 682 pure LOC, though that appears pre-existing/static-catalog risk rather than the main blocker.

## Findings

### CRITICAL

- None.

### HIGH

1. `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:89` introduces a new semantic keyword scanner violation. `uv run python scripts/check_no_semantic_keyword_matching.py` reports `runtime/domain/planning/input_policy.py:89: suspicious text membership check outside allowlist`. The full command also reports existing unrelated `runtime/state/latency_report.py` findings, but this changed file adds a new failure. Because `INPUT_POLICY_RULES` is empty at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:62`, the substring matcher does not help the intended T0 removal today and leaves a path for future keyword guardrails to bypass the planner again. Remove the unused matcher path or replace it with a structured, explicitly allowed mechanism before approval.

2. `tests/unit/domain/test_input_policy.py:28` and `tests/unit/domain/test_input_policy.py:103` fail strict typing diagnostics. `basedpyright src/.../manifests.py src/.../input_policy.py tests/unit/domain/test_input_policy.py tests/unit/llm/test_prompt_router.py` exits non-zero with 7 warnings: unknown/missing type for `**overrides`, partially unknown `payload.update`, unknown `policy`, and unknown argument to `validate_execution_plan` at `tests/unit/domain/test_input_policy.py:128`. This contradicts the reported clean diagnostics and violates the Python strict-type review lens.

### MEDIUM

1. `tests/unit/domain/test_input_policy.py:49` and `tests/unit/domain/test_input_policy.py:63` only prove that T0 messages return `no_match` from an empty rule table. They do not verify the intended positive behavior that T0 can route through Document MCP when supported. The live eval artifact covers one T0 answer, but the unit tests create false confidence because any message would currently pass the same no-match assertion.

2. `tests/unit/llm/test_prompt_router.py:126`, `tests/unit/llm/test_prompt_router.py:139`, and `tests/unit/llm/test_prompt_router.py:150` assert exact prompt prose substrings. This is brittle prompt-test coverage: it can pass while the planner still emits the wrong PlanSpec, and it can fail on harmless wording changes. Prefer parsed prompt/rule-data assertions or deterministic fake-planner/PlanSpec behavior checks for the artifact-vs-document boundary.

### LOW

1. `src/market_support_crewai_agent/runtime/domain/capabilities/manifests.py:1` is 682 pure LOC after this change. That is an existing static-catalog shape rather than a new behavior bug, but it violates the programming skill's 250 pure-LOC lens and makes further prompt/manifest changes harder to review safely.

## Verified Evidence

- Inspected relevant diffs for manifests, planner prompt fragments, taxonomy, prompt router tests, new input policy, and new input-policy tests.
- Inspected live eval artifact `tmp/xiaoyan_question_set_balance_final.md`; it exists and records 9 HTTP 200 cases with #34/#68/#85/#91 knowledge answers and #15/#19/#39/#54/#103 weekly report sends for period `20260703`.
- Re-ran `uv run --extra dev python -m pytest -q tests/unit/domain/test_input_policy.py tests/unit/llm/test_prompt_router.py tests/unit/llm/test_prompt_assembler.py tests/unit/llm/test_prompt_snapshots.py`: 28 passed.
- Re-ran `uv run --extra dev python -m pytest -q tests/unit/domain/test_capability_registry.py tests/contract/test_plan_spec_contract.py tests/integration/runtime/test_reply_contract.py`: 75 passed.
- Ran `uv run python scripts/check_no_semantic_keyword_matching.py`: failed, including new changed-file failure at `input_policy.py:89`.
- Ran basedpyright on the changed Python files/tests listed above: 0 errors, 7 warnings, non-zero exit.

## Scope/Correctness Notes

- The prompt and manifest text generally encode the intended source balance: current/recent/live performance routes to weekly/monthly artifacts, evergreen FAQ facts stay document-backed, material packs remain send collateral, and absent send capabilities are not selectable.
- No over-routing issue was found in the provided live eval evidence; the observed cases match the intended boundary.
- Approval is blocked by guard/type quality gates rather than the high-level routing intent.
