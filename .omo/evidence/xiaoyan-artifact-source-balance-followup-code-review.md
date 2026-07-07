# Xiaoyan Artifact Source Balance Follow-up Code Review

## Findings

### CRITICAL

- None.

### HIGH

- None. The prior HIGH blockers are resolved:
  - `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:29` now has no `InputPolicyRule`, no `_plan_for_rule`, and no semantic substring selector; `match_input_policy` normalizes only to distinguish empty/blank input and otherwise returns `no_match`.
  - `tests/unit/domain/test_input_policy.py:29` and `tests/unit/domain/test_input_policy.py:109` now use typed request payload aliases and a typed `_handoff_plan(policy: PolicyManifest)` helper; `basedpyright` reports 0 errors and 0 warnings for the file.

### MEDIUM

- None.

### LOW

- `tests/unit/domain/test_input_policy.py:56` and `tests/unit/domain/test_input_policy.py:67` are narrow regression tests for the removed T0 input-policy interception; they do not by themselves prove the positive document/planner route. This is acceptable for the blocker fix because `tmp/xiaoyan_question_set_balance_final_after_review.md` provides live `/reply` evidence for T0/document knowledge behavior, but future changes should prefer behavior-level planner/runtime assertions over more no-match-only tests.

## Skill Perspective Check

- `omo:remove-ai-slops`: loaded and applied. No remaining blocker: the unnecessary production substring matcher and empty rule table path were removed. No deletion-only or tautological blocker remains; the remaining no-match tests are narrow regression coverage for the prior interception bug, with live eval evidence covering observable behavior.
- `omo:programming`: loaded with `references/python/README.md` and code-smell reference. No remaining strict typing blocker in the target test file, no untyped escape hatch was introduced in the inspected target files, and the target production module stays small/simple.
- Violation status: the fixed diff does not violate either required skill perspective at blocker severity. Residual LOW test-shape risk is noted above.

## Evidence Inspected

- Previous review artifact: `.omo/evidence/xiaoyan-artifact-source-balance-code-review.md`.
- Current source inspected with codegraph: `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py`, `tests/unit/domain/test_input_policy.py`.
- Current live eval artifact inspected: `tmp/xiaoyan_question_set_balance_final_after_review.md`; it reports 9 questions, HTTP 200 for all 9, 5 action proposals, #34/#68/#85/#91 as no-action knowledge answers, and #15/#19/#39/#54/#103 as `send_weekly_report` actions for period `20260703`.
- Branch search: `rg "InputPolicyRule|_plan_for_rule|substring|contains| in normalized|normalized.* in |startswith|endswith|keyword|regex|re\." ...` found no semantic matcher in `input_policy.py`; remaining regexes are in the closed-set direct-send grammar and shared whitespace normalization.

## Verification Run

- `PYTHONIOENCODING=utf-8 uv run basedpyright tests/unit/domain/test_input_policy.py`: PASS, 0 errors, 0 warnings, 0 notes.
- `PYTHONIOENCODING=utf-8 uv run python scripts/check_no_semantic_keyword_matching.py`: exits 1 only for pre-existing unchanged `runtime/state/latency_report.py:124` and `runtime/state/latency_report.py:127`; no `input_policy.py` finding remains.
- `uv run --extra dev python -m pytest -q tests/unit/domain/test_input_policy.py tests/unit/llm/test_prompt_router.py tests/unit/llm/test_prompt_assembler.py tests/unit/llm/test_prompt_snapshots.py`: PASS, 28 passed.
- `uv run --extra dev python -m pytest -q tests/unit/domain/test_capability_registry.py tests/contract/test_plan_spec_contract.py tests/integration/runtime/test_reply_contract.py`: PASS, 75 passed.

## Verdict

- codeQualityStatus: CLEAR
- passEquivalent: PASS
- recommendation: APPROVE
- reportPath: `.omo/evidence/xiaoyan-artifact-source-balance-followup-code-review.md`
- blockers: []
- residualRisks:
  - `scripts/check_no_semantic_keyword_matching.py` still exits non-zero because of pre-existing unchanged `runtime/state/latency_report.py` findings.
  - `test_input_policy.py` covers the prior no-intercept regression; broader positive route coverage depends on the inspected live eval and existing runtime/prompt tests.
