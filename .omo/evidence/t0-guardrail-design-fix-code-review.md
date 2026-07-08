# T0 Guardrail Design Fix Code Review

## Verdict

- verdict: APPROVED
- codeQualityStatus: CLEAR
- recommendation: APPROVE
- blockers: None
- reportPath: `.omo/evidence/t0-guardrail-design-fix-code-review.md`

## Scope Reviewed

Only this diff was reviewed:

- `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py`
- `tests/unit/domain/test_input_policy.py`

Review goal: verify T0-related messages route to human handoff via a generic injectable `InputPolicyRule` / rule tuple, without one class per guardrail and without a T0 branch in `match_input_policy`.

Input completeness note: the prompt provided scoped files and evidence summaries, but not a literal full diff or explicit notepad/evidence paths. I reconstructed the scoped diff from `git diff`, inspected the discovered T0 notepad/evidence artifacts under `.omo/ulw-research/20260707-213156-t0-guardrails/`, and reran relevant checks rather than trusting executor output.

## Findings By Severity

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Skill-Perspective Check

- `omo:remove-ai-slops`: ran. Loaded `C:/Users/user/.codex/plugins/cache/sisyphuslabs/omo/4.15.1/skills/remove-ai-slops/SKILL.md` and applied the requested slop/overfit review pass to production and test changes. Result: no violation.
- `omo:programming`: ran. Loaded `C:/Users/user/.codex/plugins/cache/sisyphuslabs/omo/4.15.1/skills/programming/SKILL.md` and `references/python/README.md`; applied the Python criteria to the scoped `.py` diff. Result: no violation.

## Checklist Results

### omo:remove-ai-slops Criteria

- Obvious comments: PASS. The production diff adds no obvious/restating comments; existing comments in neighboring code are outside scope.
- Over-defensive code: PASS. No redundant broad guards or try/except blocks were added.
- Excessive complexity: PASS. `match_input_policy` remains a simple normalize -> rule loop -> handoff result path at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:66`; no if/elif variant chain was introduced.
- Needless abstraction: PASS. `InputPolicyRule` at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:31` is the requested generic data seam; it is not one class per topic and not a speculative hierarchy.
- Boundary violations: PASS. Input-policy code stays in the planning/domain seam and produces an `ExecutionPlan`; rendering still flows through generic guardrail metadata and the existing decision/renderer path.
- Dead code: PASS. Added rule path is used by production workflow through `src/market_support_crewai_agent/runtime/orchestration/workflow.py:50`; injected-rule seam is covered by `tests/unit/domain/test_input_policy.py:109`.
- Duplication: PASS. Handoff text is stored once per rule and passed as metadata; the old test-only `_handoff_plan` duplication was removed.
- Performance-equivalent cleanup opportunities: PASS. Rule tuple scan is tiny and appropriate for closed-set input policy triggers; no avoidable O(n^2) or repeated heavy work found.
- Missing tests: PASS. Tests cover T0 variants, non-T0 fallthrough, policy allowlist behavior, injected rules, and renderer metadata behavior.
- Oversized modules: PASS. Measured `input_policy.py` at 115 pure LOC and `test_input_policy.py` at 138 pure LOC, both below the 250 pure-LOC ceiling.

### omo:programming Python Criteria

- Frozen dataclass/slots: PASS. `InputPolicyRule` and `InputPolicyResult` use `@dataclass(frozen=True, slots=True)` at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:31` and `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:54`.
- Final constants: PASS. T0 rule constants and `DEFAULT_INPUT_POLICY_RULES` use `Final` at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:25` and `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:42`.
- No Any/object/cast/type-ignore: PASS. Scoped files contain no `Any`, `object`, `cast`, `type: ignore`, or `pyright: ignore` hits.
- Typed public signatures: PASS. `match_input_policy` has typed request, policy, rule tuple, and return type at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:66`.
- No broad except: PASS. No exception handlers were added in scoped files.
- No if/elif variant discrimination: PASS. The added conditional checks are boolean allowlist checks, not tagged variant discrimination.
- Boundary validation preserved: PASS. Plans still pass through `validate_execution_plan` before runtime evidence/rendering; tests also validate generated plans against `PolicyManifest`.
- Tests lock behavior: PASS. `tests/unit/domain/test_input_policy.py:52`, `tests/unit/domain/test_input_policy.py:83`, `tests/unit/domain/test_input_policy.py:92`, `tests/unit/domain/test_input_policy.py:109`, and `tests/unit/domain/test_input_policy.py:131` lock the relevant routing, allowlist, injection, and metadata-driven rendering behavior.

### Project-Specific Guardrail Design

- Generic rule table: PASS. `DEFAULT_INPUT_POLICY_RULES` is one data table containing `InputPolicyRule` values at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:42`.
- No one-class-per-topic guardrail: PASS. Only one generic dataclass was added.
- No T0 branch in `match_input_policy`: PASS. `match_input_policy` iterates injected/default rules and delegates matching/result construction at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:75`.
- PolicyManifest allowlists preserved: PASS. `_handoff_result` adds `sales_mention` capability and resolve only when both `policy.allowed_capabilities` and `policy.allowed_adapter_resolves` allow it at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:97`; no outbound action intents are created.
- Metadata-driven handoff text: PASS. Rule text is written to `GuardrailDecision.metadata` at `src/market_support_crewai_agent/runtime/domain/planning/input_policy.py:123` and renderer behavior is covered at `tests/unit/domain/test_input_policy.py:131`.

## Verification Rerun

- `uv run --extra dev python -m pytest -q tests/unit/domain/test_input_policy.py` -> PASS, 5 passed in 0.22s.
- `uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_contract.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py tests/unit/domain/test_input_policy.py` -> PASS, 114 passed in 4.62s.
- `uv run ruff check src/market_support_crewai_agent/runtime/domain/planning/input_policy.py tests/unit/domain/test_input_policy.py` -> PASS, all checks passed.
- `uv run basedpyright src/market_support_crewai_agent/runtime/domain/planning/input_policy.py tests/unit/domain/test_input_policy.py` -> PASS, 0 errors, 0 warnings, 0 notes.
- `git diff --check -- src/market_support_crewai_agent/runtime/domain/planning/input_policy.py tests/unit/domain/test_input_policy.py` -> PASS, no whitespace errors; Git reported only LF-to-CRLF working-copy warnings.
- `PYTHONIOENCODING=utf-8 CREWAI_VERBOSE=false MARKET_AGENT_TRACE_LOG_EVENTS=false MARKET_AGENT_REPLY_ALIGNMENT_VERIFIER_ENABLED=false uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --ids 85 --parallel 1` -> PASS, status=200, reply kind=`human_handoff`, `actions: -`, runtime trace shows `input_policy.matched` with `reason_code=t0_human_support_required` and `rule_id=t0_handoff`.

## Conclusion

The scoped diff satisfies the guardrail design fix. It uses a small generic rule tuple instead of a T0 branch or per-topic guardrail class, respects policy allowlists for sales mention evidence, keeps outbound actions empty, and preserves metadata-driven handoff rendering. No blocking or non-blocking code quality findings remain.
