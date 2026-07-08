# T0 Guardrail Research Notepad

## Bootstrap
- Tier: HEAVY - user requested ulw research + design correction crossing prompt/guardrail architecture.
- Skills: omo:ulw-research (explicit ulw research), omo:programming (Python edits), ponytail (keep smallest correct design), openai-docs/web primary-source research (OpenAI/Claude/current guidance).
- Success criteria:
  1. T0-related messages route to human handoff via an injectable rule/policy shape, not a hardcoded branch or one class per guardrail.
  2. Non-T0 messages still fall through to planner.
  3. Q85 reply surface returns human_handoff with no actions.
- Scenario:
  - Unit: `uv run --extra dev python -m pytest -q tests/unit/domain/test_input_policy.py`
  - Contract: `uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_contract.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py tests/unit/domain/test_input_policy.py`
  - Surface: `PYTHONIOENCODING=utf-8 CREWAI_VERBOSE=false MARKET_AGENT_TRACE_LOG_EVENTS=false MARKET_AGENT_REPLY_ALIGNMENT_VERIFIER_ENABLED=false uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --ids 85 --parallel 1 --output-md tmp/t0_q85_handoff.md`

## Research Synthesis
- OpenAI: guardrails are lifecycle checks around input, output, tools, and human review; structured outputs constrain shape but deterministic validators own business policy.
- Anthropic: deterministic hooks/approval gates are recommended when behavior must always happen; untrusted content should be data, not instructions.
- OSS/Hermes: current frameworks converge on middleware/filter/rail registries and pure decision controllers, not one class per guardrail rule.
- Repo docs: `docs/support_reply_harness/guardrails.md` already requires one input-policy rule table and metadata-driven handoff text.

## Implementation Evidence
- RED: `uv run --extra dev python -m pytest -q tests/unit/domain/test_input_policy.py` failed on missing `InputPolicyRule`/`rules=` injection seam.
- GREEN unit: `uv run --extra dev python -m pytest -q tests/unit/domain/test_input_policy.py` -> 4 passed.
- GREEN harness: `uv run --extra dev python -m pytest -q tests/integration/runtime/test_reply_contract.py tests/contract/test_adapter_preflight.py tests/unit/validation/test_structured_guardrails.py tests/unit/state/test_action_feedback.py tests/unit/domain/test_input_policy.py` -> 113 passed.
- Surface: `scripts/eval_reply_xiaoyan_question_set.py --ids 85` -> status=200 kind=human_handoff, text asks sales/support confirmation, actions=-.

## Cleanup
- Agents closed: codebase, OpenAI, Anthropic, OSS.
- No live server/tmux/browser state created.
- Changed-file LOC measured separately.

## Review Gate
- Code review: `.omo/evidence/t0-guardrail-design-fix-code-review.md` -> APPROVED.
- Gate review: `.omo/evidence/t0-guardrail-design-fix-gate-review.md` -> APPROVED.
- Manual QA matrix: `.omo/ulw-research/20260707-213156-t0-guardrails/manual-qa-matrix.md`.
- Final evidence after policy-denied edge: unit 5 passed, harness 114 passed, Ruff clean, basedpyright 0 errors, Q85 surface human_handoff/actions=-.
