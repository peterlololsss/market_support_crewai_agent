# ULW Notepad - prompt routing balance
Started: 2026-07-07 Asia/Shanghai
Tier: HEAVY - prompt changes affect LLM routing, external live adapter/doc MCP behavior, and user explicitly requested ultrawork/subagents.
Skills: ulw-loop (requested ultrawork), debugging (wrong runtime responses), programming/python (tests/scripts may change), openai-docs (prompt guidance), ponytail full active but user explicitly asked to not optimize for shortest lazy path.
Objective: rebalance planner/composer prompts so evergreen MCP knowledge questions use document_context/knowledge_answer while current valuation/report/material requests still produce the right adapter actions.
Success criteria:
- C1 RED/GREEN: focused regression tests fail before prompt change and pass after for #34/#68/#85/#91 plus report-action positives.
- C2 Real surface: live xiaoyan question-set or focused TestClient eval shows #34/#68/#85/#91 route to knowledge_answer/document evidence where supported, and #15/#19/#39/#54/#103 still send weekly report.
- C3 Risk check: no material/monthly send regressions; prompt snapshots updated intentionally; no new untracked debug artifacts.
Hypotheses:
1. H1 overbroad weekly_report prompt tokens route any "超额/表现/回撤" question to weekly_report.send; evidence: planner prompt + traces; fix: priority/exclusions.
2. H2 input_policy hard-blocks T0 before MCP; evidence: input_policy rule; fix: remove/narrow T0 hard handoff.
3. H3 "周报" lexical anchor routes fee/meaning FAQ to report-scope instead of document_context; evidence: trace/report-scope not_found; fix: clarify weekly report as current valuation-table artifact only, not every mention of 周报.
4. H4 action reply rationale flag variability causes empty vs explanatory text; evidence: risk flags; fix only if tests show user-facing need.
Prompt principles to apply: explicit source hierarchy, positive/negative examples, decision tree ordered by user intent, artifact definitions, concise schema-compatible instructions.

Artifacts:
- .omo/ulw-loop/bootstrap-notepad.md (this file)
- .debug-journal.md (debug ledger, git-excluded)
