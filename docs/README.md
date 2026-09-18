# Documentation

These docs explain the service shape, operational surface, and extension workflow. They are grouped by functionality, not by implementation history.

## Read This First

- `../README.md` - local run commands, configuration, deployment, and the public `/reply` contract.
- `engineering-principles.md` - the architecture decision, contract boundaries, source-of-truth order, and implementation, test, and deployment policies.
- `architecture.md` - runtime flow, domain model, source precedence, and module map.
- `capabilities-and-prompts.md` - how capabilities, planner contracts, and prompt assembly fit together.
- `safety-and-evals.md` - guardrail pipeline, selector rules, and regression/eval commands.
- `adapter/assistant_adapter_contract.md` - external assistant WeCom adapter contract.

## Update Rules

- Keep each doc organized by functionality, not by implementation history.
- Prefer links to code owners over repeated explanations.
- Remove stale handoff notes when the implementation lands.
- Keep adapter-facing behavior in the adapter contract or root `README.md`, not scattered across docs.
