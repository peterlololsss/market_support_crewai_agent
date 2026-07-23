# market-support-crewai-agent

External CrewAI runtime service for the existing WeCom bot.

## Current scope

- FastAPI transport layer.
- `GET /health`.
- `POST /reply`.
- CrewAI runtime boundary.
- Typed request/response schema.
- In-memory conversation history keyed by `conversation_key`.
- Adapter preflight/resolve before outbound action proposals.
- Adapter-confirmed action feedback ledger with a 24h in-memory TTL for “just sent” semantics.

The runtime returns one `ReplyResponse`: primary reply semantics plus typed outbound action proposals. The WeCom
adapter owns message execution, final action validation, and execution authorization.

## Run locally

```bash
uv sync --extra dev
set -a
. ./.env
set +a
uv run uvicorn market_support_crewai_agent.server.main:app --reload
```

## Deploy from this machine

在当前 Windows 开发机仓库根目录执行。脚本会把当前源码通过 SSH 上传到 `192.168.209.195`，再在远端用 Podman 部署。固定使用 `23003:8000`，不做备用端口切换；如果 23003 被占用，脚本会直接失败。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_remote_podman.ps1
```

默认路径和服务名：

```text
Source:    /data/xiaoyan/market_support_crewai_agent/app
Env file:  /data/xiaoyan/market_support_crewai_agent/.env
Runtime:   /data/xiaoyan/market_support_crewai_agent/runtime
Image:     market-support-crewai-agent:latest
Container: market-support-crewai-agent
Port:      23003 -> 8000
```

远端 `.env` 必须已存在于 `/data/xiaoyan/market_support_crewai_agent/.env`。脚本会先检查 adapter、Document MCP、planner LLM proxy 连通性，再构建镜像、替换同名容器，并只执行 `/health` 和不含发送意图的安全 `/reply` smoke。

如果 SSH 用户不是默认的 `xiaoyan`，用：

```powershell
$env:REMOTE="your-user@192.168.209.195"; powershell -ExecutionPolicy Bypass -File .\scripts\deploy_remote_podman.ps1
```

如果在 Git Bash 里操作，也可以用：

```bash
bash scripts/deploy_remote_podman.sh
```

## Reply checks and evals

Run the Python test suite:

```bash
uv run --extra dev python -m pytest -q
```

Run a focused category:

```bash
uv run --extra dev python -m pytest -q -m unit
uv run --extra dev python -m pytest -q -m integration
uv run --extra dev python -m pytest -q -m contract
```

Run prompt registry lint:

```bash
uv run python scripts/check_prompt_registry.py
```

Run the semantic keyword-matching guard:

```bash
uv run python scripts/check_no_semantic_keyword_matching.py
```

Run the core acceptance check suite:

```bash
uv run --extra dev python scripts/check_reply_acceptance.py
```

This default suite uses fake external dependencies and includes the deterministic agent behavior golden evals. Add
`--include-real-llm` when provider credentials and network access are available. Add `--include-live-adapter` only after
starting the xiaoyan adapter fixture.

Run only the capability/evidence/domain correctness evals:

```bash
uv run --extra dev python -m pytest -q tests/unit/domain/test_agent_behavior_eval_golden.py tests/integration/runtime/test_agent_behavior_eval_suite.py
```

Run the harness pipeline without external LLM or adapter credentials:

```bash
uv run python scripts/check_reply_runtime_fake_deps.py
```

This uses fake CrewAI planner/composer outputs and fake adapter preflight, while exercising the real runtime
orchestration, evidence, business facts, and reply/action postcondition validators.

Run a real LLM-backed `/reply` knowledge-QA eval using `.env` and the configured Document MCP:

```bash
uv run python scripts/eval_reply_real_llm_knowledge.py
```

Run a real LLM-backed action-routing eval with fake adapter preflight:

```bash
uv run python scripts/eval_reply_real_llm_actions.py
```

Run a real LLM-backed handoff boundary eval. This verifies customer-service requests and unavailable material sends
produce harness-shaped handoff replies instead of ungrounded sends.

```bash
uv run python scripts/eval_reply_handoff.py
```

Run a real LLM-backed compliance eval. This isolates compliance planning and harness-owned refusal text, so adapter
preflight is disabled by default in the script.

```bash
uv run python scripts/eval_reply_compliance.py
```

Run a real adapter-feedback ledger check. It first verifies that a “just sent” follow-up without executed feedback
does not invent a report period, then posts an executed weekly-report feedback event and verifies the follow-up is grounded
by that ledger entry rather than another send action.

```bash
uv run python scripts/check_reply_action_feedback.py
```

Run a real LLM-backed action eval with live adapter preflight. Start the xiaoyan adapter first, or use the fixture
command in `docs/adapter/xiaoyan_adapter_contract.md`.

```bash
MARKET_AGENT_LIVE_ADAPTER_BASE_URL=http://127.0.0.1:8011 \
MARKET_AGENT_LIVE_ADAPTER_API_KEY=scope-secret \
uv run python scripts/eval_reply_live_adapter.py --message "请发一下周报"
```

## LLM configuration

```bash
YANFU_LLM_BASE_URL=https://llm.yanfuinvest.com/v1
YANFU_LLM_PROVIDER=openai
YANFU_LLM_MODEL=deepseek-v4-pro
YANFU_LLM_API_KEY=your-key
YANFU_LLM_TIMEOUT_SECONDS=90
YANFU_LLM_TEMPERATURE=0.1
YANFU_LLM_MAX_TOKENS=6000
MARKET_AGENT_PLANNER_LLM_BASE_URL=
MARKET_AGENT_PLANNER_LLM_PROVIDER=
MARKET_AGENT_PLANNER_LLM_MODEL=
MARKET_AGENT_PLANNER_LLM_API_KEY=
CREWAI_VERBOSE=false
CREWAI_MAX_ITER=5
CREWAI_MAX_EXECUTION_TIME=120
CREWAI_MAX_RETRY_LIMIT=2
MARKET_AGENT_PLANNER_TRANSIENT_RETRY_ATTEMPTS=1
MARKET_AGENT_PLANNER_TRANSIENT_RETRY_BASE_SECONDS=0.5
```

## LLM health notifications

The LLM health monitor is disabled by default. When enabled, it runs outside `/reply` and sends Chinese daily/failure/recovery messages to Feishu.

```bash
MARKET_AGENT_LLM_HEALTH_ENABLED=false
MARKET_AGENT_LLM_HEALTH_CHECK_INTERVAL_SECONDS=300
MARKET_AGENT_LLM_HEALTH_FAILURE_INTERVAL_SECONDS=60
MARKET_AGENT_LLM_HEALTH_DAILY_REPORT_TIME=09:00
MARKET_AGENT_LLM_HEALTH_TIMEZONE=Asia/Shanghai
MARKET_AGENT_LLM_HEALTH_WARNING_COOLDOWN_SECONDS=900
MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_ATTEMPTS=1
MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_BASE_SECONDS=1
MARKET_AGENT_LLM_HEALTH_PROBE_TIMEOUT_SECONDS=20
MARKET_AGENT_FEISHU_APP_ID=
MARKET_AGENT_FEISHU_APP_SECRET=
MARKET_AGENT_FEISHU_CHAT_ID=
```

## Conversation history configuration

Conversation history is cold-loaded from environment at service startup.

```bash
AGENT_INPUT_MAX_MESSAGE_CHARS=
AGENT_CONVERSATION_TTL_SECONDS=86400
AGENT_CONVERSATION_MAX_MESSAGES=24
AGENT_CONVERSATION_MAX_SESSIONS=5000
AGENT_CONVERSATION_CLEANUP_INTERVAL_SECONDS=300
```

## Context projection configuration

Before each planner/composer/verifier model call, runtime state is projected into a bounded `ModelVisibleContext`.

```bash
AGENT_CONTEXT_RECENT_TURNS_VERBATIM_COUNT=12
AGENT_CONTEXT_MAX_HISTORY_MESSAGE_CHARS_INLINE=2400
AGENT_CONTEXT_MAX_EVIDENCE_CHARS_INLINE=6000
AGENT_CONTEXT_MAX_ANSWER_EVIDENCE_CHARS_INLINE=1000000
AGENT_CONTEXT_LARGE_RESULT_PREVIEW_CHARS=1200
AGENT_CONTEXT_TOKEN_BUDGET=900000
AGENT_CONTEXT_WARNING_THRESHOLD=0.75
AGENT_CONTEXT_HARD_THRESHOLD=0.92
```

## Reply alignment and trace configuration

```bash
MARKET_AGENT_REPLY_ALIGNMENT_VERIFIER_ENABLED=true
MARKET_AGENT_REPLY_ALIGNMENT_MAX_REPLANS=1
MARKET_AGENT_REPLY_ALIGNMENT_MAX_EVIDENCE_REFETCHES=1
MARKET_AGENT_REPLY_ALIGNMENT_MAX_RECOMPOSES=1
MARKET_AGENT_REPLY_ALIGNMENT_MAX_TOTAL_REMEDIATIONS=2
MARKET_AGENT_LOG_LEVEL=INFO
MARKET_AGENT_TRACE_LOG_EVENTS=false
```

## Adapter resolve/preflight configuration

```bash
MARKET_AGENT_ADAPTER_BASE_URL=http://127.0.0.1:8011
MARKET_AGENT_ADAPTER_API_KEY=
MARKET_AGENT_ADAPTER_TIMEOUT_SECONDS=5
```

## Document MCP configuration

Document MCP access is configured separately from CrewAI agent prompts. It is disabled by default and is used only
through the fixed document evidence wrapper when explicitly enabled.

```bash
MARKET_AGENT_DOC_MCP_BASE_URL=http://192.168.209.195:23000
MARKET_AGENT_DOC_MCP_TIMEOUT_SECONDS=5
MARKET_AGENT_DOC_MCP_ENABLED=false
MARKET_AGENT_DOC_MCP_ALLOWED_CHANNEL_TYPES=bank,non_bank
MARKET_AGENT_DOC_MCP_MAX_CHARS_PER_DOCUMENT=1000000
MARKET_AGENT_DOC_MCP_CACHE_TTL_SECONDS=300
MARKET_AGENT_DOC_MCP_BASELINE_CATEGORIES=常见问答
```

The current document MCP server responds as streamable HTTP on `/mcp`, requires `Accept: application/json, text/event-stream`,
and exposes wrapper-only tools `list_products` and `get_documents`. When the selector returns valid document IDs, the
wrapper fetches only those documents. If selection is unsure, it falls back to the bounded baseline/broad read.

## Incoming request authentication

Incoming request authentication is optional for local development. Configure this when the xiaoyan gateway should
authenticate `/reply` and `/actions/feedback`.

```bash
MARKET_AGENT_API_KEY=shared-secret
```

When configured, requests include either `Authorization: Bearer <key>` or `X-API-Key: <key>`. This matches xiaoyan's
`market_agent_api_key` setting.

## Public service contract

`POST /reply` requires:

```text
conversation_key
group_id
sender_id
message
is_group
```

For group-chat requests, the gateway sends `conversation_key` as `wecom:{group_id}:{sender_id}`. `context_id` is
optional and only used for tracing.

The public runtime response is `ReplyResponse { reply, actions }`. The adapter executes `reply` and typed outbound
action proposals after its own validation.

## Documentation map

- `AGENTS.md`: short operational contract for coding agents.
- `docs/agent-architecture.md`: current agent architecture, capability registry, DomainContext, PlanSpec, EvidenceContract, guardrails, answerability, and prompt layers.
- `docs/add-a-capability.md`: manifest-first capability extension guide with a full example.
- `docs/domain-model.md`: 渠道/策略/产品/材料包/周报/月报 hierarchy, artifact distinctions, and source precedence.
- `docs/guardrails.md`: input, retrieval/evidence, execution/tool, output, input-scope, and audit reason-code guidance.
- `docs/keyword-matching-cleanup.md`: banned semantic matching patterns and CI enforcement.
- `docs/support_reply_harness/README.md`: harness doc index and active source-of-truth map.
- `docs/support_reply_harness/adr/0001-support-reply-harness.md`: architecture decision record.
- `docs/support_reply_harness/architecture.md`: runtime shape, source hierarchy, and internal concepts.
- `docs/support_reply_harness/guardrails.md`: deterministic guardrail pipeline and validator behavior.
- `docs/support_reply_harness/eval_plan.md`: regression, adversarial, and golden eval plan.
- `docs/support_reply_harness/roadmap.md`: historical phased implementation plan.
- `docs/support_reply_harness/next_session.md`: immediate coding-session handoff.
- `docs/support_reply_harness/reference/agent_prompt_hygiene.md`: prompt/context hygiene for Codex-style coding agents.
- `docs/adapter/xiaoyan_adapter_contract.md`: xiaoyan WeCom adapter contract and live eval commands.
- `docs/capability-registry.md`: manifest schema and extension path for planner/verifier capability metadata.
- `docs/prompts.md`: prompt registry, layer, snapshot, and extension rules.

## Current module ownership

```text
src/market_support_crewai_agent/server/main.py                 FastAPI routes only
src/market_support_crewai_agent/schemas.py                     HTTP and action contracts
src/market_support_crewai_agent/runtime/orchestration/          reply runtime, decisions, rendering
src/market_support_crewai_agent/runtime/validation/             input/reply/action/alignment validators
src/market_support_crewai_agent/runtime/evidence/               adapter/document evidence wrappers and facts
src/market_support_crewai_agent/runtime/knowledge/              approved static knowledge catalog and selector
src/market_support_crewai_agent/runtime/domain/                 capabilities, policy, planning, canonical facts
src/market_support_crewai_agent/runtime/llm/                    prompt assembly, routing, profiles, resources
src/market_support_crewai_agent/runtime/state/                  conversation, ledger, and audit state
```

Add new runtime-only harness modules under the relevant `src/market_support_crewai_agent/runtime/` package described in `AGENTS.md` and
`docs/support_reply_harness/next_session.md`.
