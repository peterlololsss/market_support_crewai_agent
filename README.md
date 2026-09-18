# market-support-crewai-agent

External CrewAI runtime service for the existing WeCom bot.

## Current scope

- FastAPI transport layer.
- `GET /health`.
- `POST /reply`.
- One shared CrewAI Reply Harness runtime for direct and group traffic.
- Versioned `reply-request.v2` request schema and typed response schema.
- In-memory conversation history keyed by verified conversation identity.
- Adapter preflight/resolve before outbound action proposals.
- Adapter-confirmed action feedback ledger with a 24h in-memory TTL for “just sent” semantics.

The runtime returns one `ReplyResponse`: primary reply semantics plus typed outbound action proposals. The WeCom
adapter owns message execution, final action validation, and execution authorization.

`/health` is liveness only. It returns the process-level service response without checking adapter compatibility,
deployment identity, model availability, or internal-DM readiness.

## Run locally

```bash
uv sync --extra dev
set -a
. ./.env
set +a
uv run uvicorn market_support_crewai_agent.server.main:app --reload
```

## Deploy from this machine

在当前 Windows 开发机仓库根目录执行。脚本会把当前源码通过 SSH 上传到 `10.0.0.12`，再在远端用 Podman 部署。固定使用 `23003:8000`，不做备用端口切换；如果 23003 被占用，脚本会直接失败。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_remote_podman.ps1
```

默认路径和服务名：

```text
Source:    /opt/market-support-agent/app
Env file:  /opt/market-support-agent/.env
Runtime:   /opt/market-support-agent/runtime
Image:     market-support-crewai-agent:latest
Container: market-support-crewai-agent
Port:      23003 -> 8000
```

远端 `.env` 必须已存在于 `/opt/market-support-agent/.env`。脚本会先检查 adapter、Document MCP、planner LLM proxy 连通性，再构建镜像、替换同名容器，并只执行 `/health` 和不含发送意图的安全 `/reply` smoke。

如果 SSH 用户不是默认的 `deploy`，用：

```powershell
$env:REMOTE="your-user@10.0.0.12"; powershell -ExecutionPolicy Bypass -File .\scripts\deploy_remote_podman.ps1
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

The default suite runs three offline checks: the semantic keyword guard, prompt registry validation, and the fake-dependency
runtime. Add `--include-real-llm` when provider credentials and network access are available. Add `--include-live-adapter`
only after starting the xiaoyan adapter fixture.

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
YANFU_LLM_BASE_URL=https://llm.example.com/v1
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
CREWAI_MAX_ITER=1
CREWAI_MAX_EXECUTION_TIME=120
CREWAI_MAX_RETRY_LIMIT=0
MARKET_AGENT_PLANNER_TRANSIENT_RETRY_ATTEMPTS=0
MARKET_AGENT_PLANNER_TRANSIENT_RETRY_BASE_SECONDS=0
```

These iteration and retry values are fixed governance controls. Startup rejects any other or malformed value. Alignment action caps must not exceed `MARKET_AGENT_REPLY_ALIGNMENT_MAX_TOTAL_REMEDIATIONS`; the configured shared-iteration formula is checked against the 18-dispatch ceiling.

## LLM health notifications

The LLM health monitor is disabled by default. When enabled, it runs outside `/reply` and sends Chinese daily/failure/recovery messages to Feishu.

```bash
MARKET_AGENT_LLM_HEALTH_ENABLED=false
MARKET_AGENT_LLM_HEALTH_CHECK_INTERVAL_SECONDS=300
MARKET_AGENT_LLM_HEALTH_FAILURE_INTERVAL_SECONDS=60
MARKET_AGENT_LLM_HEALTH_DAILY_REPORT_TIME=09:00
MARKET_AGENT_LLM_HEALTH_TIMEZONE=Asia/Shanghai
MARKET_AGENT_LLM_HEALTH_WARNING_COOLDOWN_SECONDS=900
MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_ATTEMPTS=0
MARKET_AGENT_LLM_HEALTH_PROBE_RETRY_BASE_SECONDS=0
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
AGENT_CONVERSATION_MAX_MESSAGES=12
AGENT_CONVERSATION_MAX_SESSIONS=5000
AGENT_CONVERSATION_CLEANUP_INTERVAL_SECONDS=300
```

## Model input boundaries

Each planner, composer, and verifier call receives one frozen role-specific stage DTO. The DTO schemas bound history,
evidence, policy, plan, retry, and candidate-response views before prompt assembly. Static prompt resources are governed
separately by the registered prompt-program budgets.

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

Authenticated adapter calls reject every redirect and require the response origin to match the configured origin.
Use `https://` for normal deployments. `http://` is accepted only for literal loopback, RFC1918, or link-local
addresses; this compatibility mode must stay on an authenticated private or encrypted tunnel network. Hostnames over
cleartext HTTP, public IP addresses, URL userinfo, fragments, and non-HTTP schemes fail closed at client construction.

## Document MCP configuration

Document MCP access is configured separately from CrewAI agent prompts. It is disabled by default and is used only
through the fixed document evidence wrapper when explicitly enabled.

```bash
MARKET_AGENT_DOC_MCP_BASE_URL=http://10.0.0.12:23000
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

## V2 admission configuration

`MARKET_AGENT_API_KEY` is mandatory for `/reply` and `/actions/feedback`. `MARKET_AGENT_DEPLOYMENT_TENANT_REF` is mandatory
for every `reply-request.v2` and `action-feedback.v2` event. The settings may remain blank only for a liveness-only process;
v2 traffic then fails closed instead of entering the runtime.

`MARKET_AGENT_INTERNAL_DM_ENABLED` defaults to `false`. A direct turn also requires a valid direct-audit HMAC key, an
adapter API key, and a compatible authenticated adapter capability response before any reply reservation or state work.

```bash
MARKET_AGENT_API_KEY=shared-secret
MARKET_AGENT_DEPLOYMENT_TENANT_REF=tenant:primary
MARKET_AGENT_INTERNAL_DM_ENABLED=false
MARKET_AGENT_DIRECT_AUDIT_HMAC_KEY=replace-with-a-32-to-64-byte-url-safe-key
MARKET_AGENT_ADAPTER_API_KEY=adapter-shared-secret
```

Requests include either `Authorization: Bearer <key>` or `X-API-Key: <key>`. The request identity's `tenant_ref` must
exactly equal the configured deployment tenant; callers cannot select or switch the deployment tenant.

## Public service contract

`POST /reply` accepts only the versioned `reply-request.v2` request contract. The scene-aware harness does not provide a
legacy request wrapper, compatibility bridge, or legacy/v2 route union; unversioned legacy payloads are rejected at the
request boundary before stateful runtime work or model-visible calls.

The wire scene values remain `direct` and `group`. The operational aliases are prose only: `internal_dm` means
`identity.scene="direct"`, and `external_group` means `identity.scene="group"`. Do not send either alias as a field or
scene value.

Group request example:

```json
{
  "contract_version": "reply-request.v2",
  "request_id": "req:group-message-001",
  "message": "请发一下周报",
  "context_id": "ctx:origin-message-001",
  "identity": {
    "contract_version": "conversation-identity.v1",
    "surface": "wecom",
    "scene": "group",
    "tenant_ref": "tenant:primary",
    "group_ref": "group:opaque-001",
    "principal_ref": "principal:opaque-001"
  },
  "presentation": {
    "contract_version": "group-presentation.v1",
    "conversation_name": "Example group",
    "principal_name": "Example user"
  },
  "business_scope": {
    "kind": "distribution",
    "dist_channel_name": "银河证券",
    "channel_type": "non_bank",
    "available_artifacts": [
      {"type": "weekly_report", "options": []}
    ]
  },
  "grants": {
    "contract_version": "principal-grants.v1",
    "read_capabilities": ["resolve_weekly_report", "resolve_sales_mention"],
    "outbound_actions": ["send_weekly_report"],
    "mention_types": ["sales"]
  }
}
```

Direct reply-only request example:

```json
{
  "contract_version": "reply-request.v2",
  "request_id": "req:direct-message-001",
  "message": "介绍一下公司",
  "context_id": "ctx:origin-message-002",
  "identity": {
    "contract_version": "conversation-identity.v1",
    "surface": "wecom",
    "scene": "direct",
    "tenant_ref": "tenant:primary",
    "direct_thread_ref": "direct:opaque-001",
    "principal_ref": "principal:opaque-001"
  },
  "presentation": {
    "contract_version": "direct-presentation.v1",
    "principal_name": "Example user"
  },
  "business_scope": {"kind": "unscoped"},
  "grants": {
    "contract_version": "principal-grants.v1",
    "read_capabilities": ["query_internal_company_info"],
    "outbound_actions": [],
    "mention_types": []
  }
}
```

Required v2 identity fields:

```text
contract_version=reply-request.v2
request_id
message
identity.contract_version=conversation-identity.v1
identity.surface=wecom
identity.scene=group|direct
identity.tenant_ref
identity.group_ref or identity.direct_thread_ref
identity.principal_ref
presentation
business_scope
grants.contract_version=principal-grants.v1
grants.read_capabilities
grants.outbound_actions
grants.mention_types
```

`context_id` is optional and used for tracing and replay correlation. Raw `conversation_key`, `group_id`, `sender_id`,
and `is_group` inputs are legacy-only fields and are not accepted by the v2 boundary.

Direct requests are reply-only and unscoped. `read_capabilities` may be empty or exactly
`["query_internal_company_info"]`; the documented example explicitly opts into that internal-knowledge grant. Direct
recall is off, and direct responses cannot carry actions, sales mentions, media markers/bindings, or adapter business
resolves. External-group requests remain distribution-scoped and principal-scoped: two principals in one group do not
share history or state. Group actions remain typed proposals, and the adapter retains final validation, authorization,
execution, outbox reliability, and feedback authority.

The public runtime response is `ReplyResponse { reply, actions }`. The adapter executes `reply` and typed outbound
action proposals after its own validation.

The external adapter release is a production prerequisite for internal DM. Keep
`MARKET_AGENT_INTERNAL_DM_ENABLED=false` until the deployed adapter's authenticated, read-only capability check advertises
the required direct scene, contract versions, and matching tenant. This repository does not claim that production DM
is active; production DM remains disabled until that external proof passes.

## Documentation map

Human docs are grouped by functionality:

- `docs/README.md`: documentation index.
- `docs/engineering-principles.md`: architecture decision, contract boundaries, source-of-truth order, and implementation, test, and deployment policies.
- `docs/architecture.md`: runtime flow, domain model, source precedence, and module map.
- `docs/capabilities-and-prompts.md`: capability registry, PlanSpec boundary, prompt assembly, and extension workflow.
- `docs/safety-and-evals.md`: guardrail pipeline, selector rules, and eval/test commands.
- `docs/adapter/xiaoyan_adapter_contract.md`: xiaoyan WeCom adapter contract and live eval commands.

## Module map

```text
src/market_support_crewai_agent/server/            FastAPI app, routes, lifespan, auth, adapter compatibility
src/market_support_crewai_agent/schemas/           public HTTP/action DTOs
src/market_support_crewai_agent/runtime/           support reply harness runtime
```

For detailed runtime ownership, read `docs/architecture.md`.
