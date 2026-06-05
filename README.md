# market-support-crewai-agent

External CrewAI runtime service for the existing WeCom bot.

## Current scope

- FastAPI transport layer.
- `GET /health`.
- `POST /reply`.
- CrewAI runtime boundary.
- Typed request/response schema.
- In-memory conversation history keyed by `conversation_key`.
- Adapter preflight/resolve before side-effect action proposals.
- Adapter-confirmed action feedback ledger with a 24h in-memory TTL for “just sent” semantics.

The runtime returns one `ReplyResponse`: primary reply semantics plus typed side-effect action proposals. The WeCom
adapter owns message execution and final side-effect gating.

## Run locally

```bash
uv sync --extra dev
set -a
. ./.env
set +a
uv run uvicorn market_support_crewai_agent.server.main:app --reload
```

## MVP smoke flow

Run the harness pipeline without external LLM or adapter credentials:

```bash
uv run python scripts/smoke_reply_mvp.py
```

This uses fake CrewAI planner/composer outputs and fake adapter preflight, while exercising the real runtime
orchestration, evidence, business facts, and reply/action guardrails.

Run a real LLM-backed `/reply` knowledge-QA smoke using `.env` and the configured Document MCP:

```bash
uv run python scripts/smoke_reply_real_llm.py
```

Run real LLM-backed action-routing smokes with fake adapter preflight:

```bash
uv run python scripts/smoke_reply_real_llm_actions.py
```

Run a real LLM-backed action smoke with live adapter preflight. Start the xiaoyan adapter first, or use the fixture
command in `docs/adapter/xiaoyan_adapter_contract.md`.

```bash
MARKET_AGENT_LIVE_ADAPTER_BASE_URL=http://127.0.0.1:18112 \
MARKET_AGENT_LIVE_ADAPTER_API_KEY=scope-secret \
uv run python scripts/smoke_reply_live_adapter.py --message "请发一下周报"
```

## LLM configuration

```bash
YANFU_LLM_BASE_URL=https://llm.yanfuinvest.com/v1
YANFU_LLM_PROVIDER=openai
YANFU_LLM_MODEL=deepseek-v4-pro
YANFU_LLM_API_KEY=your-key
YANFU_LLM_TIMEOUT_SECONDS=60
YANFU_LLM_MAX_TOKENS=3000
CREWAI_MAX_RETRY_LIMIT=2
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

## Adapter resolve/preflight configuration

```bash
MARKET_AGENT_ADAPTER_BASE_URL=http://127.0.0.1:8011
MARKET_AGENT_ADAPTER_API_KEY=
MARKET_AGENT_ADAPTER_TIMEOUT_SECONDS=5
MARKET_AGENT_ADAPTER_PREFLIGHT_ENABLED=true
```

## Document MCP configuration

Document MCP access is configured separately from CrewAI agent prompts. It is disabled by default and is used only
through the fixed document evidence wrapper when explicitly enabled.

```bash
MARKET_AGENT_DOC_MCP_BASE_URL=http://192.168.209.195:23000
MARKET_AGENT_DOC_MCP_TIMEOUT_SECONDS=5
MARKET_AGENT_DOC_MCP_ENABLED=false
MARKET_AGENT_DOC_MCP_ALLOWED_CHANNEL_TYPES=bank,non_bank
```

The current document MCP server responds as streamable HTTP on `/mcp`, requires `Accept: application/json, text/event-stream`,
and exposes wrapper-only tools `list_products` and `get_documents`.

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

The public runtime response is `ReplyResponse { reply, actions }`. The adapter executes `reply` and typed side-effect
action proposals after its own validation.

## Documentation map

- `AGENTS.md`: short operational contract for coding agents.
- `docs/support_reply_harness/README.md`: harness doc index and active source-of-truth map.
- `docs/support_reply_harness/adr/0001-support-reply-harness.md`: architecture decision record.
- `docs/support_reply_harness/architecture.md`: runtime shape, source hierarchy, and internal concepts.
- `docs/support_reply_harness/guardrails.md`: deterministic guardrail pipeline and validator behavior.
- `docs/support_reply_harness/eval_plan.md`: regression, adversarial, and golden eval plan.
- `docs/support_reply_harness/roadmap.md`: phased implementation plan.
- `docs/support_reply_harness/next_session.md`: immediate coding-session handoff.
- `docs/support_reply_harness/reference/agent_prompt_hygiene.md`: prompt/context hygiene for Codex-style coding agents.
- `docs/adapter/xiaoyan_adapter_contract.md`: xiaoyan WeCom adapter contract and live smoke commands.

## Current module ownership

```text
src/market_support_crewai_agent/server/main.py          FastAPI routes only
src/market_support_crewai_agent/schemas.py              HTTP and action contracts
src/market_support_crewai_agent/runtime/reply_agent.py  CrewAI runtime orchestration
src/market_support_crewai_agent/runtime/conversation_store.py
src/market_support_crewai_agent/runtime/canonicalization.py
src/market_support_crewai_agent/runtime/adapter_preflight.py
src/market_support_crewai_agent/runtime/policy.py
src/market_support_crewai_agent/runtime/planning.py
src/market_support_crewai_agent/runtime/evidence.py
src/market_support_crewai_agent/runtime/evidence_executor.py
src/market_support_crewai_agent/runtime/business_facts.py
src/market_support_crewai_agent/runtime/guardrails.py
src/market_support_crewai_agent/runtime/action_ledger.py
src/market_support_crewai_agent/runtime/audit.py
src/market_support_crewai_agent/runtime/response_ids.py
```

Add new runtime-only harness modules under `src/market_support_crewai_agent/runtime/` as described in `AGENTS.md` and
`docs/support_reply_harness/next_session.md`.
