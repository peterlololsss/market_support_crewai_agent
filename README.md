# market-support-crewai-agent

External CrewAI runtime service for the existing WeWork bot.

Current scope:

- FastAPI transport layer
- `GET /health`
- `POST /reply`
- CrewAI runtime boundary
- typed request/response schema
- in-memory conversation history keyed by `conversation_key`

The service does not send WeWork messages directly. It returns `text` plus typed
actions for the existing WeWork bot to execute.

Run locally:

```bash
uv sync --extra dev
uv run uvicorn market_support_crewai_agent.server.main:app --reload
```

LLM configuration:

```bash
YANFU_LLM_BASE_URL=https://llm.yanfuinvest.com/v1
YANFU_LLM_PROVIDER=openai
YANFU_LLM_MODEL=deepseek-v4-pro
YANFU_LLM_API_KEY=your-key
```

Conversation history configuration is cold-loaded from environment at service
startup:

```bash
AGENT_CONVERSATION_TTL_SECONDS=86400
AGENT_CONVERSATION_MAX_MESSAGES=12
AGENT_CONVERSATION_MAX_SESSIONS=5000
AGENT_CONVERSATION_CLEANUP_INTERVAL_SECONDS=300
```

`POST /reply` requires `conversation_key`, `group_id`, `sender_id`, `message`,
and `is_group`. For group-chat requests, the gateway should send
`conversation_key` as `wecom:{group_id}:{sender_id}` and keep trigger detection
fields out of the payload. `context_id` is optional and only used for tracing.

Architecture:

- `src/market_support_crewai_agent/server/main.py`: FastAPI routes only.
- `src/market_support_crewai_agent/schemas.py`: HTTP and action contracts.
- `src/market_support_crewai_agent/runtime/reply_agent.py`: CrewAI runtime. The
  runtime owns reasoning and returns the typed `ReplyResponse`.
- `src/market_support_crewai_agent/runtime/conversation_store.py`: thread-safe
  in-memory conversation history with TTL, message trimming, and session caps.
