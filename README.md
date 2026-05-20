# market-support-crewai-agent

External CrewAI runtime service for the existing WeWork bot.

Current scope:

- FastAPI transport layer
- `GET /health`
- `POST /reply`
- CrewAI runtime boundary
- typed request/response schema

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

Architecture:

- `src/market_support_crewai_agent/server/main.py`: FastAPI routes only.
- `src/market_support_crewai_agent/schemas.py`: HTTP and action contracts.
- `src/market_support_crewai_agent/runtime/reply_agent.py`: CrewAI runtime. The
  runtime owns reasoning and returns the typed `ReplyResponse`.
