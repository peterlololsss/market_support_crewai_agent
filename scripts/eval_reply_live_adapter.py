from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi.testclient import TestClient


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _adapter_base_url() -> str:
    return os.getenv(
        "MARKET_AGENT_LIVE_ADAPTER_BASE_URL",
        os.getenv("MARKET_AGENT_ADAPTER_BASE_URL", "http://127.0.0.1:8011"),
    ).rstrip("/")


def _adapter_api_key() -> str:
    return os.getenv(
        "MARKET_AGENT_LIVE_ADAPTER_API_KEY",
        os.getenv("MARKET_AGENT_ADAPTER_API_KEY", ""),
    )


def _assert_adapter_health(base_url: str, api_key: str) -> None:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{base_url}/health", headers=headers, method="GET")
    try:
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"adapter is not reachable at {base_url}: {exc}") from exc
    if payload.get("service") != "xiaoyan-wecom-market-agent-adapter":
        raise SystemExit(f"{base_url} is not the xiaoyan market-agent adapter")


def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run a real LLM-backed /reply eval with live adapter preflight."
    )
    parser.add_argument("--message", default="请发一下周报")
    parser.add_argument("--dist-name", default=os.getenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME", "ScopeTest"))
    parser.add_argument("--strategy", default=os.getenv("MARKET_AGENT_LIVE_ADAPTER_STRATEGY", "指增"))
    parser.add_argument("--channel-type", choices=["bank", "non_bank"], default="bank")
    parser.add_argument("--adapter-base-url", default=_adapter_base_url())
    parser.add_argument("--adapter-api-key", default=_adapter_api_key())
    parser.add_argument("--available-materials", default="material,weekly,monthly")
    parser.add_argument("--available-strategies", default="")
    parser.add_argument("--llm-timeout-seconds", default=os.getenv("YANFU_LLM_TIMEOUT_SECONDS", "90"))
    parser.add_argument("--llm-max-tokens", default=os.getenv("YANFU_LLM_MAX_TOKENS", "6000"))
    args = parser.parse_args()

    os.environ["MARKET_AGENT_ADAPTER_BASE_URL"] = args.adapter_base_url
    os.environ["YANFU_LLM_TIMEOUT_SECONDS"] = str(args.llm_timeout_seconds)
    os.environ["YANFU_LLM_MAX_TOKENS"] = str(args.llm_max_tokens)
    if args.adapter_api_key:
        os.environ["MARKET_AGENT_ADAPTER_API_KEY"] = args.adapter_api_key

    _assert_adapter_health(args.adapter_base_url, args.adapter_api_key)

    from market_support_crewai_agent.server.main import app

    available_strategies = _csv_values(args.available_strategies)
    if not available_strategies and args.strategy:
        available_strategies = [args.strategy]

    payload = {
        "context_id": "live-adapter-eval-1",
        "conversation_key": "wecom:live-adapter-eval-group:live-adapter-eval-sender",
        "group_id": "live-adapter-eval-group",
        "sender_id": "live-adapter-eval-sender",
        "message": args.message,
        "is_group": True,
        "group_name": f"{args.dist_name}-群",
        "dist_channel_name": args.dist_name,
        "sender_nickname": "live adapter eval user",
        "available_materials": _csv_values(args.available_materials),
        "available_strategies": available_strategies,
        "channel_type": args.channel_type,
    }

    response = TestClient(app).post("/reply", json=payload)
    print("STATUS", response.status_code)
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
