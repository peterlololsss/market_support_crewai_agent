from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real LLM-backed /reply smoke through FastAPI TestClient."
    )
    parser.add_argument(
        "--message",
        default="介绍一下中证1000指数增强策略的因子贡献",
        help="Chinese user message to send to /reply.",
    )
    args = parser.parse_args()

    _load_dotenv()

    from market_support_crewai_agent.server.main import app

    payload = {
        "context_id": "real-llm-smoke-1",
        "conversation_key": "wecom:real-llm-smoke-group:real-llm-smoke-sender",
        "group_id": "real-llm-smoke-group",
        "sender_id": "real-llm-smoke-sender",
        "message": args.message,
        "is_group": True,
        "group_name": "real llm smoke group",
        "dist_channel_name": "测试渠道",
        "sender_nickname": "测试用户",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }

    response = TestClient(app).post("/reply", json=payload)
    print("STATUS", response.status_code)
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
