from __future__ import annotations

import argparse
import json
import os
import sys
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
        description="Run a real LLM-backed /reply knowledge eval through FastAPI TestClient."
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
        "context_id": "real-llm-knowledge-eval-1",
        "conversation_key": "wecom:real-llm-knowledge-eval-group:real-llm-knowledge-eval-sender",
        "group_id": "real-llm-knowledge-eval-group",
        "sender_id": "real-llm-knowledge-eval-sender",
        "message": args.message,
        "is_group": True,
        "group_name": "real llm knowledge eval group",
        "dist_channel_name": "测试渠道",
        "sender_nickname": "测试用户",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }

    response = TestClient(app).post("/reply", json=payload)
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw": response.text}
    print("STATUS", response.status_code)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    reply = payload.get("reply") if isinstance(payload, dict) else {}
    actions = payload.get("actions") if isinstance(payload, dict) else None
    if (
        response.status_code != 200
        or not isinstance(reply, dict)
        or reply.get("kind") != "answer"
        or not str(reply.get("text") or "").strip()
        or actions != []
    ):
        print(
            json.dumps(
                {
                    "failure": "expected answer with non-empty text and no actions",
                    "status_code": response.status_code,
                    "payload": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
