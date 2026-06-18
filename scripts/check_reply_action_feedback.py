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


def _feedback_payload(version: str, conversation_key: str) -> dict:
    return {
        "conversation_key": conversation_key,
        "group_id": "feedback-check-group",
        "sender_id": "feedback-check-sender",
        "context_id": "feedback-check-send-1",
        "response_id": "feedback-check-response-1",
        "executions": [
            {
                "action_type": "send_weekly_report",
                "status": "executed",
                "action_id": "send-weekly-1",
                "resolve_ref": "weekly:feedback-check-ref",
                "material_type": "weekly",
                "material_id": "weekly:feedback-check-ref",
                "version": version,
                "adapter_result": {"ok": True},
            }
        ],
    }


def _reply_payload(message: str, conversation_key: str, context_id: str) -> dict:
    return {
        "context_id": context_id,
        "conversation_key": conversation_key,
        "group_id": "feedback-check-group",
        "sender_id": "feedback-check-sender",
        "message": message,
        "is_group": True,
        "group_name": "feedback check group",
        "dist_channel_name": "测试渠道",
        "sender_nickname": "测试用户",
        "available_materials": ["material", "weekly", "monthly"],
        "material_pack_options": ["中证500", "中证1000"],
        "channel_type": "bank",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real /actions/feedback -> /reply ledger grounding check."
    )
    parser.add_argument("--version", default="20260529")
    parser.add_argument("--message", default="刚才发的周报是哪一版？")
    args = parser.parse_args()

    _load_dotenv()

    from market_support_crewai_agent.runtime.state.action_ledger import get_action_ledger
    from market_support_crewai_agent.server.main import app

    get_action_ledger().clear()
    client = TestClient(app)

    no_feedback_conversation_key = (
        "wecom:feedback-check-group:feedback-check-no-feedback-sender"
    )
    no_feedback_response = client.post(
        "/reply",
        json=_reply_payload(
            args.message,
            no_feedback_conversation_key,
            "feedback-check-no-feedback-query-1",
        ),
    )
    try:
        no_feedback_payload = no_feedback_response.json()
    except ValueError:
        no_feedback_payload = {"raw": no_feedback_response.text}

    no_feedback_reply = (
        no_feedback_payload.get("reply")
        if isinstance(no_feedback_payload, dict)
        else {}
    )
    no_feedback_actions = (
        no_feedback_payload.get("actions")
        if isinstance(no_feedback_payload, dict)
        else None
    )
    no_feedback_text = str(no_feedback_reply.get("text") or "")
    no_feedback_passed = (
        no_feedback_response.status_code == 200
        and args.version not in no_feedback_text
        and no_feedback_actions == []
    )

    executed_conversation_key = (
        "wecom:feedback-check-group:feedback-check-executed-sender"
    )
    feedback_response = client.post(
        "/actions/feedback",
        json=_feedback_payload(args.version, executed_conversation_key),
    )
    reply_response = client.post(
        "/reply",
        json=_reply_payload(
            args.message,
            executed_conversation_key,
            "feedback-check-executed-query-1",
        ),
    )
    try:
        reply_payload = reply_response.json()
    except ValueError:
        reply_payload = {"raw": reply_response.text}

    reply = reply_payload.get("reply") if isinstance(reply_payload, dict) else {}
    actions = reply_payload.get("actions") if isinstance(reply_payload, dict) else None
    text = str(reply.get("text") or "")
    executed_passed = (
        feedback_response.status_code == 200
        and feedback_response.json() == {"status": "accepted", "stored": 1}
        and reply_response.status_code == 200
        and args.version in text
        and actions == []
    )
    passed = no_feedback_passed and executed_passed
    result = {
        "passed": passed,
        "no_feedback": {
            "passed": no_feedback_passed,
            "reply_status_code": no_feedback_response.status_code,
            "reply": no_feedback_reply,
            "actions": no_feedback_actions,
        },
        "executed_feedback": {
            "passed": executed_passed,
            "feedback_status_code": feedback_response.status_code,
            "feedback": feedback_response.json(),
            "reply_status_code": reply_response.status_code,
            "reply": reply,
            "actions": actions,
        },
        "expected_version": args.version,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    get_action_ledger().clear()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise
