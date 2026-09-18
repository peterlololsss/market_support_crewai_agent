from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from market_support_crewai_agent.runtime.identity import VerifiedRequestEnvelopeV1
from market_support_crewai_agent.schemas.reply import PrimaryReply, ReplyResponse
from market_support_crewai_agent.server import main as server_main

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / ".omo" / "evidence" / "task-02-no-legacy-boundary"


def _group_payload() -> dict[str, Any]:
    return {
        "contract_version": "reply-request.v2",
        "request_id": "req:boundary-group-1",
        "message": "请发材料包",
        "context_id": "ctx:boundary-1",
        "identity": {
            "contract_version": "conversation-identity.v1",
            "surface": "wecom",
            "scene": "group",
            "tenant_ref": "tenant:tenant-1",
            "group_ref": "group:group-1",
            "principal_ref": "principal:principal-1",
        },
        "presentation": {
            "contract_version": "group-presentation.v1",
            "conversation_name": "测试群",
            "principal_name": "测试用户",
        },
        "business_scope": {
            "kind": "distribution",
            "dist_channel_name": "测试渠道",
            "channel_type": "bank",
            "available_artifacts": [{"type": "material_pack", "options": ["中证500"]}],
        },
        "grants": {
            "contract_version": "principal-grants.v1",
            "read_capabilities": ["resolve_material_pack"],
            "outbound_actions": ["send_material_pack"],
            "mention_types": [],
        },
    }


def _direct_payload() -> dict[str, Any]:
    payload = _group_payload()
    payload["request_id"] = "req:boundary-direct-1"
    payload["identity"] = {
        "contract_version": "conversation-identity.v1",
        "surface": "wecom",
        "scene": "direct",
        "tenant_ref": "tenant:tenant-1",
        "direct_thread_ref": "direct:thread-1",
        "principal_ref": "principal:principal-1",
    }
    payload["presentation"] = {
        "contract_version": "direct-presentation.v1",
        "principal_name": "测试用户",
    }
    payload["business_scope"] = {"kind": "unscoped"}
    payload["grants"] = {
        "contract_version": "principal-grants.v1",
        "read_capabilities": ["query_internal_company_info"],
        "outbound_actions": [],
        "mention_types": [],
    }
    return payload


def main() -> int:
    os.environ["MARKET_AGENT_API_KEY"] = "boundary-key"
    os.environ["MARKET_AGENT_DEPLOYMENT_TENANT_REF"] = "tenant:tenant-1"
    calls: list[dict[str, Any]] = []

    async def fake_build_reply(request: VerifiedRequestEnvelopeV1):
        calls.append({"scene": request.request.identity.scene})
        return ReplyResponse(reply=PrimaryReply(kind="answer", text="ok"), actions=[])

    original = server_main.build_reply
    server_main.build_reply = fake_build_reply
    try:
        client = TestClient(server_main.app)
        headers = {"X-API-Key": "boundary-key"}
        results: dict[str, Any] = {}

        ok = client.post("/reply", json=_group_payload(), headers=headers)
        results["v2_group"] = {
            "status": ok.status_code,
            "body": ok.json(),
            "calls": len(calls),
        }

        direct = client.post("/reply", json=_direct_payload(), headers=headers)
        results["v2_direct_disabled"] = {
            "status": direct.status_code,
            "body": direct.json(),
            "calls": len(calls),
        }

        invalid_cases = {
            "legacy_payload": {
                "conversation_key": "wecom:g:s",
                "group_id": "g",
                "sender_id": "s",
                "message": "hello",
                "is_group": True,
                "group_name": "g",
                "dist_channel_name": "d",
                "sender_nickname": "s",
                "available_artifacts": [],
                "channel_type": "bank",
            },
            "unknown_version": {
                **_group_payload(),
                "contract_version": "reply-request.v1",
            },
            "reserved_probe": {
                **_group_payload(),
                "identity": {
                    **_group_payload()["identity"],
                    "group_ref": "group:contract-probe",
                },
            },
        }
        before_invalid = len(calls)
        for name, payload in invalid_cases.items():
            response = client.post("/reply", json=payload, headers=headers)
            results[name] = {
                "status": response.status_code,
                "body": response.json(),
                "calls_after": len(calls),
            }
        results["zero_downstream_for_invalid"] = len(calls) == before_invalid

        if ok.status_code != 200:
            raise AssertionError(results)
        if (
            direct.status_code != 503
            or direct.json()["detail"]["code"] != "direct_reply_disabled"
        ):
            raise AssertionError(results)
        if not all(results[name]["status"] == 422 for name in invalid_cases):
            raise AssertionError(results)
        if not results["zero_downstream_for_invalid"]:
            raise AssertionError(results)

        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        (EVIDENCE_DIR / "task-02-boundary-failures.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    finally:
        server_main.build_reply = original


if __name__ == "__main__":
    raise SystemExit(main())
