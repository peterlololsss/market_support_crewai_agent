from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


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


class FakePreflightService:
    def __init__(self, status_by_type: dict[str, str] | None = None) -> None:
        self.status_by_type = status_by_type or {}

    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_strategies=None,
    ):
        del canonical_context
        from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
            AdapterPreflightItem,
            AdapterPreflightSnapshot,
        )
        from market_support_crewai_agent.schemas import AdapterResolveResult

        resolve_strategies = resolve_strategies or {}
        requested = resolve_types or [
            "material_pack",
            "weekly_report",
            "monthly_report",
            "sales_mention",
        ]
        items = []
        for resolve_type in requested:
            status = self.status_by_type.get(resolve_type, "resolved")
            strategy = resolve_strategies.get(resolve_type)
            items.append(
                AdapterPreflightItem(
                    resolve_type=resolve_type,
                    result=AdapterResolveResult.model_validate(
                        {
                            "contract_version": "adapter-resolve",
                            "resolve_type": resolve_type,
                            "status": status,
                            "display_name": request.dist_channel_name,
                            "reason_code": "ok" if status == "resolved" else "not_found",
                            "candidates": request.available_strategies,
                            "channel_type": request.channel_type,
                            "available_materials": request.available_materials,
                            "available_strategies": request.available_strategies,
                            "resolved_at": 1,
                            "resolve_ref": (
                                f"{resolve_type}:eval-ref"
                                if status == "resolved"
                                else None
                            ),
                            "strategy": strategy,
                            "period": (
                                "20260529"
                                if resolve_type == "weekly_report"
                                else None
                            ),
                            "scope_status": "unknown",
                        }
                    ),
                )
            )
        return AdapterPreflightSnapshot(items=items)


def _request(message: str, **overrides):
    from market_support_crewai_agent.schemas import ReplyRequest

    payload = {
        "context_id": "real-handoff-eval-1",
        "conversation_key": "wecom:real-handoff-eval-group:real-handoff-eval-sender",
        "group_id": "real-handoff-eval-group",
        "sender_id": "real-handoff-eval-sender",
        "message": message,
        "is_group": True,
        "group_name": "real handoff eval group",
        "dist_channel_name": "测试渠道",
        "sender_nickname": "测试用户",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


async def _run_scenario(name: str, request, preflight_service):
    from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
    from market_support_crewai_agent.runtime.state.audit import AuditStore
    from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
    from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
    from market_support_crewai_agent.settings import get_settings

    runtime = CrewAIReplyRuntime(
        get_settings(),
        conversation_store=ConversationStore(),
        action_ledger=ActionLedger(),
        preflight_service=preflight_service,
        audit_store=AuditStore(),
    )
    response = await runtime.reply(request)
    return {
        "scenario": name,
        "message": request.message,
        "response": response.model_dump(mode="json", exclude_none=True),
    }


async def main() -> None:
    _load_dotenv()
    scenarios = [
        (
            "customer_service_handoff",
            _request("我要找顾总帮忙对接一下", available_strategies=[]),
            FakePreflightService(),
        ),
        (
            "unavailable_material_pack_handoff",
            _request("请发一下中证1000材料包", available_strategies=["中证1000"]),
            FakePreflightService({"material_pack": "missing"}),
        ),
    ]
    results = [
        await _run_scenario(name, request, preflight_service)
        for name, request, preflight_service in scenarios
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    failures = _validate_results(results)
    if failures:
        print(
            json.dumps({"failures": failures}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        raise SystemExit(1)


def _validate_results(results: list[dict]) -> list[dict]:
    failures = []
    by_name = {result["scenario"]: result for result in results}

    service = by_name.get("customer_service_handoff", {})
    service_response = service.get("response", {})
    if not _is_sales_handoff(service_response):
        failures.append(
            {
                "scenario": "customer_service_handoff",
                "reason": "expected human_handoff with sales mention and no actions",
                "response": service_response,
            }
        )

    unavailable = by_name.get("unavailable_material_pack_handoff", {})
    unavailable_response = unavailable.get("response", {})
    if not _is_sales_handoff(unavailable_response):
        failures.append(
            {
                "scenario": "unavailable_material_pack_handoff",
                "reason": "expected unavailable material to fall back to sales handoff with no actions",
                "response": unavailable_response,
            }
        )
    return failures


def _is_sales_handoff(response: dict) -> bool:
    reply = response.get("reply", {})
    mentions = reply.get("mentions") or []
    return (
        reply.get("kind") == "human_handoff"
        and any(mention.get("type") == "sales" for mention in mentions)
        and response.get("actions") == []
    )


if __name__ == "__main__":
    asyncio.run(main())
