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
    async def collect(
        self,
        request,
        canonical_context=None,
        resolve_types=None,
        resolve_strategies=None,
    ):
        del canonical_context
        from market_support_crewai_agent.runtime.adapter_preflight import (
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
            strategy = resolve_strategies.get(resolve_type)
            items.append(
                AdapterPreflightItem(
                    resolve_type=resolve_type,
                    result=AdapterResolveResult.model_validate(
                        {
                            "contract_version": "adapter-resolve",
                            "resolve_type": resolve_type,
                            "status": "resolved",
                            "display_name": request.dist_channel_name,
                            "reason_code": "ok",
                            "candidates": request.available_strategies,
                            "channel_type": request.channel_type,
                            "available_materials": request.available_materials,
                            "available_strategies": request.available_strategies,
                            "resolved_at": 1,
                            "resolve_ref": f"{resolve_type}:eval-ref",
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
        "context_id": "real-action-eval-1",
        "conversation_key": "wecom:real-action-eval-group:real-action-eval-sender",
        "group_id": "real-action-eval-group",
        "sender_id": "real-action-eval-sender",
        "message": message,
        "is_group": True,
        "group_name": "real action eval group",
        "dist_channel_name": "测试渠道",
        "sender_nickname": "测试用户",
        "available_materials": ["material", "weekly", "monthly"],
        "available_strategies": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


async def _run_scenario(name: str, request):
    from market_support_crewai_agent.runtime.action_ledger import ActionLedger
    from market_support_crewai_agent.runtime.audit import AuditStore
    from market_support_crewai_agent.runtime.conversation_store import ConversationStore
    from market_support_crewai_agent.runtime.reply_agent import CrewAIReplyRuntime
    from market_support_crewai_agent.settings import get_settings

    runtime = CrewAIReplyRuntime(
        get_settings(),
        conversation_store=ConversationStore(),
        action_ledger=ActionLedger(),
        preflight_service=FakePreflightService(),
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
            "weekly_report_action",
            _request("请发一下周报", available_strategies=[]),
        ),
        (
            "bank_material_requires_strategy_confirmation",
            _request("发一下材料包"),
        ),
        (
            "bank_material_strategy_action",
            _request("发一下中证1000材料包"),
        ),
    ]
    results = [await _run_scenario(name, request) for name, request in scenarios]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    failures = _validate_results(results)
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)


def _validate_results(results: list[dict]) -> list[dict]:
    failures = []
    by_name = {result["scenario"]: result for result in results}

    weekly = by_name.get("weekly_report_action", {})
    weekly_response = weekly.get("response", {})
    if _action_types(weekly_response) != ["send_weekly_report"] or weekly_response.get("reply", {}).get("text") != "":
        failures.append(
            {
                "scenario": "weekly_report_action",
                "reason": "expected empty reply text and one send_weekly_report action",
                "response": weekly_response,
            }
        )

    ambiguous = by_name.get("bank_material_requires_strategy_confirmation", {})
    ambiguous_response = ambiguous.get("response", {})
    if (
        ambiguous_response.get("reply", {}).get("kind") != "clarification"
        or ambiguous_response.get("actions") != []
    ):
        failures.append(
            {
                "scenario": "bank_material_requires_strategy_confirmation",
                "reason": "expected clarification with no actions",
                "response": ambiguous_response,
            }
        )

    material = by_name.get("bank_material_strategy_action", {})
    material_response = material.get("response", {})
    material_actions = material_response.get("actions") or []
    if (
        _action_types(material_response) != ["send_material_pack"]
        or material_actions[0].get("strategy") != "中证1000"
        or material_response.get("reply", {}).get("text") != ""
    ):
        failures.append(
            {
                "scenario": "bank_material_strategy_action",
                "reason": "expected empty reply text and one send_material_pack action for 中证1000",
                "response": material_response,
            }
        )
    return failures


def _action_types(response: dict) -> list[str]:
    return [action.get("type", "") for action in response.get("actions") or []]


if __name__ == "__main__":
    asyncio.run(main())
