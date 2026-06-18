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
        resolve_material_pack_options=None,
    ):
        del canonical_context
        from market_support_crewai_agent.runtime.evidence.adapter_preflight import (
            AdapterPreflightItem,
            AdapterPreflightSnapshot,
        )
        from market_support_crewai_agent.schemas import AdapterResolveResult

        resolve_material_pack_options = resolve_material_pack_options or {}
        requested = resolve_types or [
            "material_pack",
            "weekly_report",
            "monthly_report",
            "sales_mention",
        ]
        items = []
        for resolve_type in requested:
            material_pack_option = resolve_material_pack_options.get(resolve_type)
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
                            "candidates": request.material_pack_options,
                            "channel_type": request.channel_type,
                            "available_materials": request.available_materials,
                            "material_pack_options": request.material_pack_options,
                            "material_pack_option": material_pack_option,
                            "resolved_at": 1,
                            "resolve_ref": f"{resolve_type}:eval-ref",
                            "period": (
                                "20260529"
                                if resolve_type == "weekly_report"
                                else "202605"
                                if resolve_type == "monthly_report"
                                else None
                            ),
                            "report_date": (
                                "2026-05-29"
                                if resolve_type == "weekly_report"
                                else "2026-05-31"
                                if resolve_type == "monthly_report"
                                else None
                            ),
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
        "material_pack_options": ["中证500", "中证1000"],
        "channel_type": "bank",
    }
    payload.update(overrides)
    return ReplyRequest.model_validate(payload)


async def _run_scenario(name: str, request):
    from market_support_crewai_agent.runtime.state.action_ledger import ActionLedger
    from market_support_crewai_agent.runtime.state.audit import AuditStore
    from market_support_crewai_agent.runtime.state.conversation_store import ConversationStore
    from market_support_crewai_agent.runtime.orchestration.reply_agent import CrewAIReplyRuntime
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
            _request("请发一下周报", material_pack_options=[]),
        ),
        (
            "bank_material_default_action",
            _request("发一下材料包"),
        ),
        (
            "bank_material_strategy_action",
            _request("发一下中证1000材料包"),
        ),
        (
            "semantic_one_pager_material_action",
            _request("麻烦同步一下中证1000的一页通"),
        ),
        (
            "semantic_weekly_metric_action",
            _request("500最近回撤修复得怎么样"),
        ),
        (
            "knowledge_question_not_monthly_send",
            _request("月报里为什么没有年化收益率"),
        ),
        (
            "multi_artifact_clarification",
            _request("材料和周报都给我"),
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

    bank_material = by_name.get("bank_material_default_action", {})
    bank_material_response = bank_material.get("response", {})
    if (
        _action_types(bank_material_response) != ["send_material_pack"]
        or bank_material_response.get("reply", {}).get("text") != ""
    ):
        failures.append(
            {
                "scenario": "bank_material_default_action",
                "reason": "expected resolved bank material-pack request to send material_pack",
                "response": bank_material_response,
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

    one_pager = by_name.get("semantic_one_pager_material_action", {})
    one_pager_response = one_pager.get("response", {})
    one_pager_actions = one_pager_response.get("actions") or []
    if (
        _action_types(one_pager_response) != ["send_material_pack"]
        or one_pager_actions[0].get("strategy") != "中证1000"
        or one_pager_response.get("reply", {}).get("text") != ""
    ):
        failures.append(
            {
                "scenario": "semantic_one_pager_material_action",
                "reason": "expected one semantic one-pager request to produce send_material_pack for 中证1000",
                "response": one_pager_response,
            }
        )

    weekly_metric = by_name.get("semantic_weekly_metric_action", {})
    weekly_metric_response = weekly_metric.get("response", {})
    weekly_metric_actions = weekly_metric_response.get("actions") or []
    if (
        _action_types(weekly_metric_response) != ["send_weekly_report"]
        or weekly_metric_actions[0].get("strategy") != "中证500"
        or "最新周报" not in weekly_metric_response.get("reply", {}).get("text", "")
    ):
        failures.append(
            {
                "scenario": "semantic_weekly_metric_action",
                "reason": "expected recent drawdown-recovery metric question to send weekly report for 中证500 with rationale text",
                "response": weekly_metric_response,
            }
        )

    knowledge = by_name.get("knowledge_question_not_monthly_send", {})
    knowledge_response = knowledge.get("response", {})
    if "send_monthly_report" in _action_types(knowledge_response):
        failures.append(
            {
                "scenario": "knowledge_question_not_monthly_send",
                "reason": "expected report-format why question not to produce a monthly report send",
                "response": knowledge_response,
            }
        )

    multi_artifact = by_name.get("multi_artifact_clarification", {})
    multi_artifact_response = multi_artifact.get("response", {})
    if (
        multi_artifact_response.get("reply", {}).get("kind") != "clarification"
        or multi_artifact_response.get("actions") != []
    ):
        failures.append(
            {
                "scenario": "multi_artifact_clarification",
                "reason": "expected multi-artifact request to clarify with no actions",
                "response": multi_artifact_response,
            }
        )
    return failures


def _action_types(response: dict) -> list[str]:
    return [action.get("type", "") for action in response.get("actions") or []]


if __name__ == "__main__":
    asyncio.run(main())
