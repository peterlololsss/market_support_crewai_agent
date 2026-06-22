"""Real-LLM final-output review over the Xiaoyan question set.

Runs each question in ``tests/fixtures/xiaoyan_question_set.py`` through the
real ``/reply`` path (real planner LLM + Document MCP) and prints the final
reply text plus outbound actions for review. The labels are rough filters only;
they are not used as a pass/fail oracle.

This is a manual diagnostic (not a CI test): it needs YANFU_LLM_API_KEY and an
enabled Document MCP; outbound ``action`` items also need the adapter running to
resolve a real send.

Examples:
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --parallel 4
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --live-adapter
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --label refuse_unsafe
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --ids 26,87,106
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

# Make the repo root importable so the labeled fixture under tests/ resolves
# when this script is run directly (pytest handles this itself).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures.xiaoyan_question_set import (  # noqa: E402
    LABELS,
    QUESTION_SET,
    Question,
)


MAX_PARALLEL = 4
_THREAD_LOCAL = threading.local()


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
                            "candidates": _material_pack_options(request),
                            "channel_type": request.channel_type,
                            "available_artifacts": _available_artifacts_payload(request),
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


def _payload(question: Question) -> dict:
    return {
        "context_id": f"xiaoyan-eval-{question.id}",
        "conversation_key": f"wecom:xiaoyan-eval:{question.id}",
        "group_id": "xiaoyan-eval-group",
        "sender_id": f"sender-{question.id}",
        "message": question.question,
        "is_group": True,
        "group_name": "xiaoyan eval group",
        "dist_channel_name": os.getenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME", "测试渠道"),
        "sender_nickname": "测试用户",
        "available_artifacts": [
            {
                "type": "material_pack",
                "options": [],
            },
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "non_bank",
    }


def _available_artifacts_payload(request) -> list[dict]:
    return [
        artifact.model_dump(mode="json", exclude_none=True)
        for artifact in request.available_artifacts
    ]


def _material_pack_options(request) -> list[str]:
    for artifact in request.available_artifacts:
        if artifact.type == "material_pack":
            return list(artifact.options)
    return []


def _post_reply(client: TestClient, payload: dict) -> tuple[int, str, list, str]:
    """POST /reply, retrying once on a transient non-200 / empty reply so a
    flaky LLM stage does not hide the final output."""
    status, kind, actions, text = 0, "", [], ""
    for _ in range(2):
        response = client.post("/reply", json=payload)
        body = response.json() or {}
        reply = body.get("reply") or {}
        status = response.status_code
        kind = str(reply.get("kind") or "")
        actions = body.get("actions") or []
        text = str(reply.get("text") or "")
        if status == 200 and kind:
            break
    return status, kind, actions, text


def _thread_client(app) -> TestClient:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = TestClient(app)
        _THREAD_LOCAL.client = client
    return client


def _selected(args: argparse.Namespace) -> list[Question]:
    items = list(QUESTION_SET)
    if args.label:
        items = [item for item in items if item.label == args.label]
    if args.ids:
        wanted = {int(piece) for piece in args.ids.split(",") if piece.strip()}
        items = [item for item in items if item.id in wanted]
    if args.limit:
        items = items[: args.limit]
    return items


def _bounded_parallel(value: int) -> int:
    return min(MAX_PARALLEL, max(1, value))


def _summarize_actions(actions: list) -> str:
    if not actions:
        return "-"
    parts = []
    for action in actions:
        fields = [
            str(action.get("type") or "?"),
            *(
                f"{key}={action[key]}"
                for key in ("material_pack_option", "period", "report_date")
                if action.get(key)
            ),
        ]
        parts.append(" ".join(fields))
    return "; ".join(parts)


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=LABELS, help="Only run one label.")
    parser.add_argument("--ids", help="Comma-separated question ids to run.")
    parser.add_argument("--limit", type=int, help="Cap the number of questions.")
    parser.add_argument("--show-text", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--parallel",
        type=int,
        default=int(os.getenv("XIAOYAN_EVAL_PARALLEL", "4")),
        help="Concurrent /reply calls, capped at 4 to avoid hammering dependencies.",
    )
    parser.add_argument(
        "--live-adapter",
        action="store_true",
        help="Use the configured live adapter instead of the default fake preflight.",
    )
    args = parser.parse_args()

    from market_support_crewai_agent.server.main import app
    if not args.live_adapter:
        from market_support_crewai_agent.runtime.orchestration import reply_agent

        reply_agent._DEFAULT_PREFLIGHT_SERVICE = FakePreflightService()

    items = _selected(args)
    parallel = _bounded_parallel(args.parallel)
    if parallel != args.parallel:
        print(f"parallel capped at {MAX_PARALLEL} to avoid hammering /reply")
    by_label_total: dict[str, int] = defaultdict(int)

    def run_one(item: Question) -> tuple[Question, int, str, list, str]:
        status, kind, actions, text = _post_reply(_thread_client(app), _payload(item))
        return item, status, kind, actions, text

    if parallel == 1:
        results = map(run_one, items)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            results = list(pool.map(run_one, items))

    for item, status, kind, actions, text in results:
        by_label_total[item.label] += 1
        flag = "ok " if status == 200 else "ERR"
        print(
            f"\n[{flag}] #{item.id:>3} {item.label:<14} "
            f"status={status} kind={kind or '-'}"
        )
        print(f"Q: {item.question}")
        print(f"A: {text.strip() or '<empty>'}")
        print(f"actions: {_summarize_actions(actions)}")

    print("\n===== review counts =====")
    for label in LABELS:
        total = by_label_total.get(label, 0)
        if not total:
            continue
        print(f"  {label:<14} {total:>3}")
    total_all = sum(by_label_total.values())
    if total_all:
        print(f"  {'TOTAL':<14} {total_all:>3}")


if __name__ == "__main__":
    main()
