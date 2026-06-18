"""Real-LLM eval over the labeled Xiaoyan question set.

Runs each question in ``tests/fixtures/xiaoyan_question_set.py`` through the
real ``/reply`` path (real planner LLM + Document MCP) and checks the observed
reply kind / actions against the label's expected behavior. Reports per-label
pass rates and prints every mismatch so recall and refusal regressions are
visible.

This is a manual diagnostic (not a CI test): it needs YANFU_LLM_API_KEY and an
enabled Document MCP; outbound ``action`` items also need the adapter running to
resolve a real send.

Examples:
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --parallel 8
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --live-adapter
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --label refuse_unsafe
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --ids 26,87,106 --show-text
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
    ACTION_LABELS,
    EXPECTED_REPLY_KINDS,
    LABELS,
    QUESTION_SET,
    Question,
)


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
        "available_materials": ["material", "weekly", "monthly"],
        "material_pack_options": [
            "中性",
            "灵活对冲",
            "沪深300指增",
            "中证500指增",
            "中证1000指增",
            "中证A500指增",
            "中证全指",
            "万得小市值",
        ],
        "channel_type": "non_bank",
    }


def _is_pass(label: str, kind: str, actions: list) -> bool:
    if label in ACTION_LABELS:
        return kind in EXPECTED_REPLY_KINDS[label] and bool(actions)
    if label == "refuse_unsafe":
        # The real failure is a substantive answer or any outbound send.
        return kind in EXPECTED_REPLY_KINDS[label] and not actions
    return kind in EXPECTED_REPLY_KINDS[label] and not actions


def _post_reply(client: TestClient, payload: dict) -> tuple[int, str, list, str]:
    """POST /reply, retrying once on a transient non-200 / empty reply so a
    flaky LLM stage is not scored as a label failure."""
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


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=LABELS, help="Only run one label.")
    parser.add_argument("--ids", help="Comma-separated question ids to run.")
    parser.add_argument("--limit", type=int, help="Cap the number of questions.")
    parser.add_argument(
        "--show-text", action="store_true", help="Print each reply text."
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=int(os.getenv("XIAOYAN_EVAL_PARALLEL", "4")),
        help="Concurrent /reply calls. Use 1 for the old sequential behavior.",
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
    parallel = max(1, args.parallel)
    by_label_total: dict[str, int] = defaultdict(int)
    by_label_pass: dict[str, int] = defaultdict(int)
    mismatches: list[str] = []

    def run_one(item: Question) -> tuple[Question, int, str, list, str]:
        status, kind, actions, text = _post_reply(_thread_client(app), _payload(item))
        return item, status, kind, actions, text

    if parallel == 1:
        results = map(run_one, items)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            results = list(pool.map(run_one, items))

    for item, status, kind, actions, text in results:
        ok = status == 200 and _is_pass(item.label, kind, actions)
        by_label_total[item.label] += 1
        by_label_pass[item.label] += int(ok)
        flag = "ok  " if ok else "MISS"
        line = (
            f"[{flag}] #{item.id:>3} {item.label:<14} -> kind={kind or '-':<16}"
            f" actions={len(actions)}  {item.question[:32]}"
        )
        print(line)
        if args.show_text:
            print(f"        text: {text.strip()[:160]}")
        if not ok:
            mismatches.append(
                f"#{item.id} [{item.label}] expected {EXPECTED_REPLY_KINDS[item.label]}"
                f" got status={status} kind={kind!r} actions={len(actions)} :: {item.question}"
                + (f"  ({item.note})" if item.note else "")
            )

    print("\n===== per-label pass rate =====")
    for label in LABELS:
        total = by_label_total.get(label, 0)
        if not total:
            continue
        passed = by_label_pass.get(label, 0)
        print(f"  {label:<14} {passed:>3}/{total:<3} ({passed / total:.0%})")
    total_all = sum(by_label_total.values())
    pass_all = sum(by_label_pass.values())
    if total_all:
        print(f"  {'TOTAL':<14} {pass_all:>3}/{total_all:<3} ({pass_all / total_all:.0%})")

    if mismatches:
        print("\n===== mismatches (review) =====")
        for line in mismatches:
            print(" -", line)


if __name__ == "__main__":
    main()
