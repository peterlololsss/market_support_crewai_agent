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
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --output-md report.md
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
                            "available_artifacts": _available_artifacts_payload(
                                request
                            ),
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


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _payload(question: Question) -> dict:
    conversation_key_base = _env(
        "MARKET_AGENT_EVAL_CONVERSATION_KEY_BASE",
        "wecom:R:208304695202088:1688857180791030",
    )
    return {
        "context_id": f"xiaoyan-eval-{question.id}",
        "conversation_key": f"{conversation_key_base}:q{question.id}",
        "group_id": _env("MARKET_AGENT_EVAL_GROUP_ID", "R:208304695202088"),
        "sender_id": _env("MARKET_AGENT_EVAL_SENDER_ID", "1688857180791030"),
        "message": question.question,
        "is_group": True,
        "group_name": _env(
            "MARKET_AGENT_EVAL_GROUP_NAME",
            "银河证券-衍复投资沟通交流测试4",
        ),
        "dist_channel_name": os.getenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME")
        or _env("MARKET_AGENT_EVAL_DIST_CHANNEL_NAME", "银河证券"),
        "sender_nickname": _env("MARKET_AGENT_EVAL_SENDER_NICKNAME", "孙逸凡"),
        "available_artifacts": [
            {
                "type": "material_pack",
                "options": [],
            },
            {"type": "weekly_report"},
            {"type": "monthly_report"},
        ],
        "channel_type": "non_bank",
        "allowed_read_capabilities": [
            "query_internal_company_info",
            "resolve_material_pack",
            "resolve_monthly_report",
            "resolve_sales_mention",
            "resolve_weekly_report",
        ],
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


def _write_markdown_report(
    path: Path,
    *,
    results: list[tuple[Question, int, str, list, str]],
    started: datetime,
    finished: datetime,
    parallel: int,
    live_adapter: bool,
) -> None:
    sample_payload = _payload(results[0][0]) if results else {}
    by_label: dict[str, int] = defaultdict(int)
    by_status: dict[int, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    action_count = 0
    for item, status, kind, actions, _text in results:
        by_label[item.label] += 1
        by_status[status] += 1
        by_kind[kind or "-"] += 1
        action_count += len(actions)

    lines = [
        "# Xiaoyan Question Set Eval Report",
        "",
        f"Generated: {finished.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Started: {started.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Finished: {finished.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Questions: {len(results)}",
        f"Parallel: {parallel}",
        "Runner: in-process POST /reply via FastAPI TestClient",
        f"Adapter: {'live xiaoyan adapter' if live_adapter else 'fake preflight'}",
        "Document MCP / LLM: .env settings",
        f"dist_channel_name: `{sample_payload.get('dist_channel_name', '-')}`",
        f"group_id: `{sample_payload.get('group_id', '-')}`",
        f"group_name: {sample_payload.get('group_name', '-')}",
        f"sender_id: `{sample_payload.get('sender_id', '-')}`",
        "",
        "## Counts",
        "",
        "### By label",
        "",
    ]
    lines += [f"- {label}: {count}" for label, count in by_label.items()]
    lines += ["", "### By HTTP status", ""]
    lines += [f"- {status}: {count}" for status, count in sorted(by_status.items())]
    lines += ["", "### By reply kind", ""]
    lines += [f"- {kind}: {count}" for kind, count in by_kind.items()]
    lines += ["", f"Actions proposed: {action_count}", "", "## Results", ""]

    for item, status, kind, actions, text in results:
        flag = "ok" if status == 200 else "ERR"
        lines += [
            f"### [{flag}] #{item.id} {item.label}",
            "",
            f"- status: `{status}`",
            f"- kind: `{kind or '-'}`",
            f"- actions: {_summarize_actions(actions)}",
            "",
            "Q:",
            "",
            item.question,
            "",
            "A:",
            "",
            text.strip() or "<empty>",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-md",
        type=Path,
        help="Write a markdown report for manual quality review.",
    )
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

    from market_support_crewai_agent.server import main as server_main

    app = server_main.app
    if not args.live_adapter:
        original_build_reply = server_main.build_reply
        fake_preflight = FakePreflightService()

        async def fake_build_reply(request):
            return await original_build_reply(request, preflight_service=fake_preflight)

        server_main.build_reply = fake_build_reply

    items = _selected(args)
    parallel = _bounded_parallel(args.parallel)
    if parallel != args.parallel:
        print(f"parallel capped at {MAX_PARALLEL} to avoid hammering /reply")
    by_label_total: dict[str, int] = defaultdict(int)

    def run_one(item: Question) -> tuple[Question, int, str, list, str]:
        status, kind, actions, text = _post_reply(_thread_client(app), _payload(item))
        return item, status, kind, actions, text

    started = datetime.now().astimezone()
    if parallel == 1:
        results = list(map(run_one, items))
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            results = list(pool.map(run_one, items))
    finished = datetime.now().astimezone()

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
    if args.output_md:
        _write_markdown_report(
            args.output_md,
            results=results,
            started=started,
            finished=finished,
            parallel=parallel,
            live_adapter=args.live_adapter,
        )
        print(f"\nreport written: {args.output_md}")


if __name__ == "__main__":
    main()
