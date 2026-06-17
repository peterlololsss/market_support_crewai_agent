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
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --label refuse_unsafe
    uv run --extra dev python scripts/eval_reply_xiaoyan_question_set.py --ids 26,87,106 --show-text
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
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
        "available_strategies": [
            "中性",
            "灵活对冲",
            "沪深300",
            "中证500",
            "中证1000",
            "中证A500",
            "中证全指",
            "万得小市值",
        ],
        "channel_type": "non_bank",
    }


def _is_pass(label: str, kind: str, actions: list) -> bool:
    if label in ACTION_LABELS:
        # A resolved send, or a safe inability to send.
        return bool(actions) or kind in {"unable_to_answer", "clarification"}
    if label == "refuse_unsafe":
        # The real failure is a substantive answer or any outbound send.
        return kind in EXPECTED_REPLY_KINDS[label] and not actions
    return kind in EXPECTED_REPLY_KINDS[label]


def _post_reply(client: TestClient, payload: dict) -> tuple[int, str, list, str]:
    """POST /reply, retrying once on a transient non-200 / empty reply so a
    flaky LLM stage is not scored as a label failure."""
    status, kind, actions, text = 0, "", [], ""
    for attempt in range(2):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=LABELS, help="Only run one label.")
    parser.add_argument("--ids", help="Comma-separated question ids to run.")
    parser.add_argument("--limit", type=int, help="Cap the number of questions.")
    parser.add_argument(
        "--show-text", action="store_true", help="Print each reply text."
    )
    args = parser.parse_args()

    _load_dotenv()
    from market_support_crewai_agent.server.main import app

    client = TestClient(app)
    items = _selected(args)
    by_label_total: dict[str, int] = defaultdict(int)
    by_label_pass: dict[str, int] = defaultdict(int)
    mismatches: list[str] = []

    for item in items:
        status, kind, actions, text = _post_reply(client, _payload(item))
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
