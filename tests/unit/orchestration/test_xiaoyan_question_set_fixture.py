"""Integrity guards for the Xiaoyan final-output review question set.

This keeps the manual eval corpus loadable without pretending the rough labels
are an intent oracle. Review the final runtime output with
``scripts/eval_reply_xiaoyan_question_set.py``.
"""

from __future__ import annotations

from datetime import datetime

from tests.fixtures.xiaoyan_question_set import (
    LABELS,
    QUESTION_SET,
    Question,
    questions_for_label,
)
from scripts.eval_reply_xiaoyan_question_set import (
    MAX_PARALLEL,
    _bounded_parallel,
    _payload,
    _write_markdown_report,
)


def test_question_set_is_non_trivial():
    assert len(QUESTION_SET) >= 100


def test_every_question_has_unique_id_and_text():
    ids = [item.id for item in QUESTION_SET]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    texts = [item.question.strip() for item in QUESTION_SET]
    assert all(texts), "empty question text"
    assert len(texts) == len(set(texts)), "duplicate question text"


def test_every_label_is_a_known_review_bucket():
    for item in QUESTION_SET:
        assert item.label in LABELS, f"unknown label {item.label!r} on id {item.id}"


def test_questions_for_label_filters_without_scoring_intent():
    for label in LABELS:
        assert all(item.label == label for item in questions_for_label(label))


def test_question_dataclass_is_frozen():
    item = QUESTION_SET[0]
    assert isinstance(item, Question)
    try:
        item.label = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Question should be immutable")


def test_eval_parallel_is_capped_to_avoid_dos():
    assert _bounded_parallel(0) == 1
    assert _bounded_parallel(MAX_PARALLEL) == MAX_PARALLEL
    assert _bounded_parallel(MAX_PARALLEL + 1) == MAX_PARALLEL


def test_eval_payload_defaults_to_galaxy_channel(monkeypatch):
    monkeypatch.delenv("MARKET_AGENT_LIVE_ADAPTER_DIST_NAME", raising=False)
    monkeypatch.delenv("MARKET_AGENT_EVAL_DIST_CHANNEL_NAME", raising=False)

    payload = _payload(Question(id=1, question="问题", label="brand_grounded"))

    assert payload["dist_channel_name"] == "银河证券"
    assert payload["group_id"] == "R:208304695202088"
    assert payload["conversation_key"].endswith(":q1")
    assert "resolve_weekly_report" in payload["allowed_read_capabilities"]


def test_eval_can_write_markdown_report(tmp_path):
    report = tmp_path / "report.md"
    item = Question(id=1, question="请发一下周报", label="action")

    _write_markdown_report(
        report,
        results=[(item, 200, "answer", [{"type": "send_weekly_report"}], "")],
        started=datetime(2026, 6, 22, 22, 0, 0),
        finished=datetime(2026, 6, 22, 22, 1, 0),
        parallel=1,
        live_adapter=False,
    )

    text = report.read_text(encoding="utf-8")
    assert "# Xiaoyan Question Set Eval Report" in text
    assert "Q:\n\n请发一下周报" in text
    assert "actions: send_weekly_report" in text
