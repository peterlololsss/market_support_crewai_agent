"""Integrity guards for the labeled Xiaoyan question set.

This keeps the labeled corpus well-formed and the taxonomy honest. The actual
runtime behavior is validated separately by
``scripts/eval_reply_xiaoyan_question_set.py`` (real LLM + Document MCP).
"""

from __future__ import annotations

from tests.fixtures.xiaoyan_question_set import (
    EXPECTED_REPLY_KINDS,
    LABELS,
    QUESTION_SET,
    Question,
    questions_for_label,
)

_VALID_REPLY_KINDS = {
    "answer",
    "clarification",
    "human_handoff",
    "unable_to_answer",
    "no_reply",
}


def test_question_set_is_non_trivial():
    assert len(QUESTION_SET) >= 100


def test_every_question_has_unique_id_and_text():
    ids = [item.id for item in QUESTION_SET]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    texts = [item.question.strip() for item in QUESTION_SET]
    assert all(texts), "empty question text"
    assert len(texts) == len(set(texts)), "duplicate question text"


def test_every_label_is_valid_and_used():
    for item in QUESTION_SET:
        assert item.label in LABELS, f"unknown label {item.label!r} on id {item.id}"
    used = {item.label for item in QUESTION_SET}
    assert used == set(LABELS), f"unused labels: {set(LABELS) - used}"


def test_expected_reply_kinds_cover_all_labels_and_are_valid():
    assert set(EXPECTED_REPLY_KINDS) == set(LABELS)
    for label, kinds in EXPECTED_REPLY_KINDS.items():
        assert kinds, f"no expected kinds for {label}"
        assert set(kinds) <= _VALID_REPLY_KINDS, f"invalid kinds for {label}: {kinds}"


def test_unsafe_questions_never_expect_a_plain_answer():
    # The whole point of the refuse_unsafe label: a substantive "answer" is a
    # failure, so it must not be an accepted outcome.
    assert "answer" not in EXPECTED_REPLY_KINDS["refuse_unsafe"]


def test_refuse_unsafe_covers_the_known_advice_and_guarantee_cases():
    unsafe = {item.question for item in questions_for_label("refuse_unsafe")}
    for needle in ("保证业绩", "给客户推哪个产品", "后市的研判", "自营盘规模", "止盈"):
        assert any(needle in q for q in unsafe), f"missing unsafe case: {needle}"


def test_holdings_questions_are_brand_grounded_not_refused():
    # Holdings/选股域 questions are answerable from the approved ratio statements,
    # so they must route to the grounded-answer path, not a flat refusal.
    holdings = [item for item in QUESTION_SET if "持仓" in item.question]
    assert holdings
    assert all(item.label == "brand_grounded" for item in holdings)


def test_send_requests_are_actions():
    send = [
        item
        for item in QUESTION_SET
        if any(k in item.question for k in ("材料包", "推介材料", "一页通", "开放日历"))
    ]
    assert send
    assert all(item.label == "action" for item in send)


def test_question_dataclass_is_frozen():
    item = QUESTION_SET[0]
    assert isinstance(item, Question)
    try:
        item.label = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Question should be immutable")
