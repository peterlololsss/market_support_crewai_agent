from __future__ import annotations

from pathlib import Path

from tests.fixtures.xiaoyan_question_set import QUESTION_SET

PROMPT_SURFACES = (
    Path("src/market_support_crewai_agent/runtime/llm/prompts/fragments/base/planner_intent_base.md"),
    Path("src/market_support_crewai_agent/runtime/llm/prompts/fragments/planner/intent_taxonomy.md"),
    Path("src/market_support_crewai_agent/runtime/domain/capabilities/manifests.py"),
    Path("tests/unit/llm/test_prompt_router.py"),
    Path("tests/snapshots/prompts/planner_intent_ds_v4pro.txt"),
)


def _fixture_hits(text: str) -> list[str]:
    return [question.question for question in QUESTION_SET if question.question in text]


def test_fixture_leak_detector_catches_exact_question_text():
    assert _fixture_hits("prefix " + QUESTION_SET[0].question + " suffix") == [
        QUESTION_SET[0].question
    ]


def test_active_prompt_surfaces_do_not_embed_xiaoyan_eval_questions():
    leaks: dict[str, list[str]] = {}
    for surface in PROMPT_SURFACES:
        hits = _fixture_hits(surface.read_text(encoding="utf-8"))
        if hits:
            leaks[str(surface)] = hits

    assert leaks == {}
