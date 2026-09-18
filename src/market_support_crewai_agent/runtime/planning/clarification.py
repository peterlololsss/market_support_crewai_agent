from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ClarificationSpec:
    slot: str
    blocked_decision: str
    reason_text: str
    question_text: str
    candidate_prefix: str = ""


_SPECS: Final[dict[str, ClarificationSpec]] = {
    "artifact": ClarificationSpec(
        slot="artifact",
        blocked_decision="which supported artifact/action type to use",
        reason_text="我需要再确认你要的是材料包、周报、月报，还是产品/策略信息。",
        question_text="请确认你要处理的是材料包、周报、月报，还是产品/策略信息。",
        candidate_prefix="请确认你要处理哪类内容",
    ),
    "material_pack_option": ClarificationSpec(
        slot="material_pack_option",
        blocked_decision="which material-pack option to send",
        reason_text="我需要再确认你要的是哪一个材料包选项。",
        question_text="请确认你要的是哪一个材料包选项。",
        candidate_prefix="请确认你要的是哪一个材料包选项",
    ),
}

CLARIFICATION_PRIORITY: Final = ("artifact", "material_pack_option")
SUPPORTED_CLARIFICATION_SLOTS: Final = frozenset(_SPECS)


def supported_clarification_slots(values: Iterable[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        slot = str(value).strip()
        if slot in SUPPORTED_CLARIFICATION_SLOTS and slot not in output:
            output.append(slot)
    return output


def clarification_spec(slots: Iterable[str]) -> ClarificationSpec | None:
    slot_set = set(slots)
    for slot in CLARIFICATION_PRIORITY:
        if slot in slot_set:
            return _SPECS[slot]
    return None
