from __future__ import annotations

from dataclasses import dataclass
from typing import override


@dataclass(frozen=True, slots=True)
class ReplyContractError(RuntimeError):
    detail: str

    @override
    def __str__(self) -> str:
        return self.detail
