from __future__ import annotations


class ContextViewInvariantError(ValueError):
    __slots__ = ("code",)

    code: str

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
