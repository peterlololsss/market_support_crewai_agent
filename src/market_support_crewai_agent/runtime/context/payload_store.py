from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import RLock
from typing import Any

from market_support_crewai_agent.runtime.context.models import stable_json


class ContextPayloadStore:
    def __init__(self, max_payloads: int = 128) -> None:
        if max_payloads <= 0:
            raise ValueError("max_payloads must be greater than zero")
        self._max_payloads = max_payloads
        self._payloads: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = RLock()

    def put(self, payload: Any, metadata: dict[str, Any] | None = None) -> str:
        serialized = stable_json({"payload": payload, "metadata": metadata or {}})
        handle = "ctx-payload:{}".format(
            hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        )
        with self._lock:
            self._payloads[handle] = {"payload": payload, "metadata": metadata or {}}
            self._payloads.move_to_end(handle)
            while len(self._payloads) > self._max_payloads:
                self._payloads.popitem(last=False)
        return handle

    def get(self, reload_handle: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._payloads.get(reload_handle)
            if value is not None:
                self._payloads.move_to_end(reload_handle)
            return value

    def count(self) -> int:
        with self._lock:
            return len(self._payloads)

    def clear(self) -> None:
        with self._lock:
            self._payloads.clear()
