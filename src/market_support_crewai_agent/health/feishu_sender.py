from __future__ import annotations

import json
from collections.abc import Mapping
from types import TracebackType
from typing import Protocol, Self, final
from urllib import request

from pydantic import JsonValue, TypeAdapter

from market_support_crewai_agent.settings_model import Settings

_JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(
    dict[str, JsonValue]
)


class _FeishuResponse(Protocol):
    def read(self) -> bytes: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...


class _FeishuOpener(Protocol):
    def open(
        self,
        fullurl: str | request.Request,
        data: bytes | None = None,
        timeout: float | None = None,
    ) -> _FeishuResponse: ...


@final
class FeishuTextSender:
    def __init__(self, settings: Settings) -> None:
        self.app_id: str | None = settings.feishu_app_id
        self.app_secret: str | None = settings.feishu_app_secret
        self.chat_id: str | None = settings.feishu_chat_id
        self.timeout_seconds: int = 10
        self._opener: _FeishuOpener = request.build_opener(request.ProxyHandler({}))

    @property
    def enabled(self) -> bool:
        return bool(self.app_id and self.app_secret and self.chat_id)

    def send_text(self, text: str) -> None:
        if not self.enabled:
            return
        token = self._tenant_access_token()
        payload = {
            "receive_id": self.chat_id,
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        _ = self._post_json(
            "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id",
            payload,
            headers={"Authorization": f"Bearer {token}"},
        )

    def _tenant_access_token(self) -> str:
        data = self._post_json(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/",
            {"app_id": self.app_id, "app_secret": self.app_secret},
        )
        token = data.get("tenant_access_token")
        if not token:
            raise RuntimeError(f"Feishu token missing: {data}")
        return str(token)

    def _post_json(
        self,
        url: str,
        payload: Mapping[str, JsonValue],
        *,
        headers: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                **(headers or {}),
            },
            method="POST",
        )
        with self._opener.open(http_request, timeout=self.timeout_seconds) as response:
            raw_bytes: bytes = response.read()
        data = _JSON_OBJECT_ADAPTER.validate_json(raw_bytes.decode("utf-8"))
        if data.get("code", 0) != 0:
            raise RuntimeError(f"Feishu API error: {data}")
        return data
