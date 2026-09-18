from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    GeminiGenerateContentAdapterV1,
    GeminiModelsClientV1,
    GeminiSyncClientV1,
)
from market_support_crewai_agent.runtime.integrations.crewai.sdk_payload import (
    mapping_value,
    optional_int_value,
    optional_str_value,
    str_value,
)


def gemini_retry_attempts(payload: dict[str, JsonValue]) -> int | None:
    options = mapping_value(mapping_value(payload, "client_params"), "http_options")
    retry_options = mapping_value(options, "retry_options")
    return optional_int_value(retry_options, "attempts")


def gemini_sync_client_factory(
    payload: dict[str, JsonValue],
) -> Callable[[], GeminiSyncClientV1] | None:
    provider = str_value(payload, "provider", "openai").lower()
    if provider not in {"gemini", "google"}:
        return None

    api_key = optional_str_value(payload, "api_key")
    http_options = mapping_value(
        mapping_value(payload, "client_params"), "http_options"
    )
    base_url = optional_str_value(http_options, "base_url")
    api_version = optional_str_value(http_options, "api_version")
    timeout = optional_int_value(http_options, "timeout")
    retry_options = mapping_value(http_options, "retry_options")
    retry_attempts = optional_int_value(retry_options, "attempts")

    def sync_client() -> GeminiSyncClientV1:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                api_version=api_version,
                timeout=timeout,
                retry_options=types.HttpRetryOptions(attempts=retry_attempts),
            ),
        )

        def generate_content(request: GeminiGenerateContentAdapterV1) -> JsonValue:
            config = types.GenerateContentConfig.model_validate(
                request.config.model_dump(mode="json", exclude_none=True)
            )
            response = client.models.generate_content(
                model=request.model,
                contents=request.contents,
                config=config,
            )
            usage = response.usage_metadata
            return {
                "text": response.text or "",
                "usage_metadata": (
                    usage.model_dump(mode="json")
                    if isinstance(usage, BaseModel)
                    else None
                ),
            }

        return GeminiSyncClientV1(
            models=GeminiModelsClientV1(generate_content_fn=generate_content)
        )

    return sync_client
