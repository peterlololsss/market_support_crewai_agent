from __future__ import annotations

import ipaddress
from typing import Final
from urllib.parse import urlsplit

from pydantic import JsonValue

from market_support_crewai_agent.runtime.prompts.profiles import ModelFamily
from market_support_crewai_agent.runtime.prompts.program_models import (
    ProviderIdV1,
    ProviderTargetConfigV1,
    ProviderTargetIdentityV1,
    TargetSlotV1,
    TransportVariantV1,
)


class ProviderTargetError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


INTERNAL_CREWAI_SDK_ENDPOINT: Final = "sdk://crewai"


def normalize_provider_id(provider: str) -> ProviderIdV1:
    normalized = provider.strip().lower()
    match normalized:
        case "gemini" | "google" | "google-genai" | "google_genai":
            return "gemini"
        case (
            "openai" | "deepseek" | "yanfu" | "openai-compatible" | "openai_compatible"
        ):
            return "openai_compatible"
        case _:
            raise ProviderTargetError("provider_variant_unknown")


def model_family_from_model_name(model_name: str) -> ModelFamily:
    normalized = model_name.lower()
    if (
        "deepseek-v4-pro" in normalized
        or "ds-v4pro" in normalized
        or "v4-pro" in normalized
    ):
        return "ds_v4pro"
    if "deepseek" in normalized:
        return "deepseek"
    if "gpt" in normalized:
        return "gpt"
    if "claude" in normalized:
        return "claude"
    return "generic"


def transport_variant_for_provider(provider_id: ProviderIdV1) -> TransportVariantV1:
    match provider_id:
        case "openai_compatible":
            return "openai_chat_completions"
        case "gemini":
            return "gemini_generate_content"


def normalize_provider_endpoint(base_url: str) -> str:
    value = base_url.strip(" \t\n\r\f\v")
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ProviderTargetError("provider_endpoint_scheme_invalid")
    if not parsed.netloc or parsed.hostname is None:
        raise ProviderTargetError("provider_endpoint_absolute_required")
    if parsed.username is not None or parsed.password is not None:
        raise ProviderTargetError("provider_endpoint_userinfo_forbidden")
    if parsed.query or parsed.fragment:
        raise ProviderTargetError("provider_endpoint_query_fragment_forbidden")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ProviderTargetError("provider_endpoint_port_invalid") from exc
    host = _normalize_host(parsed.hostname)
    if isinstance(host, ipaddress.IPv6Address):
        authority = f"[{host.compressed}]"
    else:
        authority = str(host)
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        authority = f"{authority}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return f"{scheme}://{authority}{path}"


def build_provider_target_identity(
    *,
    provider: str,
    model_name: str,
    base_url: str,
    target_slot: TargetSlotV1,
) -> ProviderTargetIdentityV1:
    provider_id = normalize_provider_id(provider)
    return ProviderTargetIdentityV1(
        provider_id=provider_id,
        target_slot=target_slot,
        model_family=model_family_from_model_name(model_name),
        model_name=model_name,
        normalized_endpoint=normalize_provider_endpoint(base_url),
        transport_variant=transport_variant_for_provider(provider_id),
    )


def build_provider_target_from_settings(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key_configured: bool,
    target_slot: TargetSlotV1,
    timeout_seconds: float,
    temperature: float,
    max_tokens: int,
) -> ProviderTargetConfigV1:
    identity = build_provider_target_identity(
        provider=provider,
        model_name=model,
        base_url=base_url,
        target_slot=target_slot,
    )
    return ProviderTargetConfigV1(
        **identity.model_dump(mode="python"),
        api_key_configured=api_key_configured,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_internal_crewai_sdk_target(
    *,
    provider: str,
    model: str,
    api_key_configured: bool,
    target_slot: TargetSlotV1,
    timeout_seconds: float,
    temperature: float,
    max_tokens: int,
    thinking_config: JsonValue | None = None,
) -> ProviderTargetConfigV1:
    provider_id = normalize_provider_id(provider)
    return ProviderTargetConfigV1(
        provider_id=provider_id,
        target_slot=target_slot,
        model_family=model_family_from_model_name(model),
        model_name=model,
        normalized_endpoint=INTERNAL_CREWAI_SDK_ENDPOINT,
        transport_variant=transport_variant_for_provider(provider_id),
        api_key_configured=api_key_configured,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking_config=thinking_config,
    )


def _normalize_host(host: str) -> str | ipaddress.IPv4Address | ipaddress.IPv6Address:
    normalized = host.rstrip(".").lower()
    if not normalized or host.endswith(".."):
        raise ProviderTargetError("provider_endpoint_host_invalid")
    try:
        return ipaddress.ip_address(normalized)
    except ValueError:
        if _looks_numeric_host(normalized):
            raise ProviderTargetError("provider_endpoint_host_invalid") from None
    try:
        return normalized.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ProviderTargetError("provider_endpoint_host_invalid") from exc


def _looks_numeric_host(host: str) -> bool:
    if host.startswith("0x") or host.isdecimal() or ":" in host:
        return True
    labels = host.split(".")
    return bool(labels) and all(
        label.isdecimal()
        or (len(label) > 1 and label.startswith("0") and label[1:].isdigit())
        or (label.startswith("0x") and len(label) > 2)
        for label in labels
    )
