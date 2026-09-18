from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qsl, unquote, urlsplit

from market_support_crewai_agent.settings_model import Settings

LocatorSafety = Literal["public_url", "internal_locator", "not_locator"]

_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "token",
        "secret",
        "signature",
        "sig",
        "authorization",
        "key",
    }
)


@dataclass(frozen=True, slots=True)
class LocatorSafetyClassifierV1:
    internal_origins: frozenset[tuple[str, str, int | None]]
    secrets: tuple[str, ...]

    @classmethod
    def from_settings(cls, settings: Settings) -> "LocatorSafetyClassifierV1":
        endpoints = (
            settings.adapter_base_url,
            settings.doc_mcp_base_url,
            settings.llm_base_url,
            settings.planner_llm_base_url,
        )
        return cls(
            internal_origins=frozenset(
                origin
                for endpoint in endpoints
                if endpoint is not None
                for origin in (_origin(endpoint),)
                if origin is not None
            ),
            secrets=tuple(
                secret
                for secret in (
                    settings.api_key,
                    settings.adapter_api_key,
                    settings.llm_api_key,
                    settings.planner_llm_api_key,
                    settings.feishu_app_secret,
                    settings.direct_audit_hmac_key,
                )
                if secret
            ),
        )

    def classify(self, value: str) -> LocatorSafety:
        if not value:
            return "not_locator"
        if any(secret in value for secret in self.secrets):
            return "internal_locator"
        decoded = unquote(value)
        if _path_like(decoded):
            return "internal_locator"
        parsed = urlsplit(value)
        if not parsed.scheme and not parsed.netloc:
            return "not_locator"
        if parsed.scheme not in {"http", "https"}:
            return "internal_locator"
        if not parsed.hostname or parsed.username or parsed.password:
            return "internal_locator"
        host = _normalized_host(parsed.hostname)
        if host is None or _blocked_host(host):
            return "internal_locator"
        if _has_sensitive_query_or_fragment(parsed.query, parsed.fragment):
            return "internal_locator"
        origin = _origin(value)
        if origin is None or origin in self.internal_origins:
            return "internal_locator"
        return "public_url"

    def public_url(self, value: str) -> str | None:
        if self.classify(value) == "public_url":
            return value
        return None


def _origin(value: str) -> tuple[str, str, int | None] | None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    host = _normalized_host(parsed.hostname)
    if host is None:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if (parsed.scheme == "http" and port == 80) or (
        parsed.scheme == "https" and port == 443
    ):
        port = None
    return parsed.scheme, host, port


def _normalized_host(host: str) -> str | None:
    normalized = host.rstrip(".").lower()
    if not normalized or host.endswith(".."):
        return None
    try:
        return normalized.encode("idna").decode("ascii")
    except UnicodeError:
        return None


def _blocked_host(host: str) -> bool:
    if host == "localhost" or host.endswith(
        (".localhost", ".local", ".internal", ".lan")
    ):
        return True
    if _numeric_host(host):
        return _blocked_numeric_host(host)
    return False


def _numeric_host(host: str) -> bool:
    if host.startswith("0x") or host.isdecimal() or ":" in host:
        return True
    labels = host.split(".")
    return bool(labels) and all(
        label.isdecimal()
        or (len(label) > 1 and label.startswith("0") and label[1:].isdigit())
        or (label.startswith("0x") and len(label) > 2)
        for label in labels
    )


def _blocked_numeric_host(host: str) -> bool:
    if "%" in host:
        return True
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return True
    if isinstance(parsed, ipaddress.IPv4Address) and str(parsed) != host:
        return True
    return (
        parsed.is_loopback
        or parsed.is_private
        or parsed.is_link_local
        or parsed.is_unspecified
        or parsed.is_multicast
        or parsed.is_reserved
    )


def _has_sensitive_query_or_fragment(query: str, fragment: str) -> bool:
    return any(
        key.lower() in _SENSITIVE_QUERY_KEYS
        for key, _ in parse_qsl(query, keep_blank_values=True)
        + parse_qsl(fragment, keep_blank_values=True)
    )


def _path_like(value: str) -> bool:
    lowered = value.lower()
    return (
        (
            not value.startswith(("http://", "https://"))
            and ("/" in value or "\\" in value)
        )
        or value.startswith((".", "~", "/", "\\"))
        or ":\\" in value
        or "/../" in f"/{lowered}/"
        or "\\..\\" in f"\\{lowered}\\"
    )
