from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from market_support_crewai_agent.schemas import ReplyRequest

CanonicalEntityType = Literal["strategy"]
CanonicalEntitySource = Literal[
    "request_catalog",
    "request_catalog_default",
    "alias_table",
]
CanonicalStatus = Literal["resolved", "ambiguous", "unknown"]


@dataclass(frozen=True)
class CanonicalEntity:
    type: CanonicalEntityType
    raw_text: str
    canonical_name: str
    source: CanonicalEntitySource
    confidence: float = 1.0

    def to_prompt_dict(self) -> dict:
        return {
            "type": self.type,
            "raw_text": self.raw_text,
            "canonical_name": self.canonical_name,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CanonicalContext:
    strategy_status: CanonicalStatus = "unknown"
    selected_strategy: str | None = None
    strategy_candidates: tuple[str, ...] = ()
    entities: tuple[CanonicalEntity, ...] = ()
    ambiguities: tuple[str, ...] = ()

    def to_prompt_dict(self) -> dict:
        return {
            "strategy_status": self.strategy_status,
            "selected_strategy": self.selected_strategy,
            "strategy_candidates": list(self.strategy_candidates),
            "entities": [entity.to_prompt_dict() for entity in self.entities],
            "ambiguities": list(self.ambiguities),
        }


@dataclass(frozen=True)
class _Match:
    raw_text: str
    canonical_name: str
    source: CanonicalEntitySource


@dataclass(frozen=True)
class _AliasSpec:
    target: str
    patterns: tuple[re.Pattern[str], ...]
    keys: tuple[str, ...] = field(default_factory=tuple)


def _pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


_ALIAS_SPECS: tuple[_AliasSpec, ...] = (
    _AliasSpec(
        "中证A500",
        (
            _pattern(r"中证\s*a\s*500"),
            _pattern(r"(?<![0-9a-z])a\s*500(?![0-9a-z])"),
        ),
        ("a500",),
    ),
    _AliasSpec(
        "中证1000",
        (
            _pattern(r"中证\s*1000"),
            _pattern(r"(?<![0-9a-z])1000(?![0-9a-z])"),
        ),
        ("中证1000", "1000"),
    ),
    _AliasSpec(
        "中证500",
        (
            _pattern(r"中证\s*500"),
            _pattern(r"(?<![0-9a-z])500(?![0-9a-z])"),
        ),
        ("中证500", "500"),
    ),
    _AliasSpec(
        "沪深300",
        (
            _pattern(r"沪深\s*300"),
            _pattern(r"(?<![0-9a-z])300(?![0-9a-z])"),
        ),
        ("沪深300", "300"),
    ),
    _AliasSpec(
        "中证全指",
        (_pattern(r"中证全指"), _pattern(r"全指")),
        ("中证全指", "全指"),
    ),
    _AliasSpec(
        "万得小市值",
        (_pattern(r"万得小市值"), _pattern(r"小市值")),
        ("万得小市值", "小市值"),
    ),
    _AliasSpec("灵活对冲", (_pattern(r"灵活对冲"),), ("灵活对冲",)),
    _AliasSpec("完全对冲", (_pattern(r"完全对冲"),), ("完全对冲",)),
    _AliasSpec("中性", (_pattern(r"中性"),), ("中性",)),
)


def canonicalize_request(request: ReplyRequest) -> CanonicalContext:
    available_strategies = _clean_available_strategies(request.available_strategies)
    matches = _find_strategy_matches(request.message, available_strategies)

    if not matches and len(available_strategies) == 1:
        strategy = available_strategies[0]
        entity = CanonicalEntity(
            type="strategy",
            raw_text="",
            canonical_name=strategy,
            source="request_catalog_default",
            confidence=0.8,
        )
        return CanonicalContext(
            strategy_status="resolved",
            selected_strategy=strategy,
            strategy_candidates=(strategy,),
            entities=(entity,),
        )

    entities = tuple(
        CanonicalEntity(
            type="strategy",
            raw_text=match.raw_text,
            canonical_name=match.canonical_name,
            source=match.source,
        )
        for match in matches
    )
    candidates = _unique(match.canonical_name for match in matches)
    if len(candidates) == 1:
        return CanonicalContext(
            strategy_status="resolved",
            selected_strategy=candidates[0],
            strategy_candidates=tuple(candidates),
            entities=entities,
        )
    if len(candidates) > 1:
        return CanonicalContext(
            strategy_status="ambiguous",
            selected_strategy=None,
            strategy_candidates=tuple(candidates),
            entities=entities,
            ambiguities=("multiple_strategy_candidates",),
        )

    return CanonicalContext(
        strategy_status="unknown",
        selected_strategy=None,
        strategy_candidates=available_strategies,
    )


def _find_strategy_matches(
    message: str,
    available_strategies: tuple[str, ...],
) -> list[_Match]:
    matches: list[_Match] = []
    normalized_message = _normalize(message)

    for strategy in available_strategies:
        if _normalize(strategy) and _normalize(strategy) in normalized_message:
            matches.append(
                _Match(
                    raw_text=strategy,
                    canonical_name=strategy,
                    source="request_catalog",
                )
            )

    for spec in _ALIAS_SPECS:
        raw_text = _first_pattern_match(message, spec.patterns)
        if not raw_text:
            continue
        for strategy in _resolve_alias_target(spec, available_strategies):
            matches.append(
                _Match(
                    raw_text=raw_text,
                    canonical_name=strategy,
                    source="alias_table",
                )
            )

    if not matches and "指增" in normalized_message:
        for strategy in _generic_index_enhancement_candidates(available_strategies):
            matches.append(
                _Match(
                    raw_text="指增",
                    canonical_name=strategy,
                    source="alias_table",
                )
            )

    return _dedupe_matches(matches, message)


def _resolve_alias_target(
    spec: _AliasSpec,
    available_strategies: tuple[str, ...],
) -> tuple[str, ...]:
    if not available_strategies:
        return (spec.target,)

    keys = tuple(_normalize(key) for key in (spec.keys or (spec.target,)))
    candidates = []
    for strategy in available_strategies:
        normalized_strategy = _normalize(strategy)
        if not normalized_strategy:
            continue
        if _normalize(spec.target) == normalized_strategy:
            candidates.append(strategy)
            continue
        if _normalize(spec.target) in normalized_strategy:
            candidates.append(strategy)
            continue
        if _target_key_matches_strategy(keys, normalized_strategy):
            candidates.append(strategy)
    return tuple(_unique(candidates))


def _target_key_matches_strategy(keys: tuple[str, ...], normalized_strategy: str) -> bool:
    for key in keys:
        if key == "500" and "a500" in normalized_strategy:
            continue
        if key and key in normalized_strategy:
            return True
    return False


def _generic_index_enhancement_candidates(
    available_strategies: tuple[str, ...],
) -> tuple[str, ...]:
    candidates = [
        strategy
        for strategy in available_strategies
        if _looks_like_index_enhancement_strategy(strategy)
    ]
    return tuple(_unique(candidates))


def _looks_like_index_enhancement_strategy(strategy: str) -> bool:
    normalized = _normalize(strategy)
    return any(
        token in normalized
        for token in (
            "指增",
            "指数增强",
            "中证",
            "沪深",
            "小市值",
            "a500",
            "300",
            "500",
            "1000",
            "全指",
        )
    )


def _first_pattern_match(
    message: str,
    patterns: tuple[re.Pattern[str], ...],
) -> str:
    for pattern in patterns:
        match = pattern.search(message)
        if match:
            return match.group(0)
    return ""


def _dedupe_matches(matches: list[_Match], message: str) -> list[_Match]:
    seen = set()
    result = []
    for match in matches:
        key = (match.raw_text, match.canonical_name, match.source)
        if key in seen:
            continue
        seen.add(key)
        result.append(match)
    result.sort(key=lambda match: _message_position(message, match.raw_text))
    return result


def _message_position(message: str, raw_text: str) -> int:
    index = message.find(raw_text)
    if index < 0:
        return len(message)
    return index


def _clean_available_strategies(strategies: list[str]) -> tuple[str, ...]:
    return tuple(_unique(strategy.strip() for strategy in strategies if strategy.strip()))


def _unique(values) -> tuple[str, ...]:
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _normalize(value: str) -> str:
    return re.sub(r"[\s_\-—–]+", "", value.strip().lower())
