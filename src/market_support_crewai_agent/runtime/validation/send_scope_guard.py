from __future__ import annotations

import re
from dataclasses import dataclass

from market_support_crewai_agent.runtime.domain.capabilities import ArtifactKind
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.schemas import ReplyRequest


SEND_ARTIFACTS: frozenset[ArtifactKind] = frozenset(
    {"material_pack", "weekly_report", "monthly_report"}
)

_ARTIFACT_KEYWORDS: dict[ArtifactKind, tuple[str, ...]] = {
    "weekly_report": ("周报", "weekly"),
    "monthly_report": ("月报", "monthly"),
    "material_pack": ("材料包", "推介材料", "材料", "material"),
}

_LEADING_TARGET_TOKENS = (
    "请帮忙",
    "请帮我",
    "麻烦你",
    "可不可以",
    "能不能",
    "帮我",
    "帮忙",
    "麻烦",
    "然后",
    "顺便",
    "另外",
    "并且",
    "同时",
    "再",
    "请你",
    "请",
    "把",
    "给",
    "我",
    "我们",
    "发一下",
    "发一份",
    "发下",
    "发个",
    "发送",
    "转发",
    "下发",
    "发",
    "来个",
    "来份",
    "来",
    "整",
    "弄",
    "一个",
    "一份",
    "一下",
    "个",
    "份",
    "下",
)
_TRAILING_TARGET_TOKENS = (
    "发送一下",
    "转发一下",
    "发一下",
    "发送",
    "转发",
    "下发",
    "发",
    "一份",
    "一下",
    "的",
    "个",
    "份",
    "下",
)
_CURRENT_SCOPE_REFERENCES = {
    "这个渠道",
    "当前渠道",
    "本渠道",
    "该渠道",
    "咱们渠道",
    "我们渠道",
    "这个群",
    "当前群",
    "本群",
    "该群",
    "群里",
    "这边",
    "这里",
}
_GENERIC_MODIFIERS = {
    "最新",
    "最近",
    "本期",
    "这期",
    "上期",
    "上一期",
    "完整",
    "完整版",
    "全部",
    "所有",
    "整体",
    "对应",
    "相关",
    "可发",
    "能发",
    "可以发",
    "投资",
    "产品",
    "策略",
    "推介",
    "介绍",
    "客户",
    "渠道",
    "机构",
    "然后",
    "顺便",
    "另外",
    "并且",
    "同时",
    "再",
}
_NON_TARGET_PHRASE_TOKENS = (
    "怎么",
    "如何",
    "什么",
    "多少",
    "为什么",
    "是否",
    "吗",
    "呢",
    "分布",
    "规模",
    "然后",
    "顺便",
    "另外",
    "并且",
    "同时",
    "发我",
    "发给",
    "发送",
    "转发",
)
_ORG_HINT_TOKENS = (
    "证券",
    "银行",
    "资管",
    "基金",
    "保险",
    "信托",
    "期货",
    "财富",
    "投资",
    "资本",
    "集团",
    "分行",
    "支行",
    "营业部",
    "渠道",
)
_ENTITY_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "证券",
    "银行",
    "资管",
    "基金",
    "保险",
    "信托",
    "期货",
    "财富",
    "投资",
    "资本",
    "集团",
    "分行",
    "支行",
    "营业部",
    "渠道",
)
_CLAUSE_SPLIT_RE = re.compile(r"[\n,，。.!！?？;；：:、呢]+")
_NORMALIZE_RE = re.compile(r"[\s_\-—–·•,，。.!！?？:：;；、（）()\[\]【】《》<>\"'“”‘’]+")
_YEAR_MONTH_RE = re.compile(r"^\d{4}年?\d{1,2}月?(\d{1,2}日?)?$")
_COMPACT_DATE_RE = re.compile(r"^\d{4,8}$")


@dataclass(frozen=True)
class SendScopeConflict:
    requested_target: str
    current_scope: str
    artifact_kind: ArtifactKind


def detect_send_scope_conflict(
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    artifact_kind: ArtifactKind | str,
) -> SendScopeConflict | None:
    if artifact_kind not in SEND_ARTIFACTS:
        return None

    for target in explicit_send_targets(request.message, artifact_kind):
        if _is_current_scope_reference(target):
            continue
        if _target_matches_current_scope(target, request.dist_channel_name):
            continue
        if _target_matches_strategy(target, request, canonical_context):
            continue
        if not _looks_like_scope_target(target):
            continue
        return SendScopeConflict(
            requested_target=target,
            current_scope=request.dist_channel_name,
            artifact_kind=artifact_kind,  # type: ignore[arg-type]
        )
    return None


def explicit_send_targets(
    message: str,
    artifact_kind: ArtifactKind | str,
) -> tuple[str, ...]:
    keywords = _ARTIFACT_KEYWORDS.get(artifact_kind, ())
    if not keywords:
        return ()

    text = _user_visible_text(message)
    targets: list[str] = []
    for keyword in keywords:
        for match in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
            before = _last_clause(text[: match.start()])
            before_target = _clean_target_candidate(before)
            if before_target:
                targets.append(before_target)

            after = _first_clause(text[match.end() :])
            after_target = _target_after_artifact(after)
            if after_target:
                targets.append(after_target)
    return _unique(targets)


def conflict_explanation(conflict: SendScopeConflict) -> str:
    label = _artifact_label(conflict.artifact_kind)
    return (
        f"当前群是{conflict.current_scope}相关沟通群，你要的是"
        f"{conflict.requested_target}的{label}，我不能把它替换成当前渠道发送。"
        f"请在对应渠道群操作，或确认是否发送{conflict.current_scope}的{label}。"
    )


def _target_after_artifact(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    for marker in ("发给", "转给", "给", "到", "至"):
        if stripped.startswith(marker):
            return _clean_target_candidate(stripped[len(marker) :])
    return ""


def _clean_target_candidate(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        return ""
    candidate = _CLAUSE_SPLIT_RE.split(candidate)[-1].strip()
    candidate = _strip_target_tokens(candidate, _LEADING_TARGET_TOKENS, from_start=True)
    candidate = _strip_target_tokens(candidate, _TRAILING_TARGET_TOKENS, from_start=False)
    candidate = candidate.strip()
    return candidate if _has_word_char(candidate) else ""


def _strip_target_tokens(
    value: str,
    tokens: tuple[str, ...],
    *,
    from_start: bool,
) -> str:
    candidate = value.strip()
    changed = True
    while changed and candidate:
        changed = False
        for token in tokens:
            if from_start and candidate.startswith(token):
                candidate = candidate[len(token) :].strip()
                changed = True
                break
            if not from_start and candidate.endswith(token):
                candidate = candidate[: -len(token)].strip()
                changed = True
                break
    return candidate


def _looks_like_scope_target(target: str) -> bool:
    normalized = _normalize(target)
    if not normalized:
        return False
    if normalized in {_normalize(value) for value in _GENERIC_MODIFIERS}:
        return False
    if _looks_temporal(normalized):
        return False
    if any(token in target for token in _ORG_HINT_TOKENS):
        return True
    if len(normalized) > 12:
        return False
    if any(token in target for token in _NON_TARGET_PHRASE_TOKENS):
        return False
    return len(normalized) >= 2 and _has_cjk(normalized)


def _target_matches_current_scope(target: str, current_scope: str) -> bool:
    target_norm = _normalize(target)
    current_norm = _normalize(current_scope)
    if not target_norm or not current_norm:
        return False
    if _names_overlap(target_norm, current_norm):
        return True
    target_base = _strip_entity_suffix(target_norm)
    current_base = _strip_entity_suffix(current_norm)
    return bool(target_base and current_base and _names_overlap(target_base, current_base))


def _target_matches_strategy(
    target: str,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
) -> bool:
    target_norm = _normalize(target)
    if not target_norm:
        return False
    strategy_values = list(request.available_strategies)
    for entity in canonical_context.entities:
        strategy_values.extend([entity.raw_text, entity.canonical_name])
    return any(_names_overlap(target_norm, _normalize(strategy)) for strategy in strategy_values)


def _names_overlap(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= 2 and left in right:
        return True
    return len(right) >= 2 and right in left


def _strip_entity_suffix(value: str) -> str:
    for suffix in _ENTITY_SUFFIXES:
        normalized_suffix = _normalize(suffix)
        if value.endswith(normalized_suffix) and len(value) > len(normalized_suffix):
            return value[: -len(normalized_suffix)]
    return value


def _is_current_scope_reference(target: str) -> bool:
    normalized = _normalize(target)
    return normalized in {_normalize(value) for value in _CURRENT_SCOPE_REFERENCES}


def _looks_temporal(normalized: str) -> bool:
    if _YEAR_MONTH_RE.match(normalized) or _COMPACT_DATE_RE.match(normalized):
        return True
    if normalized in {"本周", "这周", "上周", "上上周", "下周", "上月", "本月", "这个月", "这月"}:
        return True
    if normalized.endswith(("月", "月份")) and 1 <= len(normalized) <= 4:
        return True
    return False


def _artifact_label(artifact_kind: ArtifactKind) -> str:
    if artifact_kind == "weekly_report":
        return "周报"
    if artifact_kind == "monthly_report":
        return "月报"
    return "材料"


def _last_clause(text: str) -> str:
    parts = _CLAUSE_SPLIT_RE.split(text)
    return parts[-1] if parts else ""


def _first_clause(text: str) -> str:
    parts = _CLAUSE_SPLIT_RE.split(text)
    return parts[0] if parts else ""


def _user_visible_text(message: str) -> str:
    lines = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("[adapter_") and stripped.endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _normalize(value: str) -> str:
    return _NORMALIZE_RE.sub("", str(value or "").strip().lower()).replace("的", "")


def _has_word_char(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]", value))


def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = _normalize(value)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return tuple(output)
