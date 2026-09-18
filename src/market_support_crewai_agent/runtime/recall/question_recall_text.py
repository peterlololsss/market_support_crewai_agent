from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

from market_support_crewai_agent.runtime.integrations.document_mcp.sanitizer import (
    sanitize_document_text_for_evidence,
)

_ZERO_WIDTH_RE: Final = re.compile("[\u200b\u200c\u200d\ufeff]")
_SPACE_RE: Final = re.compile(r"\s+")
_TOKEN_RE: Final = re.compile(r"[a-z0-9_.+-]+|[\u4e00-\u9fff]+")
_DASH_TRANSLATION: Final = str.maketrans({"－": "-", "–": "-", "—": "-"})
_NORMALIZER_VERSION: Final = "unicode-nfkc-v1"
_LEXICON_VERSION: Final = "approved-knowledge-2026-07-09"
_COMMON_TERMS: Final = frozenset("的了是什么吗呢请问这个") | {
    "一下",
    "一个",
    "可以",
}
_ALIAS_GRAPH: Final = (
    ("website", "网址"),
    ("web site", "网址"),
    ("url", "网址"),
    ("alpha", "超额"),
    ("beta", "指数"),
)
_COMPLIANCE_RISK_PHRASES: Final = (
    "预计收益",
    "预期收益",
    "目标收益",
    "最低收益",
    "基准收益",
    "到期收益",
    "多少收益",
    "保本",
    "无风险",
    "稳赚",
    "保证收益",
    "承诺收益",
    "保证回报",
    "刚兑",
    "本金保障",
    "保证本金",
    "私人微信",
    "加你微信",
    "加微信",
    "通过一下",
    "个人微信",
    "私人联系方式",
    "私人电话",
    "个人电话",
    "个人手机号",
    "私下沟通",
    "其他管理人",
    "同行",
    "竞品",
    "和你们比",
    "比怎么样",
    "四级估值表",
    "业绩归因报告",
    "受限文件",
    "合同",
    "赎回费",
    "申购费减免",
    "管理费减免",
    "费用减免",
    "费率减免",
    "免了吗",
    "免了",
    "看电影",
    "一起吃饭",
    "一起喝酒",
    "周末一起",
    "合格投资人",
    "100万以下",
    "低于100万",
    "起投门槛",
)
_ADAPTER_EXECUTION_CLAIM_PHRASES: Final = (
    "已发送",
    "已经发送",
    "发送给您",
    "已为您发送",
    "已帮您发送",
    "已发",
    "已发给",
    "已经发给",
    "已发您",
    "发您",
    "发好了",
    "发送完成",
    "以上是最新",
    "请查收",
    "has been sent",
    "sent successfully",
    "我已发",
    "我已经发",
    "已同步",
    "已经同步",
    "同步给您",
)


@dataclass(frozen=True, slots=True)
class NormalizationProfile:
    version: str = _NORMALIZER_VERSION
    lexicon_version: str = _LEXICON_VERSION

    def normalize(self, value: str) -> str:
        content = unicodedata.normalize("NFKC", value or "")
        content = _ZERO_WIDTH_RE.sub("", content)
        content = content.translate(_DASH_TRANSLATION)
        content = _SPACE_RE.sub(" ", content).strip().casefold()
        for source, target in _ALIAS_GRAPH:
            content = content.replace(source, target)
        return content


def tokens(value: str, profile: NormalizationProfile) -> list[str]:
    content = profile.normalize(value)
    output: list[str] = []
    for raw in _TOKEN_RE.findall(content):
        if _is_cjk(raw):
            output.extend(char for char in raw if char not in _COMMON_TERMS)
            continue
        if raw not in _COMMON_TERMS:
            output.append(raw)
    return output


def covered_by_ordered_terms(
    surface_terms: tuple[str, ...],
    phrase_terms: tuple[str, ...],
) -> bool:
    if not surface_terms or len(surface_terms) > len(phrase_terms):
        return False
    width = len(surface_terms)
    span_count = len(phrase_terms) - width + 1
    for start in range(span_count):
        if phrase_terms[start : start + width] == surface_terms:
            return True
    return False


def recall_shortcut_blocked(value: str, profile: NormalizationProfile) -> bool:
    content = profile.normalize(value)
    compact = content.replace(" ", "")
    return any(
        phrase in content or phrase in compact for phrase in _COMPLIANCE_RISK_PHRASES
    )


def answer_claims_adapter_execution(value: str, profile: NormalizationProfile) -> bool:
    content = profile.normalize(value)
    return any(phrase in content for phrase in _ADAPTER_EXECUTION_CLAIM_PHRASES)


def safe_metadata_label(value: str) -> str:
    sanitized = sanitize_document_text_for_evidence(value)
    if bool(sanitized.metadata.get("internal_locator_redacted")) or bool(
        sanitized.metadata.get("secret_redacted")
    ):
        return "document"
    return sanitized.text[:160].strip()


def _is_cjk(value: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in value)
