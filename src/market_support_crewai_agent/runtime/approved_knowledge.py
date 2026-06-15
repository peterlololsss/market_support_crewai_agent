from __future__ import annotations

import re
from dataclasses import dataclass

from market_support_crewai_agent.runtime.capabilities import (
    read_capabilities_for_artifact,
)
from market_support_crewai_agent.runtime.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.planning import ExecutionPlan
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.schemas import ReplyRequest

_DOC_CAPABILITY = next(iter(read_capabilities_for_artifact("knowledge_answer")), "")
_IMAGE_MARKER_RE = re.compile(r"%%[\w\d_.-]+\.png%%")
_MIN_MATCH_SCORE = 8
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
_STOP_TERMS = frozenset(
    {
        "衍复",
        "衍复投资",
        "yanfu",
        "公司",
        "介绍",
        "基本",
        "信息",
        "情况",
        "你们",
        "我们",
        "请问",
        "一下",
        "可以",
        "什么",
        "是什么",
        "是什",
        "有吗",
        "的吗",
        "在吗",
    }
)


@dataclass(frozen=True)
class ApprovedKnowledgeEntry:
    entry_id: str
    title: str
    questions: tuple[str, ...]
    answer: str
    tags: tuple[str, ...] = ()

    def to_document_context(self) -> str:
        questions = " / ".join(self.questions)
        return f"Q：{questions}\nA：{self.answer}"


class ApprovedKnowledgeEvidenceService:
    """Returns compliance-approved static FAQ snippets as bounded evidence.

    These entries replace the old Dify prompt examples as data. The LLM may
    compose from them, but policy and reply validators still own permissions
    and image-marker safety.
    """

    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        policy: PolicyManifest,
        *,
        max_entries: int = 1,
    ) -> list[EvidenceFact]:
        if plan.compliance.is_compliant is not True:
            return []
        if "document_context" not in plan.capabilities:
            return []
        if _DOC_CAPABILITY not in policy.allowed_read_capabilities:
            return []

        query = str(plan.evidence_query or request.message or "").strip()
        selected = _select_entries(query, canonical_context, max_entries=max_entries)
        facts: list[EvidenceFact] = []
        for score, entry in selected:
            text = entry.to_document_context()
            facts.append(
                EvidenceFact(
                    fact_type="document_context",
                    value=text,
                    source_type="approved_static_knowledge",
                    source_id=entry.entry_id,
                    metadata={
                        "title": entry.title,
                        "content_is_data_only": True,
                        "approved_static_knowledge": True,
                        "contains_image_markers": bool(_IMAGE_MARKER_RE.search(text)),
                        "match_score": score,
                    },
                )
            )
        return facts


class NoopApprovedKnowledgeEvidenceService:
    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        policy: PolicyManifest,
        *,
        max_entries: int = 1,
    ) -> list[EvidenceFact]:
        del request, canonical_context, plan, policy, max_entries
        return []


def _select_entries(
    query: str,
    canonical_context: CanonicalContext,
    *,
    max_entries: int,
) -> list[tuple[int, ApprovedKnowledgeEntry]]:
    scored = [
        (_score_entry(entry, query, canonical_context), entry)
        for entry in _APPROVED_KNOWLEDGE
    ]
    return [
        (score, entry)
        for score, entry in sorted(scored, key=lambda item: item[0], reverse=True)
        if score >= _MIN_MATCH_SCORE
    ][:max_entries]


def _score_entry(
    entry: ApprovedKnowledgeEntry,
    query: str,
    canonical_context: CanonicalContext,
) -> int:
    del canonical_context
    searchable = " ".join(
        [
            entry.entry_id,
            entry.title,
            " ".join(entry.questions),
            " ".join(entry.tags),
        ]
    ).lower()
    return _text_similarity_score(query.lower(), searchable)


def _text_similarity_score(query: str, text: str) -> int:
    query_terms = _semantic_terms(query)
    if not query_terms:
        return 0
    score = 0
    for term in query_terms:
        if term in text:
            score += min(30, max(2, len(term))) * 2
    text_terms = _semantic_terms(text)
    if text_terms:
        score += len(query_terms & text_terms)
    return score


def _semantic_terms(text: str) -> set[str]:
    normalized = str(text or "").lower()
    terms: set[str] = set()
    for token in _TOKEN_RE.findall(normalized):
        if re.fullmatch(r"[a-z0-9]+", token):
            if len(token) >= 3:
                terms.add(token)
            continue
        if len(token) >= 2:
            terms.add(token)
            terms.update(token[index : index + 2] for index in range(0, len(token) - 1))
        if len(token) >= 3:
            terms.update(token[index : index + 3] for index in range(0, len(token) - 2))
    return {term for term in terms if term not in _STOP_TERMS}


_APPROVED_KNOWLEDGE: tuple[ApprovedKnowledgeEntry, ...] = (
    ApprovedKnowledgeEntry(
        entry_id="company_public_account",
        title="公众号二维码",
        questions=(
            "衍复公众号二维码",
            "你们有微信公众号吗？",
            "从微信上面能搜到你们吗？",
            "想关注一下你们的微信公众号",
            "你们这些科普文章可以从哪里找到？",
        ),
        answer=(
            "%%comp_wx_qr_code.png%%\n"
            "欢迎搜索【衍复投资】或扫描二维码关注衍复投资公众号，获取更多资讯。\n"
            "https://doc.weixin.qq.com/doc/w3_AQ8AiwYNACsGU7ZiIQ0QcubFw0411?scode=AJkAJAc9AA0X0s8Vsk"
        ),
        tags=("公众号", "微信", "二维码", "豹豹说", "科普文章"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="company_basic_contact",
        title="联系方式",
        questions=("衍复投资地址在哪里", "衍复投资网址是什么", "衍复投资公众号是什么"),
        answer=(
            "公司名称：上海衍复投资\n"
            "地址：上海市徐汇区瑞平路275号保利西岸C座7楼\n"
            "网址：http://www.yanfuinvestments.com/\n"
            "公众号：%%comp_wx_qr_code.png%%"
        ),
        tags=("地址", "网址", "官网", "公众号", "二维码"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="double_layer_structure",
        title="代销产品双层结构",
        questions=(
            "产品为什么要设置双层结构？",
            "为什么我从银行买的不是基金而是信托计划？",
            "我不能从银行直接买你们的基金产品吗？",
            "我不想要双层结构行不行？",
        ),
        answer="因为按照监管规定银行不可以直接代销私募基金，所以实际上是银行代销信托计划，信托计划可以投向阳光私募基金。",
        tags=("银行", "信托计划", "私募基金", "双层结构", "代销"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="alpha_beta_comparison",
        title="超额收益与指数收益",
        questions=(
            "超额收益是什么？",
            "超额收益与指数收益的区别",
            "什么叫超额收益？",
            "超额收益与指数收益差很多吗？",
            "有哪些因素会影响到超额收益呢？",
        ),
        answer="%%alpha_beta_comparison.png%%",
        tags=("超额收益", "指数收益", "alpha", "beta"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="quant_difference",
        title="量化投资和主观投资区别",
        questions=(
            "量化投资和主观投资有什么区别？",
            "量化与主观最大的不同是什么？",
            "主观多头和你们的策略差异很大吗？",
        ),
        answer="%%quant_difference.png%%",
        tags=("量化投资", "主观投资", "主观多头"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="company_shareholders",
        title="衍复股权结构",
        questions=(
            "你们公司的股权结构是什么？",
            "衍复的股东情况是什么样的？",
            "衍复的股东构成是什么样的？",
            "衍复的股权图是什么样的？",
        ),
        answer="%%company_shareholders.png%%",
        tags=("股权结构", "股东", "持股结构", "股份结构"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="sh000985_weights",
        title="中证全指权重分布",
        questions=(
            "全指中有多少沪深300的股票？",
            "全指中有多少中证500的股票？",
            "全指中有多少中证1000的股票？",
            "全指中小市值的股票占比多不多？",
            "你们中证全指指数增强策略的选股域是什么？",
            "衍复中证全指指数增强策略持有多少支股票？",
        ),
        answer=(
            "%%SH000985_weights.png%%\n"
            "中证全指的权重分布如下图，中证全指指增持仓会与下述权重相似。数据截止至2025.12.31。"
        ),
        tags=("中证全指", "全指", "沪深300", "中证500", "中证1000", "持仓", "选股域"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="sh000985_features",
        title="中证全指指数特征",
        questions=(
            "中证全指指数的是什么风格？",
            "中证全指指数有哪些特征？",
            "中证全指指数与其他指数有什么不同？",
            "中证全指指数的介绍有吗？",
            "全指指数过往表现有吗？",
        ),
        answer=(
            "中证全指指数风格更为均衡，4000+成分股涵盖了A股市场上大、中、小不同市值的股票，"
            "兼具价值、平衡、成长的不同特征，在过去7年大小盘风格轮动过程中展现出相对更加稳定的收益率表现情况。\n"
            "%%SH000985_features.png%%"
        ),
        tags=("中证全指", "全指", "指数特征", "风格", "过往表现"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="sh000985_constituents",
        title="中证全指成分股说明",
        questions=(
            "中证全指指数是什么？",
            "什么是中证全指指数？",
            "中证全指包含哪些股票？",
            "中证全指是不是所有的股票都包括了？",
        ),
        answer=(
            "中证全指指数成分股数量众多，它在A股市场所有股票的基础上剔除了ST、*ST，科创板上市不足一年，"
            "北交所上市不足两年和其他上市不足三个月的股票（其中，针对其他上市不足三个月的股票，"
            "除非该证券自上市以来日均总市值排在前 30 位），其成分股选择方式使得中证全指指数极具市场代表性，"
            "能够更旗帜鲜明地代表A股市场的整体表现。\n"
            "%%SH000985_constituents.png%%"
        ),
        tags=("中证全指", "全指", "成分股", "股票"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="company_historical_aum",
        title="公司历史规模数据",
        questions=(
            "是否有咱们公司成立以来每年整体规模增长的趋势数据呢？",
            "有历史的规模数据吗？",
            "每年规模的趋势数据发一下可以吗？",
            "历史规模数据可以发一下吗？",
            "请问衍复过往几年到现在的规模大致是怎样发展的？",
        ),
        answer=(
            "在咱们ppt当中有哈，供参考。行业规模数据为估算数据，我司不确保该等数据的真实准确完整\n"
            "%%company_historical_aum.png%%"
        ),
        tags=("历史规模", "规模趋势", "规模变化", "aum"),
    ),
    ApprovedKnowledgeEntry(
        entry_id="indice_comparison",
        title="中证1000与其他指数比较",
        questions=(
            "分别分析一下中证500指数和中证1000指数",
            "为什么投资者应该选择中证1000指数增强？",
            "中证1000指增策略与其他策略相比好在哪儿？",
            "比起300的话，中证1000指数增强有什么优势呢？",
            "比起500的话，中证1000指数增强有什么优势呢？",
        ),
        answer=(
            "中证1000指数成分股具有更为分散、风格偏向中小盘、交易活跃、波动性高等特征，并且机构投资者交易占比更低，"
            "因此纠正股票错误定价的力量更小，这可以给量化机构带来更多的交易机会，因此中证1000指数增强策略具备更为丰厚的超额收益水平。\n"
            "%%indice_comparison.png%%\n"
            "目前市场上对标中证500指数的量化策略规模较大，因此中证500指数增强策略的竞争相对更加激烈，"
            "致使超额收益有所衰减，而对标中证1000指数的量化策略规模较小，中证1000指数增强策略还有较大的扩容空间。"
        ),
        tags=("中证1000", "中证500", "沪深300", "指数比较", "指增优势"),
    ),
)
