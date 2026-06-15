from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from market_support_crewai_agent.runtime.domain.capabilities import (
    read_capabilities_for_artifact,
)
from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.runtime.domain.policy import PolicyManifest
from market_support_crewai_agent.schemas import ReplyRequest, StrictModel
from market_support_crewai_agent.settings import Settings, get_settings

_DOC_CAPABILITY = next(iter(read_capabilities_for_artifact("knowledge_answer")), "")


@dataclass(frozen=True)
class ApprovedImageAsset:
    asset_id: str
    marker_filename: str
    title: str
    semantic_purpose: str
    usage_notes: str = ""

    @property
    def marker(self) -> str:
        return f"%%{self.marker_filename}%%"


@dataclass(frozen=True)
class ApprovedKnowledgeEntry:
    entry_id: str
    title: str
    approved_answer: str
    semantic_purpose: str
    user_request_examples: tuple[str, ...] = ()
    image_asset_ids: tuple[str, ...] = ()

    def to_document_context(self, selected_image_asset_ids: tuple[str, ...]) -> str:
        answer = self.approved_answer
        selected_assets = set(selected_image_asset_ids)
        for asset_id in self.image_asset_ids:
            asset = _APPROVED_IMAGE_ASSETS_BY_ID.get(asset_id)
            if asset is None or asset_id in selected_assets:
                continue
            answer = answer.replace(asset.marker, "").strip()
        if not answer.strip():
            return ""
        return f"Q：{self.title}\nA：{answer.strip()}"


class ApprovedImageAssetCandidate(StrictModel):
    asset_id: str
    title: str
    semantic_purpose: str
    usage_notes: str = ""


class ApprovedKnowledgeCandidate(StrictModel):
    entry_id: str
    title: str
    semantic_purpose: str
    user_request_examples: tuple[str, ...] = ()
    image_assets: tuple[ApprovedImageAssetCandidate, ...] = ()


class ApprovedKnowledgeSelection(StrictModel):
    selected_entry_ids: tuple[str, ...] = ()
    selected_image_asset_ids: tuple[str, ...] = ()
    confidence: Literal["none", "low", "medium", "high"] = "none"
    rationale: str = Field(default="", max_length=600)


class ApprovedKnowledgeSelector(Protocol):
    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        canonical_context: CanonicalContext,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection: ...


class NoopApprovedKnowledgeSelector:
    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        canonical_context: CanonicalContext,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection:
        del (
            user_message,
            evidence_query,
            canonical_context,
            catalog_manifest,
            max_entries,
            max_images,
        )
        return ApprovedKnowledgeSelection(confidence="none")


class CrewAIApprovedKnowledgeSelector:
    """Semantic selector for the small approved static-knowledge catalog."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def select(
        self,
        *,
        user_message: str,
        evidence_query: str,
        canonical_context: CanonicalContext,
        catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
        max_entries: int,
        max_images: int,
    ) -> ApprovedKnowledgeSelection:
        if not self.settings.llm_api_key:
            return ApprovedKnowledgeSelection(confidence="none")
        prompt = _selector_prompt(
            user_message=user_message,
            evidence_query=evidence_query,
            canonical_context=canonical_context,
            catalog_manifest=catalog_manifest,
            max_entries=max_entries,
            max_images=max_images,
        )
        return await _run_crewai_selector(
            prompt,
            self.settings,
            timeout_seconds=self.settings.llm_timeout_seconds,
        )


class ApprovedKnowledgeEvidenceService:
    """Returns compliance-approved static snippets as bounded evidence."""

    def __init__(
        self,
        selector: ApprovedKnowledgeSelector | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.selector = selector or CrewAIApprovedKnowledgeSelector(settings)

    async def collect(
        self,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        policy: PolicyManifest,
        *,
        max_entries: int = 1,
        max_images: int = 2,
    ) -> list[EvidenceFact]:
        if plan.compliance.is_compliant is not True:
            return []
        if "document_context" not in plan.capabilities:
            return []
        if _DOC_CAPABILITY not in policy.allowed_read_capabilities:
            return []

        manifest = approved_knowledge_manifest()
        query = str(plan.evidence_query or "").strip()
        try:
            selection = await self.selector.select(
                user_message=str(request.message or "").strip(),
                evidence_query=query,
                canonical_context=canonical_context,
                catalog_manifest=manifest,
                max_entries=max_entries,
                max_images=max_images,
            )
        except Exception:
            return []

        validated = _validate_selection(
            selection,
            max_entries=max_entries,
            max_images=max_images,
        )
        if validated.confidence == "none" or not validated.selected_entry_ids:
            return []

        facts: list[EvidenceFact] = []
        for entry_id in validated.selected_entry_ids:
            entry = _APPROVED_KNOWLEDGE_BY_ID.get(entry_id)
            if entry is None:
                continue
            selected_image_asset_ids = tuple(
                asset_id
                for asset_id in validated.selected_image_asset_ids
                if asset_id in entry.image_asset_ids
            )
            text = entry.to_document_context(selected_image_asset_ids)
            if not text.strip():
                continue
            selected_assets = tuple(
                _APPROVED_IMAGE_ASSETS_BY_ID[asset_id]
                for asset_id in selected_image_asset_ids
            )
            facts.append(
                EvidenceFact(
                    fact_type="document_context",
                    value=text,
                    source_type="approved_static_knowledge",
                    source_id=entry.entry_id,
                    metadata={
                        "title": entry.title,
                        "approved_static_knowledge": True,
                        "approved_entry_id": entry.entry_id,
                        "selected_by": "approved_knowledge_semantic_selector",
                        "selector_confidence": validated.confidence,
                        "selector_rationale": validated.rationale,
                        "image_asset_ids": selected_image_asset_ids,
                        "image_markers": tuple(asset.marker for asset in selected_assets),
                        "image_marker_filenames": tuple(
                            asset.marker_filename for asset in selected_assets
                        ),
                        "image_asset_semantic_purposes": {
                            asset.asset_id: asset.semantic_purpose
                            for asset in selected_assets
                        },
                        "content_is_data_only": True,
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
        max_images: int = 2,
    ) -> list[EvidenceFact]:
        del request, canonical_context, plan, policy, max_entries, max_images
        return []


def approved_knowledge_manifest() -> tuple[ApprovedKnowledgeCandidate, ...]:
    candidates: list[ApprovedKnowledgeCandidate] = []
    for entry in APPROVED_KNOWLEDGE:
        candidates.append(
            ApprovedKnowledgeCandidate(
                entry_id=entry.entry_id,
                title=entry.title,
                semantic_purpose=entry.semantic_purpose,
                user_request_examples=entry.user_request_examples,
                image_assets=tuple(
                    ApprovedImageAssetCandidate(
                        asset_id=asset.asset_id,
                        title=asset.title,
                        semantic_purpose=asset.semantic_purpose,
                        usage_notes=asset.usage_notes,
                    )
                    for asset in _entry_image_assets(entry)
                ),
            )
        )
    return tuple(candidates)


def approved_image_markers() -> frozenset[str]:
    return frozenset(asset.marker_filename for asset in APPROVED_IMAGE_ASSETS)


def approved_image_asset_by_marker(filename: str) -> ApprovedImageAsset | None:
    return _APPROVED_IMAGE_ASSETS_BY_MARKER_FILENAME.get(filename)


def _validate_selection(
    selection: ApprovedKnowledgeSelection,
    *,
    max_entries: int,
    max_images: int,
) -> ApprovedKnowledgeSelection:
    if selection.confidence == "none":
        return ApprovedKnowledgeSelection(
            confidence="none",
            rationale=selection.rationale,
        )

    selected_entry_ids = _valid_unique_entry_ids(selection.selected_entry_ids)
    selected_entry_ids = selected_entry_ids[:max_entries]
    if not selected_entry_ids:
        return ApprovedKnowledgeSelection(
            confidence="none",
            rationale=selection.rationale,
        )

    selected_entry_asset_ids = {
        asset_id
        for entry_id in selected_entry_ids
        for asset_id in _APPROVED_KNOWLEDGE_BY_ID[entry_id].image_asset_ids
    }
    selected_image_asset_ids = _valid_unique_asset_ids(
        selection.selected_image_asset_ids,
        allowed_asset_ids=selected_entry_asset_ids,
    )
    selected_image_asset_ids = selected_image_asset_ids[:max_images]
    return ApprovedKnowledgeSelection(
        selected_entry_ids=tuple(selected_entry_ids),
        selected_image_asset_ids=tuple(selected_image_asset_ids),
        confidence=selection.confidence,
        rationale=selection.rationale,
    )


def _valid_unique_entry_ids(entry_ids: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for entry_id in entry_ids:
        if entry_id not in _APPROVED_KNOWLEDGE_BY_ID or entry_id in seen:
            continue
        seen.add(entry_id)
        output.append(entry_id)
    return output


def _valid_unique_asset_ids(
    asset_ids: tuple[str, ...],
    *,
    allowed_asset_ids: set[str],
) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for asset_id in asset_ids:
        if asset_id not in allowed_asset_ids or asset_id in seen:
            continue
        if asset_id not in _APPROVED_IMAGE_ASSETS_BY_ID:
            continue
        seen.add(asset_id)
        output.append(asset_id)
    return output


def _entry_image_assets(entry: ApprovedKnowledgeEntry) -> tuple[ApprovedImageAsset, ...]:
    return tuple(
        _APPROVED_IMAGE_ASSETS_BY_ID[asset_id]
        for asset_id in entry.image_asset_ids
        if asset_id in _APPROVED_IMAGE_ASSETS_BY_ID
    )


def _selector_prompt(
    *,
    user_message: str,
    evidence_query: str,
    canonical_context: CanonicalContext,
    catalog_manifest: tuple[ApprovedKnowledgeCandidate, ...],
    max_entries: int,
    max_images: int,
) -> str:
    payload = {
        "user_message": user_message,
        "evidence_query": evidence_query,
        "canonical_context": canonical_context.to_prompt_dict(),
        "max_entries": max_entries,
        "max_images": max_images,
        "catalog_manifest": [
            candidate.model_dump(mode="json", exclude_none=True)
            for candidate in catalog_manifest
        ],
    }
    return (
        "You are the approved static knowledge semantic selector for a deterministic "
        "support reply harness.\n\n"
        "Select approved knowledge entries only from the provided candidate IDs.\n"
        "Select image assets only from the provided asset IDs that belong to selected entries.\n"
        "Do not create new IDs, filenames, image markers, facts, actions, or final reply text.\n"
        "Return confidence='none' and no IDs if the catalog does not directly answer "
        "the current user request.\n"
        "Use the natural-language fields as semantic context only. Do not perform "
        "or describe keyword, substring, regex, fuzzy, or n-gram matching.\n"
        "Return only ApprovedKnowledgeSelection matching the response schema.\n\n"
        "Selector input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}"
    )


async def _run_crewai_selector(
    prompt: str,
    settings: Settings,
    *,
    timeout_seconds: float,
) -> ApprovedKnowledgeSelection:
    from crewai import Agent, LLM

    agent = Agent(
        role="Approved Static Knowledge Semantic Selector",
        goal="Select only closed-set approved knowledge IDs for the harness.",
        backstory=(
            "You choose from a small vetted catalog. You do not compose replies, "
            "invent IDs, or call tools."
        ),
        llm=LLM(
            model=settings.llm_model,
            provider=settings.llm_provider,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0,
            max_tokens=min(settings.llm_max_tokens, 1200),
            timeout=settings.llm_timeout_seconds,
        ),
        allow_delegation=False,
        verbose=settings.crewai_verbose,
        max_iter=1,
        max_execution_time=settings.crewai_max_execution_time,
        max_retry_limit=settings.crewai_max_retry_limit,
        planning=False,
    )
    result = await asyncio.wait_for(
        agent.kickoff_async(prompt, response_format=ApprovedKnowledgeSelection),
        timeout=timeout_seconds,
    )
    if result.pydantic is not None:
        return ApprovedKnowledgeSelection.model_validate(result.pydantic)
    return ApprovedKnowledgeSelection.model_validate_json(result.raw)


APPROVED_IMAGE_ASSETS: tuple[ApprovedImageAsset, ...] = (
    ApprovedImageAsset(
        asset_id="company_public_account_qr",
        marker_filename="comp_wx_qr_code.png",
        title="衍复投资公众号二维码",
        semantic_purpose="Use only when the user asks for the company's WeChat public account, QR code, or where to follow official public-account content.",
    ),
    ApprovedImageAsset(
        asset_id="alpha_beta_comparison_chart",
        marker_filename="alpha_beta_comparison.png",
        title="超额收益与指数收益示意图",
        semantic_purpose="Use only when the user asks to understand alpha/超额收益 versus beta/指数收益.",
    ),
    ApprovedImageAsset(
        asset_id="quant_difference_chart",
        marker_filename="quant_difference.png",
        title="量化投资与主观投资区别图",
        semantic_purpose="Use only when the user asks about differences between quantitative and discretionary investment.",
    ),
    ApprovedImageAsset(
        asset_id="company_shareholders_chart",
        marker_filename="company_shareholders.png",
        title="衍复股权结构图",
        semantic_purpose="Use only when the user asks about Yanfu ownership, shareholders, or equity structure.",
    ),
    ApprovedImageAsset(
        asset_id="sh000985_weights_chart",
        marker_filename="SH000985_weights.png",
        title="中证全指权重分布图",
        semantic_purpose="Use only when the user asks about CSI All Share constituent or market-cap weight distribution.",
    ),
    ApprovedImageAsset(
        asset_id="sh000985_features_chart",
        marker_filename="SH000985_features.png",
        title="中证全指指数特征图",
        semantic_purpose="Use only when the user asks about CSI All Share style, characteristics, or historical behavior.",
    ),
    ApprovedImageAsset(
        asset_id="sh000985_constituents_chart",
        marker_filename="SH000985_constituents.png",
        title="中证全指成分股说明图",
        semantic_purpose="Use only when the user asks what CSI All Share includes or how constituents are selected.",
    ),
    ApprovedImageAsset(
        asset_id="company_historical_aum_chart",
        marker_filename="company_historical_aum.png",
        title="公司历史规模趋势图",
        semantic_purpose="Use only when the user asks for historical company AUM or scale trend data.",
    ),
    ApprovedImageAsset(
        asset_id="indice_comparison_chart",
        marker_filename="indice_comparison.png",
        title="中证1000与其他指数比较图",
        semantic_purpose="Use only when the user asks why choose CSI 1000 index enhancement or compares CSI 1000 with CSI 500/沪深300.",
    ),
)

_APPROVED_IMAGE_ASSETS_BY_ID = {asset.asset_id: asset for asset in APPROVED_IMAGE_ASSETS}
_APPROVED_IMAGE_ASSETS_BY_MARKER_FILENAME = {
    asset.marker_filename: asset for asset in APPROVED_IMAGE_ASSETS
}


APPROVED_KNOWLEDGE: tuple[ApprovedKnowledgeEntry, ...] = (
    ApprovedKnowledgeEntry(
        entry_id="company_public_account",
        title="公众号二维码",
        semantic_purpose="Answer requests for Yanfu's official WeChat public account, QR code, or where to follow public education articles.",
        user_request_examples=(
            "衍复公众号二维码",
            "你们有微信公众号吗？",
            "从微信上面能搜到你们吗？",
            "想关注一下你们的微信公众号",
            "你们这些科普文章可以从哪里找到？",
        ),
        approved_answer=(
            "%%comp_wx_qr_code.png%%\n"
            "欢迎搜索【衍复投资】或扫描二维码关注衍复投资公众号，获取更多资讯。\n"
            "https://doc.weixin.qq.com/doc/w3_AQ8AiwYNACsGU7ZiIQ0QcubFw0411?scode=AJkAJAc9AA0X0s8Vsk"
        ),
        image_asset_ids=("company_public_account_qr",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="company_basic_contact",
        title="联系方式",
        semantic_purpose="Answer requests for Yanfu's company name, office address, website, or basic contact information.",
        user_request_examples=(
            "衍复投资地址在哪里",
            "衍复投资网址是什么",
            "衍复投资公众号是什么",
        ),
        approved_answer=(
            "公司名称：上海衍复投资\n"
            "地址：上海市徐汇区瑞平路275号保利西岸C座7楼\n"
            "网址：http://www.yanfuinvestments.com/\n"
            "公众号：%%comp_wx_qr_code.png%%"
        ),
        image_asset_ids=("company_public_account_qr",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="double_layer_structure",
        title="代销产品双层结构",
        semantic_purpose="Explain why bank-distributed products use a trust-plan/private-fund double-layer structure.",
        user_request_examples=(
            "产品为什么要设置双层结构？",
            "为什么我从银行买的不是基金而是信托计划？",
            "我不能从银行直接买你们的基金产品吗？",
            "我不想要双层结构行不行？",
        ),
        approved_answer="因为按照监管规定银行不可以直接代销私募基金，所以实际上是银行代销信托计划，信托计划可以投向阳光私募基金。",
    ),
    ApprovedKnowledgeEntry(
        entry_id="alpha_beta_comparison",
        title="超额收益与指数收益",
        semantic_purpose="Answer conceptual requests about alpha/超额收益 versus beta/指数收益.",
        user_request_examples=(
            "超额收益是什么？",
            "超额收益与指数收益的区别",
            "什么叫超额收益？",
            "超额收益与指数收益差很多吗？",
            "有哪些因素会影响到超额收益呢？",
        ),
        approved_answer="%%alpha_beta_comparison.png%%",
        image_asset_ids=("alpha_beta_comparison_chart",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="quant_difference",
        title="量化投资和主观投资区别",
        semantic_purpose="Answer requests comparing quantitative investment and discretionary/subjective investment.",
        user_request_examples=(
            "量化投资和主观投资有什么区别？",
            "量化与主观最大的不同是什么？",
            "主观多头和你们的策略差异很大吗？",
        ),
        approved_answer="%%quant_difference.png%%",
        image_asset_ids=("quant_difference_chart",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="company_shareholders",
        title="衍复股权结构",
        semantic_purpose="Answer requests for Yanfu's shareholder composition or equity structure chart.",
        user_request_examples=(
            "你们公司的股权结构是什么？",
            "衍复的股东情况是什么样的？",
            "衍复的股东构成是什么样的？",
            "衍复的股权图是什么样的？",
        ),
        approved_answer="%%company_shareholders.png%%",
        image_asset_ids=("company_shareholders_chart",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="sh000985_weights",
        title="中证全指权重分布",
        semantic_purpose="Answer requests about CSI All Share weight distribution across major size/index buckets or stock universe.",
        user_request_examples=(
            "全指中有多少沪深300的股票？",
            "全指中有多少中证500的股票？",
            "全指中有多少中证1000的股票？",
            "全指中小市值的股票占比多不多？",
            "你们中证全指指数增强策略的选股域是什么？",
            "衍复中证全指指数增强策略持有多少支股票？",
        ),
        approved_answer=(
            "%%SH000985_weights.png%%\n"
            "中证全指的权重分布如下图，中证全指指增持仓会与下述权重相似。数据截止至2025.12.31。"
        ),
        image_asset_ids=("sh000985_weights_chart",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="sh000985_features",
        title="中证全指指数特征",
        semantic_purpose="Answer requests about CSI All Share style, characteristics, differences, or past performance summary.",
        user_request_examples=(
            "中证全指指数的是什么风格？",
            "中证全指指数有哪些特征？",
            "中证全指指数与其他指数有什么不同？",
            "中证全指指数的介绍有吗？",
            "全指指数过往表现有吗？",
        ),
        approved_answer=(
            "中证全指指数风格更为均衡，4000+成分股涵盖了A股市场上大、中、小不同市值的股票，"
            "兼具价值、平衡、成长的不同特征，在过去7年大小盘风格轮动过程中展现出相对更加稳定的收益率表现情况。\n"
            "%%SH000985_features.png%%"
        ),
        image_asset_ids=("sh000985_features_chart",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="sh000985_constituents",
        title="中证全指成分股说明",
        semantic_purpose="Answer requests for what CSI All Share is, what it includes, or how constituents are selected.",
        user_request_examples=(
            "中证全指指数是什么？",
            "什么是中证全指指数？",
            "中证全指包含哪些股票？",
            "中证全指是不是所有的股票都包括了？",
        ),
        approved_answer=(
            "中证全指指数成分股数量众多，它在A股市场所有股票的基础上剔除了ST、*ST，科创板上市不足一年，"
            "北交所上市不足两年和其他上市不足三个月的股票（其中，针对其他上市不足三个月的股票，"
            "除非该证券自上市以来日均总市值排在前 30 位），其成分股选择方式使得中证全指指数极具市场代表性，"
            "能够更旗帜鲜明地代表A股市场的整体表现。\n"
            "%%SH000985_constituents.png%%"
        ),
        image_asset_ids=("sh000985_constituents_chart",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="company_historical_aum",
        title="公司历史规模数据",
        semantic_purpose="Answer requests for historical Yanfu company scale, AUM, or annual scale trend data.",
        user_request_examples=(
            "是否有咱们公司成立以来每年整体规模增长的趋势数据呢？",
            "有历史的规模数据吗？",
            "每年规模的趋势数据发一下可以吗？",
            "历史规模数据可以发一下吗？",
            "请问衍复过往几年到现在的规模大致是怎样发展的？",
        ),
        approved_answer=(
            "在咱们ppt当中有哈，供参考。行业规模数据为估算数据，我司不确保该等数据的真实准确完整\n"
            "%%company_historical_aum.png%%"
        ),
        image_asset_ids=("company_historical_aum_chart",),
    ),
    ApprovedKnowledgeEntry(
        entry_id="indice_comparison",
        title="中证1000与其他指数比较",
        semantic_purpose="Answer requests comparing CSI 1000 index enhancement with CSI 500, HS300, or other index enhancement strategies.",
        user_request_examples=(
            "分别分析一下中证500指数和中证1000指数",
            "为什么投资者应该选择中证1000指数增强？",
            "中证1000指增策略与其他策略相比好在哪儿？",
            "比起300的话，中证1000指数增强有什么优势呢？",
            "比起500的话，中证1000指数增强有什么优势呢？",
        ),
        approved_answer=(
            "中证1000指数成分股具有更为分散、风格偏向中小盘、交易活跃、波动性高等特征，并且机构投资者交易占比更低，"
            "因此纠正股票错误定价的力量更小，这可以给量化机构带来更多的交易机会，因此中证1000指数增强策略具备更为丰厚的超额收益水平。\n"
            "%%indice_comparison.png%%\n"
            "目前市场上对标中证500指数的量化策略规模较大，因此中证500指数增强策略的竞争相对更加激烈，"
            "致使超额收益有所衰减，而对标中证1000指数的量化策略规模较小，中证1000指数增强策略还有较大的扩容空间。"
        ),
        image_asset_ids=("indice_comparison_chart",),
    ),
)

_APPROVED_KNOWLEDGE_BY_ID = {entry.entry_id: entry for entry in APPROVED_KNOWLEDGE}
