from __future__ import annotations

import json
import re

from market_support_crewai_agent.runtime.capabilities import (
    capability_by_name,
)
from market_support_crewai_agent.runtime.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.conversation_store import ConversationMessage
from market_support_crewai_agent.runtime.policy import PolicyManifest
from market_support_crewai_agent.runtime.prompt_assembler import (
    PromptProgram,
    assemble_prompt_program,
)
from market_support_crewai_agent.runtime.prompt_context import (
    IntentGateResult,
    PromptAssemblyContext,
)
from market_support_crewai_agent.runtime.prompt_profiles import (
    ModelFamily,
    prompt_profile_by_stage,
)
from market_support_crewai_agent.schemas import ReplyRequest
from market_support_crewai_agent.settings import Settings


_SEND_KEYWORDS = ("发", "发送", "转发", "给我", "麻烦给", "辛苦给", "send")
_MATERIAL_KEYWORDS = (
    "材料",
    "材料包",
    "推介材料",
    "推荐资料",
    "宣传材料",
    "介绍材料",
    "对客材料",
    "产品资料",
    "产品介绍",
    "一页通",
    "一夜通",
    "一夜痛",
    "ppt",
    "要素表",
    "路演材料",
    "培训视频",
    "开放日历",
    "排期",
    "销售日期表",
    "material",
)
_WEEKLY_KEYWORDS = ("周报", "weekly")
_MONTHLY_KEYWORDS = ("月报", "monthly")
_ARTIFACT_KEYWORDS = {
    "weekly_report": _WEEKLY_KEYWORDS,
    "monthly_report": _MONTHLY_KEYWORDS,
    "material_pack": _MATERIAL_KEYWORDS,
}
_HUMAN_SUPPORT_KEYWORDS = (
    "销售",
    "客户经理",
    "人工",
    "真人",
    "投诉",
    "帮我问",
    "问一下销售",
    "支持同事",
    "联系我",
    "转人工",
    "顾总",
    "高总",
    "姚总",
)
_SERVICE_PRIORITY_KEYWORDS = (
    "投诉",
    "人工",
    "真人",
    "转人工",
    "客户经理",
    "顾总",
    "高总",
    "姚总",
    "联系我",
)
_PRIVATE_CONTACT_KEYWORDS = (
    "加你微信",
    "微信通过",
    "手机号",
    "打个电话",
    "电话多少",
    "私人微信",
)
_BLOCKED_KEYWORDS = (
    "保本",
    "保证收益",
    "稳赚",
    "预期收益",
    "目标收益",
    "基准收益",
    "最低收益",
    "承诺收益",
    "无风险",
    "没有风险",
    "安全",
    "合同",
    "交易文件",
    "认购文件",
    "其他管理人",
    "同行",
    "竞品",
    "同类型产品比较",
    "比较下其他",
    *_PRIVATE_CONTACT_KEYWORDS,
    "自营盘",
    "四级估值",
    "估值表",
    "归因报告",
    "绩效归因",
    "免赎回费",
    "赎回费可以免",
    "赎回费就别收",
    "达不到直销门槛",
)
_SMALLTALK_HINTS = (
    "hi",
    "hello",
    "hey",
    "help",
    "thanks",
    "你好",
    "在吗",
    "在么",
    "谢谢",
    "你能做什么",
    "你可以做什么",
    "能做什么",
    "能帮我做什么",
    "你是谁",
    "你叫什么",
    "自我介绍",
    "介绍一下你自己",
    "男的女的",
    "你是男是女",
    "性别",
)
_MATERIAL_DETAIL_KEYWORDS = (
    "开放日历",
    "排期",
    "销售日期表",
    "申购时间表",
    "开放日",
    "申购日",
    "赎回日",
    "什么时候开放",
    "最晚几号能下单",
    "最早什么时候可以买",
    "最早什么时候可以卖",
    "哪个代码能买",
    "代码什么时候开放",
    "有哪些产品可以买",
    "哪些产品可以买",
    "有什么产品可以买",
    "还有产品可以买",
    "还有产品在售",
    "产品持营",
    "下周持营",
    "下周有产品",
    "封闭期",
    "锁定期",
    "硬锁",
    "止损线",
    "预警线",
    "平仓线",
    "预警止损线",
    "赎回费",
    "认购费",
    "申购费",
    "管理费",
    "固定管理费",
    "浮动管理费",
    "业绩报酬",
    "计提",
    "预约申购",
    "可追加",
    "可以追加",
    "历史分红数据",
    "在售",
)
_PERFORMANCE_METRIC_KEYWORDS = (
    "净值",
    "收益",
    "超额",
    "年化",
    "胜率",
    "最大回撤",
    "回撤修复",
    "夏普",
    "表现",
    "涨跌幅",
    "赚钱",
    "亏了多少",
    "怎么样",
)
_WEEKLY_TIME_KEYWORDS = (
    "这周",
    "本周",
    "上周",
    "周度",
    "近期",
    "最近",
    "今年以来",
    "最新",
)
_MONTHLY_TIME_KEYWORDS = (
    "这个月",
    "本月",
    "上个月",
    "上月",
)
_PERFORMANCE_EXPLANATION_KEYWORDS = (
    "为什么",
    "为何",
    "原因",
    "区别",
    "计算方式",
    "怎么算",
    "公式",
    "占比",
    "因子贡献",
    "因子来源",
    "量价因子",
    "高频因子",
    "没有显示",
)

def model_family_from_settings(settings: Settings) -> ModelFamily:
    model = settings.llm_model.lower()
    if "deepseek-v4-pro" in model or "ds-v4pro" in model or "v4-pro" in model:
        return "ds_v4pro"
    if "deepseek" in model:
        return "deepseek"
    if "gpt" in model:
        return "gpt"
    if "claude" in model:
        return "claude"
    return "generic"


def route_intent(
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    policy: PolicyManifest,
    history: list[ConversationMessage] | None = None,
) -> IntentGateResult:
    del policy
    text = _user_visible_text(request.message).lower()
    matched: list[str] = []
    human_matches = _matches(text, _HUMAN_SUPPORT_KEYWORDS)

    if (
        human_matches
        and _matches(text, _SERVICE_PRIORITY_KEYWORDS)
        and not _matches(text, _PRIVATE_CONTACT_KEYWORDS)
    ):
        return IntentGateResult(
            artifact_hint="human_support",
            side_effect_hint=False,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(human_matches),
            confidence=0.85,
        )

    blocked_matches = _matches(text, _BLOCKED_KEYWORDS)
    if blocked_matches:
        return IntentGateResult(
            artifact_hint="refusal",
            side_effect_hint=False,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="blocked",
            matched_keywords=_limit(blocked_matches),
            confidence=0.95,
        )

    send = _has_any(text, _SEND_KEYWORDS) or _looks_like_send_shorthand(text)
    side_effect_artifacts: list[str] = []
    if send and _has_any(text, _WEEKLY_KEYWORDS):
        side_effect_artifacts.append("weekly_report")
        matched.extend(_matches(text, _WEEKLY_KEYWORDS))
    if send and _has_any(text, _MONTHLY_KEYWORDS):
        side_effect_artifacts.append("monthly_report")
        matched.extend(_matches(text, _MONTHLY_KEYWORDS))
    if send and _has_any(text, _MATERIAL_KEYWORDS):
        side_effect_artifacts.append("material_pack")
        matched.extend(_matches(text, _MATERIAL_KEYWORDS))

    if len(side_effect_artifacts) > 1:
        return IntentGateResult(
            artifact_hint="unclear",
            side_effect_hint=True,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(matched),
            confidence=0.85,
        )
    if side_effect_artifacts:
        return IntentGateResult(
            artifact_hint=side_effect_artifacts[0],  # type: ignore[arg-type]
            side_effect_hint=True,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(matched),
            confidence=0.9,
        )

    material_detail_matches = _material_detail_matches(text)
    if material_detail_matches:
        return IntentGateResult(
            artifact_hint="material_pack",
            side_effect_hint=True,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(material_detail_matches),
            confidence=0.82,
        )

    performance_artifact, performance_matches = _performance_artifact(text)
    if performance_artifact:
        return IntentGateResult(
            artifact_hint=performance_artifact,  # type: ignore[arg-type]
            side_effect_hint=True,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(performance_matches),
            confidence=0.8,
        )

    followup_artifact = _followup_side_effect_artifact(text, canonical_context, history)
    if followup_artifact:
        return IntentGateResult(
            artifact_hint=followup_artifact,  # type: ignore[arg-type]
            side_effect_hint=True,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=["history_followup"],
            confidence=0.78,
        )

    if human_matches:
        return IntentGateResult(
            artifact_hint="human_support",
            side_effect_hint=False,
            named_strategy_count=_named_strategy_count(request, canonical_context),
            compliance_hint="clean",
            matched_keywords=_limit(human_matches),
            confidence=0.8,
        )

    return IntentGateResult(
        artifact_hint="unclear",
        side_effect_hint=False,
        named_strategy_count=_named_strategy_count(request, canonical_context),
        compliance_hint="unknown",
        matched_keywords=[],
        confidence=0.2,
    )


def select_prompt_program(ctx: PromptAssemblyContext) -> PromptProgram:
    profile = prompt_profile_by_stage(ctx.stage, ctx.model_family)
    if ctx.stage == "planner_intent":
        fragment_ids = _planner_fragments(ctx)
    elif ctx.stage == "knowledge_composer":
        fragment_ids = _knowledge_composer_fragments(ctx)
    elif ctx.stage == "smalltalk_composer":
        fragment_ids = _smalltalk_composer_fragments(ctx)
    else:
        raise ValueError(f"unsupported prompt stage: {ctx.stage}")
    return assemble_prompt_program(ctx, profile, tuple(fragment_ids))


def _planner_fragments(ctx: PromptAssemblyContext) -> list[str]:
    gate = ctx.intent_gate or route_intent(
        ctx.request,
        ctx.canonical_context,
        ctx.policy,
    )
    fragment_ids = [
        "base.planner_intent",
        _model_fragment(ctx.model_family),
        "output.intent_frame_schema",
        "compliance.reason_codes",
    ]
    artifact = gate.artifact_hint
    if artifact == "material_pack":
        fragment_ids.extend(_capability_fragments("material_pack"))
    elif artifact == "weekly_report":
        fragment_ids.extend(_capability_fragments("weekly_report"))
    elif artifact == "monthly_report":
        fragment_ids.extend(_capability_fragments("monthly_report"))
    elif artifact == "knowledge_answer":
        fragment_ids.extend(_capability_fragments("document_context"))
    elif artifact == "human_support":
        fragment_ids.extend(_capability_fragments("sales_mention"))
    elif artifact == "smalltalk":
        fragment_ids.append("examples.smalltalk")
    elif artifact == "unclear" and gate.side_effect_hint:
        fragment_ids.append("examples.multi_artifact_clarification")

    if (
        artifact == "unclear"
        and not gate.side_effect_hint
        and _looks_like_smalltalk_candidate(_user_visible_text(ctx.request.message).lower())
    ):
        fragment_ids.append("examples.smalltalk")

    if (
        "document_context" in ctx.policy.allowed_capabilities
        and not gate.side_effect_hint
        and artifact not in {"refusal", "human_support", "smalltalk"}
    ):
        fragment_ids.extend(_capability_fragments("document_context"))

    if artifact == "refusal" or gate.compliance_hint in {"blocked", "risky"}:
        fragment_ids.append("compliance.refusal_examples")

    if ctx.request.channel_type == "bank" and artifact == "material_pack":
        fragment_ids.append("channel.bank_material_rules")

    return _dedupe(fragment_ids)


def _knowledge_composer_fragments(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.knowledge_composer",
        _model_fragment(ctx.model_family),
        "output.reply_response_no_actions",
        "evidence.document_grounding",
        "style.wecom_concise_zh",
    ]


def _smalltalk_composer_fragments(ctx: PromptAssemblyContext) -> list[str]:
    return [
        "base.smalltalk_composer",
        _model_fragment(ctx.model_family),
        "output.reply_response_no_actions",
        "style.wecom_concise_zh",
    ]


def _capability_fragments(capability_name: str) -> list[str]:
    capability = capability_by_name(capability_name)
    if capability is None:
        return []
    return list(capability.planner_fragment_ids)


def _model_fragment(model_family: ModelFamily) -> str:
    if model_family in {"ds_v4pro", "deepseek"}:
        return "model.ds_v4pro.structured"
    return "model.generic.structured"


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return bool(_matches(text, keywords))


def _matches(text: str, keywords: tuple[str, ...]) -> list[str]:
    return [keyword for keyword in keywords if keyword.lower() in text]


def _limit(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= 20:
            break
    return output


def _user_visible_text(message: str) -> str:
    lines = []
    for line in message.splitlines():
        stripped = line.strip()
        if stripped.startswith("[adapter_") and stripped.endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _looks_like_send_shorthand(text: str) -> bool:
    if not text.strip() or _has_question_mark(text):
        return False
    artifact = _single_artifact_from_text(text)
    if artifact is None:
        return False
    # Only exact terse artifact nouns mean "send it".
    return _is_bare_artifact_shorthand(text, artifact)


def _is_bare_artifact_shorthand(text: str, artifact: str) -> bool:
    compact = _compact_command_text(text)
    keywords = _ARTIFACT_KEYWORDS.get(artifact, ())
    return any(compact == _compact_command_text(keyword) for keyword in keywords)


def _compact_command_text(text: str) -> str:
    return re.sub("[\\s,.\\u3001\\u3002\\uff0c\\uff0e\\uff1f?]+|\\u7684", "", text)


def _looks_like_smalltalk_candidate(text: str) -> bool:
    compact = _compact_command_text(text)
    if any(compact == _compact_command_text(hint) for hint in _SMALLTALK_HINTS):
        return True
    return any(
        _compact_command_text(hint) in compact
        for hint in (
            "你是谁",
            "你叫什么",
            "介绍一下你自己",
            "你是男是女",
            "男的女的",
        )
    )


def _material_detail_matches(text: str) -> list[str]:
    if any(token in text for token in ("材料包里", "材料里", "资料里", "一页通里")):
        return []
    matches = _matches(text, _MATERIAL_KEYWORDS)
    matches.extend(_matches(text, _MATERIAL_DETAIL_KEYWORDS))
    if not matches:
        return []
    if _looks_like_simple_process_question(text):
        return []
    return matches


def _performance_artifact(text: str) -> tuple[str | None, list[str]]:
    if _looks_like_performance_explanation_question(text):
        return None, []

    metric_matches = _matches(text, _PERFORMANCE_METRIC_KEYWORDS)
    if not metric_matches:
        return None, []

    monthly_matches = _matches(text, _MONTHLY_TIME_KEYWORDS)
    calendar_month_match = _matches_calendar_month(text)
    if (monthly_matches or calendar_month_match) and (
        "月报" in text
        or "表现" in text
        or "收益" in text
        or "亏了多少" in text
        or calendar_month_match
    ):
        return "monthly_report", [*monthly_matches, *metric_matches]

    weekly_matches = _matches(text, _WEEKLY_TIME_KEYWORDS)
    if weekly_matches:
        return "weekly_report", [*weekly_matches, *metric_matches]

    if any(
        token in text
        for token in (
            "最新净值",
            "绝对收益",
            "相对收益",
            "超额收益",
            "年化收益",
            "年化超额",
            "日胜率",
            "周胜率",
            "月胜率",
            "最大回撤",
            "超额夏普",
        )
    ):
        return "weekly_report", metric_matches

    return None, []


def _looks_like_performance_explanation_question(text: str) -> bool:
    if any(keyword in text for keyword in _PERFORMANCE_EXPLANATION_KEYWORDS):
        return True
    if "月报" in text and any(token in text for token in ("没有", "显示", "不一样")):
        return True
    return False


def _looks_like_simple_process_question(text: str) -> bool:
    return any(
        token in text
        for token in (
            "赎回流程",
            "流程图",
            "怎么预约赎回",
            "怎么赎回",
            "资金什么时候到账",
            "多久到账",
        )
    )


def _matches_calendar_month(text: str) -> bool:
    return bool(re.search(r"(?<!\d)(?:[1-9]|1[0-2])\s*月", text))


def _followup_side_effect_artifact(
    text: str,
    canonical_context: CanonicalContext,
    history: list[ConversationMessage] | None,
) -> str | None:
    if canonical_context.strategy_status != "resolved":
        return None
    if not _looks_like_strategy_only_followup(text):
        return None
    previous_user_text = _previous_user_text_for_pending_clarification(history)
    if not previous_user_text:
        return None
    return _single_artifact_from_text(previous_user_text)


def _looks_like_strategy_only_followup(text: str) -> bool:
    if not text.strip() or _has_question_mark(text):
        return False
    if _single_artifact_from_text(text) is not None:
        return False
    compact = _compact_command_text(text)
    return 0 < len(compact) <= 24


def _previous_user_text_for_pending_clarification(
    history: list[ConversationMessage] | None,
) -> str | None:
    if not history:
        return None
    last_assistant_index = None
    for index in range(len(history) - 1, -1, -1):
        if history[index].role == "assistant":
            last_assistant_index = index
            break
    if last_assistant_index is None:
        return None
    if not _assistant_was_clarification(history[last_assistant_index].content):
        return None
    for index in range(last_assistant_index - 1, -1, -1):
        if history[index].role == "user":
            return _user_visible_text(history[index].content).lower()
    return None


def _assistant_was_clarification(content: str) -> bool:
    try:
        payload = json.loads(content)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        reply = payload.get("reply")
        if isinstance(reply, dict) and reply.get("kind") == "clarification":
            return True
    return "再确认" in content and (
        "策略" in content or "材料包" in content or "周报" in content or "月报" in content
    )


def _single_artifact_from_text(text: str) -> str | None:
    matches = [
        artifact
        for artifact, keywords in _ARTIFACT_KEYWORDS.items()
        if _has_any(text, keywords)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _has_question_mark(text: str) -> bool:
    return "?" in text or "？" in text or "吗" in text


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _named_strategy_count(
    request: ReplyRequest,
    canonical_context: CanonicalContext,
) -> int:
    if canonical_context.strategy_status == "resolved":
        return 1
    if canonical_context.strategy_status == "ambiguous":
        return len(canonical_context.strategy_candidates)
    text = _user_visible_text(request.message).lower()
    return sum(
        1
        for strategy in request.available_strategies
        if strategy and strategy.lower() in text
    )
