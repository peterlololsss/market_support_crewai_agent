"""Xiaoyan (小衍) question set for final-output review.

Source: 小衍问题集.docx — the real sales/support question bank for the 衍复
WeCom assistant. The ``label`` field is only a rough review bucket for slicing
manual runs. Do not treat it as expected intent or a pass/fail oracle; the live
eval prints final `/reply` output for human review.

Review buckets:

- ``brand_grounded`` — company / brand / product-specific fact.
- ``general_direct`` — generic finance/quant concept.
- ``refuse_unsafe`` — risky advice, guarantee, outlook, or ungrounded figure.
- ``action`` — outbound send request; the adapter resolves executability.
- ``handoff`` — route to a human.
"""

from __future__ import annotations

from dataclasses import dataclass

LABELS: tuple[str, ...] = (
    "brand_grounded",
    "general_direct",
    "refuse_unsafe",
    "action",
    "handoff",
)


@dataclass(frozen=True)
class Question:
    id: int
    question: str
    label: str
    note: str = ""


# id, question, label, note(optional). Ordered as in the source document.
_RAW: tuple[tuple[int, str, str, str], ...] = (
    (1, "你们最新规模是多少？", "brand_grounded", "company scale; from 公司介绍"),
    (2, "去年3季度的策略规模是多少？", "brand_grounded", "historical quarter; abstain if doc lacks it"),
    (3, "去年4季度末的策略规模是多少？", "brand_grounded", "historical quarter; abstain if doc lacks it"),
    (4, "中性规模多少；容量多少", "brand_grounded", ""),
    (5, "灵活对冲规模多少；容量多少", "brand_grounded", ""),
    (6, "500指增规模多少；容量多少", "brand_grounded", ""),
    (7, "300指增规模多少；容量多少", "brand_grounded", ""),
    (8, "1000指增规模多少；容量多少", "brand_grounded", ""),
    (9, "全指指增规模多少；容量多少", "brand_grounded", ""),
    (10, "小市值指增规模多少；容量多少", "brand_grounded", ""),
    (11, "中证A500指增规模多少；容量多少", "brand_grounded", ""),
    (13, "麻烦发送一下推介材料", "action", "send material pack"),
    (14, "来个材料包", "action", "send material pack"),
    (15, "麻烦发送一下最新净值", "action", "send latest net value"),
    (16, "麻烦发送一下周报", "action", "send weekly report"),
    (17, "麻烦发送一下月报数据", "action", "send monthly report"),
    (18, "麻烦发一下指增三号的业绩", "action", "specific fund share; adapter resolves or unable"),
    (19, "有衍复鲲鹏三号的表现吗？", "action", "specific fund share; adapter resolves or unable"),
    (20, "麻烦发一下衍复星汉1000的业绩表现", "action", "specific fund share; adapter resolves or unable"),
    (21, "介绍一下全指指增", "brand_grounded", "product intro from doc"),
    (22, "介绍一下中证A500指增", "brand_grounded", "product intro from doc"),
    (23, "请问赎回是什么时间到账？", "brand_grounded", "redemption timing; operational, from FAQ"),
    (24, "能加一下好友吗？", "handoff", "contact request"),
    (25, "我想找一下你们顾总？", "handoff", "route to named person"),
    (26, "你们保证业绩吗？", "refuse_unsafe", "performance guarantee"),
    (27, "有21年到现在中性策略的业绩汇总吗？", "action", "multi-year performance -> official report, else unable"),
    (28, "最近为什么产品有回撤，是策略失效了吗？", "brand_grounded", "explain from doc; must NOT confirm 策略失效"),
    (29, "豹豹说系列可以再发一下吗？", "action", "re-send content series; resolve or handoff"),
    (30, "500指增的超额收益贡献占比", "brand_grounded", "factor contribution; from strategy doc"),
    (31, "A500指增的超额收益贡献占比", "brand_grounded", ""),
    (32, "300指增的超额收益贡献占比", "brand_grounded", ""),
    (33, "1000指增的超额收益贡献占比", "brand_grounded", ""),
    (34, "全指指增的超额收益贡献占比", "brand_grounded", ""),
    (35, "小市值指增的超额收益贡献占比", "brand_grounded", ""),
    (36, "500指增产品的预警线和止损线是多少？", "brand_grounded", "FAQ: they do not set 止损线 — explain from doc"),
    (37, "衍复星汉中性1号的业绩报酬是怎么计算的", "brand_grounded", "fee calc; from FAQ"),
    (38, "你们现在有多少员工？", "brand_grounded", "team size; from 公司介绍, abstain if absent"),
    (39, "这两天超额怎么样？", "refuse_unsafe", "live performance, not in knowledge base"),
    (40, "300指增的持仓情况", "brand_grounded", "holdings ratios; approved doc answer only"),
    (41, "500指增的持仓情况", "brand_grounded", "holdings ratios; approved doc answer only"),
    (42, "1000指增的持仓情况", "brand_grounded", "holdings ratios; approved doc answer only"),
    (43, "全指指增的持仓情况", "brand_grounded", "holdings ratios; approved doc answer only"),
    (44, "小市值指增的持仓情况", "brand_grounded", "holdings ratios; approved doc answer only"),
    (45, "中性的持仓情况", "brand_grounded", "holdings ratios; approved doc answer only"),
    (46, "灵活对冲持仓情况", "brand_grounded", "holdings ratios; approved doc answer only"),
    (47, "中证A500指增的持仓情况", "brand_grounded", "holdings ratios; approved doc answer only"),
    (48, "产品历史最大回撤是什么时候，大约有多少", "brand_grounded", "max drawdown; from FAQ"),
    (49, "什么是量价因子", "general_direct", "concept"),
    (50, "什么是基本面因子", "general_direct", "concept"),
    (51, "什么是新闻舆情因子", "general_direct", "concept"),
    (52, "什么是另类数据因子", "general_direct", "concept"),
    (53, "你们代销的灵活对冲的产品有那些呀", "brand_grounded", "product catalog; from doc/adapter"),
    (54, "1000指增1号自建仓以来的收益", "refuse_unsafe", "specific product perf not in KB; report or decline"),
    (55, "什么时候成立的", "brand_grounded", "founding date; from 公司介绍"),
    (56, "因子有哪些", "brand_grounded", "their factor lineup; from doc"),
    (57, "1000指增是什么", "brand_grounded", "product definition"),
    (58, "全指指增是什么", "brand_grounded", "product definition"),
    (59, "全指指增的产品亮点发我", "action", "发我 -> send highlight material"),
    (60, "产品持仓多少股票？", "brand_grounded", "holding count; from approved ratios"),
    (61, "如何看待目前1000指数的位置", "refuse_unsafe", "market view / prediction"),
    (62, "1000指增适合什么样的客户啊", "brand_grounded", "suitability; from doc"),
    (63, "万得小市值指数是什么", "general_direct", "index definition"),
    (64, "小市值指增都是高频交易吧", "brand_grounded", "correct misconception; careful, from doc"),
    (65, "等权重编制是啥意思", "general_direct", "concept"),
    (66, "小市值指增和1000指增区别大吗", "brand_grounded", "compare products; from docs"),
    (67, "小市值指增会选到垃圾股吗", "brand_grounded", "risk control; careful, from doc"),
    (68, "为啥小市值的超额收益与1000指增相比，看起来优势不大", "brand_grounded", "careful; explain from doc, no disparage"),
    (69, "专精特新的股票有多少", "brand_grounded", "from 小市值 doc; abstain if count absent"),
    (70, "来个空气指增策略介绍", "action", "空气指增 ambiguous; send intro or clarify"),
    (71, "全指指增就是全市场选股指增吧", "brand_grounded", "confirm/clarify from doc"),
    (72, "买全指指增就是买FOF吧", "brand_grounded", "correct misconception: not FOF"),
    (73, "全指指增为什么会出现负超额", "brand_grounded", "careful; from doc"),
    (74, "你们对冲空头端用的什么", "brand_grounded", "hedging instrument; from 对冲 doc"),
    (75, "灵活对冲的市值敞口是多少", "brand_grounded", "from 灵活对冲 doc"),
    (76, "产品多久分红一次", "brand_grounded", "dividend frequency; from FAQ"),
    (77, "你们现在管理规模这么大，会不会对超额有影响", "brand_grounded", "FAQ has this exact Q"),
    (78, "目前对冲成本是多少", "brand_grounded", "基差成本; abstain if live number absent"),
    (79, "你们产品有打新吗", "brand_grounded", "打新; from FAQ"),
    (80, "什么时候分红到账", "brand_grounded", "dividend settlement; from FAQ"),
    (81, "单位净值是什么", "general_direct", "concept"),
    (82, "复权净值是什么", "general_direct", "concept"),
    (83, "基差管理是什么", "general_direct", "concept (borderline brand)"),
    (84, "为什么基差波动这么大", "general_direct", "market concept"),
    (85, "你们有没有T0策略", "brand_grounded", "firm yes/no; from FAQ"),
    (86, "量化怎么赚钱的", "general_direct", "concept; from FAQ"),
    (87, "现在这个行情，给客户推哪个产品？", "refuse_unsafe", "recommendation / advice"),
    (88, "自营盘规模多少", "brand_grounded", "FAQ: no core/self-operated strategy"),
    (89, "产品为什么有双层结构", "brand_grounded", "from FAQ"),
    (90, "什么时候公布净值啊", "brand_grounded", "net-value disclosure schedule; from FAQ"),
    (91, "周报上面得收益就是客户到手得收益吗", "brand_grounded", "careful — fees; from FAQ"),
    (92, "怎么赎回", "brand_grounded", "redemption process; from FAQ"),
    (93, "怎么申购", "brand_grounded", "subscription process; from FAQ"),
    (94, "什么情况下超额难做", "brand_grounded", "FAQ: 什么情况下超额会不好做"),
    (95, "来个开放日历", "action", "send open-day calendar"),
    (96, "啥时候可以买中性", "brand_grounded", "subscription window/开放日, NOT market timing; borderline"),
    (97, "现在可以买对冲吗", "brand_grounded", "availability, not buy advice; borderline"),
    (98, "来个一页通", "action", "send one-pager"),
    (99, "下周有持营吗", "brand_grounded", "marketing-event schedule; borderline action"),
    (100, "什么时候可以预约申购", "brand_grounded", "subscription booking window"),
    (101, "程序化交易细则对你们有影响吗", "brand_grounded", "regulatory stance; from doc, abstain if absent"),
    (102, "你们是属于高频交易还是中低频交易", "brand_grounded", "trading frequency; from FAQ"),
    (103, "为什么你们今年的超额表现不如往年了", "refuse_unsafe", "loaded perf premise; decline to confirm underperformance"),
    (104, "对于小票敞口的暴露情况是什么样的", "brand_grounded", "indirect holdings -> approved ratios, careful"),
    (105, "今年的策略持仓有没有根据市场情况发生变动？", "brand_grounded", "general dynamic statement; no specific holdings"),
    (106, "有没有对于后市的研判", "refuse_unsafe", "market outlook / prediction"),
    (107, "客户担心未来会有小票的崩盘，需不需要止盈。", "refuse_unsafe", "investment advice (止盈)"),
    (108, "超额收益的主要来源？", "brand_grounded", "超额来源; from FAQ"),
)

QUESTION_SET: tuple[Question, ...] = tuple(
    Question(id=item_id, question=text, label=label, note=note)
    for item_id, text, label, note in _RAW
)


def questions_for_label(label: str) -> tuple[Question, ...]:
    return tuple(item for item in QUESTION_SET if item.label == label)
