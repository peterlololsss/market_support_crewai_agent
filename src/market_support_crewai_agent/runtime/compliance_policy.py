from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ComplianceReasonCode = Literal[
    "compliant_product_request",
    "customer_service_request",
    "expected_or_target_return",
    "principal_or_risk_guarantee",
    "peer_or_competitor_comparison",
    "private_contact_request",
    "proprietary_trading_or_core_strategy",
    "contract_or_restricted_document",
    "restricted_internal_document",
    "fee_waiver_request",
    "qualified_investor_or_threshold",
    "unrelated_request",
    "unknown",
]


@dataclass(frozen=True)
class ComplianceReasonSpec:
    reason_code: ComplianceReasonCode
    label: str
    planner_guidance: str
    safe_fallback_text: str = ""


COMPLIANCE_REASON_SPECS: tuple[ComplianceReasonSpec, ...] = (
    ComplianceReasonSpec(
        "compliant_product_request",
        "合规产品/业务请求",
        "Yanfu product/service requests, material/report requests, historical product data requests, and normal vague product questions.",
    ),
    ComplianceReasonSpec(
        "customer_service_request",
        "合规客户服务请求",
        "Human support, complaints, or requests to contact a named internal service person are compliant service needs.",
    ),
    ComplianceReasonSpec(
        "expected_or_target_return",
        "预计/目标/基准收益",
        "Expected return, target return, benchmark return, lowest return, maturity return, or return commitment requests must be refused.",
        "您好，我司产品不设置预计收益、目标收益或最低收益。过往业绩仅供参考，产品仅适合风险等级匹配的私募合格投资者。",
    ),
    ComplianceReasonSpec(
        "principal_or_risk_guarantee",
        "保本/安全/无风险",
        "Principal guarantee, product safety guarantee, risk-free, low-risk absolute wording, or minimum-return guarantee requests must be refused.",
        "您好，我司产品不承诺保本、最低收益或无风险。过往业绩仅供参考，产品仅适合风险等级匹配的私募合格投资者。",
    ),
    ComplianceReasonSpec(
        "peer_or_competitor_comparison",
        "同行/竞品评价",
        "Peer manager evaluation, competitor comparison, or public explanation of why another manager is better/worse must be refused.",
        "这个问题涉及同行评价，我不做横向评论。",
    ),
    ComplianceReasonSpec(
        "private_contact_request",
        "私人联系方式",
        "Private WeChat, phone, or off-channel business communication requests must be refused; normal human-support requests remain customer_service_request.",
        "老师请问具体是什么产品需求？业务问题请在当前群内沟通，便于留痕和统一回复。",
    ),
    ComplianceReasonSpec(
        "proprietary_trading_or_core_strategy",
        "自营盘/核心策略收益",
        "Proprietary account returns, self-operated account gains, or core-strategy profit questions must be refused.",
        "自营盘或内部收益信息我不展开。衍复各指数增强策略基于统一的 Alpha 多因子研究框架，对标不同指数获取超额。",
    ),
    ComplianceReasonSpec(
        "contract_or_restricted_document",
        "合同/受限文件",
        "Contract files or other restricted document delivery requests must be refused.",
        "因合规要求，合同及受限文件我无法直接提供。",
    ),
    ComplianceReasonSpec(
        "restricted_internal_document",
        "内部敏感材料",
        "Level-four valuation tables, performance attribution reports, or other sensitive internal documents must be refused unless an explicit adapter/evidence permission later allows them.",
        "因合规要求，四级估值表、业绩归因报告等内部敏感材料我无法直接提供。",
    ),
    ComplianceReasonSpec(
        "fee_waiver_request",
        "费用减免",
        "Requests to waive or change redemption fees or other contract-defined fees must be refused.",
        "赎回费等费用安排需严格按照基金合同及相关文件执行，不能临时减免。",
    ),
    ComplianceReasonSpec(
        "qualified_investor_or_threshold",
        "投资者适当性/门槛不符",
        "Requests to view or receive product promotion materials when qualification, channel, or threshold suitability is explicitly not satisfied must be refused.",
        "因合规要求，未确认适当性或不符合相应参与门槛时，我无法直接发送产品材料或进行产品推介。",
    ),
    ComplianceReasonSpec(
        "unrelated_request",
        "无关请求",
        "Requests unrelated to Yanfu products, services, market education, or customer service must be refused.",
        "这个问题与衍复产品或服务无关，我先不展开。",
    ),
    ComplianceReasonSpec(
        "unknown",
        "无法判断",
        "Use only when the message cannot be safely interpreted from current context; do not propose side-effect actions.",
        "这个问题我无法按当前合规要求展开。",
    ),
)

_REASON_SPEC_BY_CODE: dict[ComplianceReasonCode, ComplianceReasonSpec] = {
    spec.reason_code: spec for spec in COMPLIANCE_REASON_SPECS
}
NON_COMPLIANT_REASON_CODES: tuple[ComplianceReasonCode, ...] = tuple(
    spec.reason_code
    for spec in COMPLIANCE_REASON_SPECS
    if spec.safe_fallback_text
    and spec.reason_code not in {"compliant_product_request", "customer_service_request"}
)


def safe_fallback_text(reason_code: str) -> str:
    spec = _REASON_SPEC_BY_CODE.get(reason_code)  # type: ignore[arg-type]
    if spec is None or not spec.safe_fallback_text:
        return _REASON_SPEC_BY_CODE["unknown"].safe_fallback_text
    return spec.safe_fallback_text


def compliance_policy_prompt_lines() -> list[str]:
    return [
        f"- {spec.reason_code}: {spec.planner_guidance}"
        for spec in COMPLIANCE_REASON_SPECS
    ]
