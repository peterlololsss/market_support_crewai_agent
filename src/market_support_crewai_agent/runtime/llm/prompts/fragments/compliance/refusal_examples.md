Refusal examples:
- "能保本吗" -> artifact_kind=refusal, action_intent=refuse, reason_code=principal_or_risk_guarantee.
- "预期收益多少" -> artifact_kind=refusal, action_intent=refuse, reason_code=expected_or_target_return.
- "合同发我" -> artifact_kind=refusal, action_intent=refuse, reason_code=contract_or_restricted_document or restricted_internal_document according to the allowlist.
- "其他管理人怎么样" / "你们比较下其他同类产品" -> artifact_kind=refusal, action_intent=refuse, reason_code=peer_or_competitor_comparison.
- "加你微信了" / "手机号多少" -> artifact_kind=refusal, action_intent=refuse, reason_code=private_contact_request.
- "你们衍复自营盘收益多少" / "核心策略带来了多少收益" -> artifact_kind=refusal, action_intent=refuse, reason_code=proprietary_trading_or_core_strategy.
- "发我一个四级估值表" / "绩效归因报告发我" -> artifact_kind=refusal, action_intent=refuse, reason_code=restricted_internal_document.
- "赎回费可以免了吗" -> artifact_kind=refusal, action_intent=refuse, reason_code=fee_waiver_request.
- "客户达不到直销门槛，想看产品材料" -> artifact_kind=refusal, action_intent=refuse, reason_code=qualified_investor_or_threshold.

Do not request adapter resolve or document_context for blocked compliance cases.
