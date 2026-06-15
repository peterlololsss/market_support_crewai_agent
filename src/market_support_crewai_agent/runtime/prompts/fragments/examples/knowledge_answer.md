Knowledge-answer examples:
- "这个策略怎么样" -> artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], evidence_query="相关策略 基本介绍 特点".
- "报告里有没有小市值" -> artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], evidence_query="报告 小市值 是否包含".
- "刚发的周报为什么没有某策略" -> artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], evidence_query="周报 策略 生成范围".
- "yanfu有多少人" / "衍复现在多少人" / "可以介绍下投研团队人员吗" -> artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], evidence_query="衍复 公司介绍 员工人数 团队".
- "豹豹说讲因子的那篇可以发一下吗" / "你们有微信公众号吗" -> artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], evidence_query="衍复 豹豹说 量化科普 公众号".
- "1000指增超额收益占比" / "小市值量价因子占比多少" -> artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], evidence_query="对应策略 超额收益 因子贡献 占比".
- "全指都选了什么票" / "中性2号现在持仓多少" / "灵活对冲策略现在敞口是多少" -> artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], evidence_query="对应策略 持仓 敞口 概况".
- "为什么月报里没有显示产品的年化收益率" / "产品净值更新是保管费后的吗" -> artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], evidence_query="报告展示 费用扣除 净值口径".
- "介绍一下宗曾全子" -> use canonical context if resolved to 中证全指; artifact_kind=knowledge_answer, action_intent=answer, evidence_query="中证全指 指数增强 基本介绍".

Do not classify Yanfu/衍复 company, staff-count, or team questions as smalltalk.
