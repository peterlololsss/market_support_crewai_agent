Material examples:
- "发一下中证1000材料" -> artifact_kind=material_pack, action_intent=send, selected_strategy="中证1000", report_scope=none, requested_capabilities=["material_pack"].
- "材料包发我" with one available strategy -> artifact_kind=material_pack, action_intent=send, selected_strategy from canonical context, report_scope=none.
- "中性一号的一页通有没有" / "给我中性一号的一夜痛" -> artifact_kind=material_pack, action_intent=send, selected_strategy from canonical context, report_scope=none.
- "小市值下周有哪些产品可以买" / "这个代码什么时候开放" -> artifact_kind=material_pack, action_intent=send, requested_capabilities=["material_pack"].
- "封闭期多长" / "预警止损线是多少" / "10万可以追加吗" -> artifact_kind=material_pack, action_intent=send, requested_capabilities=["material_pack"].
- "代销的小市值指增产品有哪些" -> artifact_kind=material_pack, action_intent=send, selected_strategy="万得小市值" if canonical context resolves it.
- "材料包里显示的收益都是计提后的吗" -> artifact_kind=knowledge_answer if document_context is allowed; do not send another material pack just because "材料包" appears.
- "赎回流程图发一下" / "赎回资金什么时候到账" -> not material_pack unless request/catalog metadata explicitly says this is an adapter-sendable material.
- "A500和1000材料都发一下" -> ambiguity_slots includes strategy, action_intent=none, requested_capabilities=["material_pack"].
