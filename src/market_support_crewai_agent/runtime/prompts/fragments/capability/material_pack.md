Capability: material_pack.

Use this for clear requests to send official product materials or for product-element questions that should be handled through official material pack resolution. Do not classify weekly_report or monthly_report requests as material_pack.

For a clear material send request, use artifact_kind=material_pack, action_intent=send, requested_capabilities=["material_pack"], report_scope=none.

Material words include: 材料包, 产品资料, 宣传材料, 推介材料, 对客材料, PPT, 一页通/一夜通/一夜痛, 要素表, 开放日历, 排期表, 销售日期表, 推荐资料, 培训视频.

Product-element words include: 费率, 赎回费, 认购费, 申购费, 管理费, 业绩报酬/计提, 封闭期, 锁定期, 开放日, 申购日, 赎回日, 持营, 预约申购, 可追加金额, 预警线, 止损线, 平仓线, 历史分红数据, 哪些产品可以买, 哪个代码能买.

If the user is asking about content already seen inside a material, such as "材料包里显示的收益是否计提后", prefer knowledge_answer when document_context is allowed; do not trigger another material send only because the word "材料包" appears.

If multiple strategies are requested in one send request, mark ambiguity_slots with strategy and do not force a combined send intent.
