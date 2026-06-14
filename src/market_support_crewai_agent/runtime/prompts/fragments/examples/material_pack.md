Material examples:
- "发一下中证1000材料" -> artifact_kind=material_pack, action_intent=send, selected_strategy="中证1000", report_scope=none, requested_capabilities=["material_pack"].
- "材料包发我" with one available strategy -> artifact_kind=material_pack, action_intent=send, selected_strategy from canonical context, report_scope=none.
- "A500和1000材料都发一下" -> ambiguity_slots includes strategy, action_intent=none, requested_capabilities=["material_pack"].
