Multi-artifact clarification examples:
- "材料和周报都发一下" -> artifact_kind=unclear, action_intent=none, ambiguity_slots includes artifact.
- "A500材料和1000周报都发一下" -> artifact_kind=unclear, action_intent=none, ambiguity_slots includes artifact and strategy.

Do not inject multiple action capability intents into one normal planner output. Ask for clarification through ambiguity_slots.
