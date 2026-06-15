Smalltalk examples:

- User says "hi", "hello", "zaima", "nihao", "thanks", or another simple conversational check-in:
  artifact_kind=smalltalk, action_intent=none, report_scope=none, ambiguity_slots=[], requested_capabilities=[].

- User asks "what can you do", "how can you help", or another no-action capability/help question:
  artifact_kind=smalltalk, action_intent=none, report_scope=none, ambiguity_slots=[], requested_capabilities=[].

- User asks "你是谁" or "介绍一下你自己":
  artifact_kind=smalltalk, action_intent=none, report_scope=none, ambiguity_slots=[], requested_capabilities=[].

- User asks "你是男是女" or another direct gender question:
  artifact_kind=smalltalk, action_intent=none, report_scope=none, ambiguity_slots=[], requested_capabilities=[].

- Do not request adapter resolves for smalltalk.
- Do not produce a final reply text in the planner. The no-action composer will write the visible reply.
