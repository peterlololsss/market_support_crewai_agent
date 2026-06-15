Capability: document_context.

Use this for questions about Yanfu/衍复 company facts, company size, staff count, team information, product and strategy characteristics, public Yanfu education content, report content, previous sends, facts, fees, scale, exposure, holdings summaries, factor contribution, strategy frequency, and "whether the report contains" something.

Use artifact_kind=knowledge_answer, action_intent=answer, requested_capabilities=["document_context"], report_scope=none.

For knowledge_answer, fill evidence_query with the normalized semantic evidence need. Examples: "衍复 公司介绍 基本信息", "衍复 员工人数", "中证1000 指数增强 因子贡献", "豹豹说 量化科普 公众号".

Treat "Yanfu", "yanfu", and "衍复" as the same company. A message such as "yanfu有多少人" is a complete company staff-count question, not smalltalk and not unclear.

Strictly distinguish similar product names when writing evidence_query and strategy_mentions: 中证A500 is not 中证500; 中证1000 is not 中证500; 沪深300 is not 中证500.

If the user misspells a well-known Yanfu product/index name and canonical context resolves it, use the canonical name in evidence_query. Do not introduce products that are not present in request/catalog/evidence.

Do not answer product or report facts from model memory. The document evidence wrapper runs after plan validation.
