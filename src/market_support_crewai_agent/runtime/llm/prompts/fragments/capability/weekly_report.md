Capability: weekly_report.

Use this for weekly report send requests and for performance-data questions that should be satisfied by the latest weekly report. A named strategy inside a weekly report request remains weekly_report.

Performance-data questions include historical/period metrics such as absolute return, relative/excess return, annualized return, annualized excess return, day/week/month win-rate, NAV, latest NAV, max drawdown, drawdown repair period, excess max drawdown, excess Sharpe, recent performance, "这周/上周/近期怎么样", "赚钱了吗", or "亏了多少".

If the user asks why a metric behaves a certain way, asks for a definition/calculation method, asks for factor contribution/source/占比, or asks why a report does not show a metric, prefer knowledge_answer when document_context is allowed instead of weekly_report.

For an unnamed current-channel weekly report send, use report_scope=channel_all and selected_strategy=null.

Do not ask for a strategy only because Request metadata lists multiple available_strategies. Treat available_strategies as a material_pack catalog, not a weekly_report selection menu.

For one clearly named strategy weekly report send, use report_scope=strategy and selected_strategy set to that strategy. Adapter evidence later decides whether the report covers the strategy.
