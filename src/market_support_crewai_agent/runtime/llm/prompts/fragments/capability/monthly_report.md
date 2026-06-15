Capability: monthly_report.

Use this for monthly report send requests and for direct calendar-month monthly performance questions. A named strategy inside a monthly report request remains monthly_report.

Monthly report intent includes "发月报", "X月月报", "有没有X月份月报", and direct questions such as "X月表现怎么样", "上个月表现怎么样", or "这个月亏了多少". If the user asks a non-calendar recent period such as "最近一个月超额怎么样", prefer weekly_report because the latest weekly report carries continuously updated performance data.

For an unnamed current-channel monthly report send, use report_scope=channel_all and selected_strategy=null.

Do not ask for a strategy only because Request metadata lists multiple available_strategies. Treat available_strategies as a material_pack catalog, not a monthly_report selection menu.

For one clearly named strategy monthly report send, use report_scope=strategy and selected_strategy set to that strategy. Adapter evidence later decides whether the report covers the strategy.
