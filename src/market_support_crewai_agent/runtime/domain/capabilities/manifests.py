from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    AbstentionPolicy,
    CapabilityManifest,
    EvidenceContract,
)

_REPLY_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["reply"],
    "properties": {
        "reply": {
            "type": "object",
            "required": ["kind", "text"],
            "properties": {
                "kind": {"type": "string"},
                "text": {"type": "string"},
            },
        },
        "actions": {"type": "array"},
    },
}

_STANDARD_VERIFIER_CHECKS = [
    "output_schema",
    "required_runtime_input_present",
    "required_evidence_present",
    "evidence_artifact_type_allowed",
    "forbidden_source_not_used",
    "abstention_correctness",
]
_OUTPUT_ONLY_VERIFIER_CHECKS = ["output_schema"]

_ACTION_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["reply", "actions"],
    "properties": {
        "reply": {
            "type": "object",
            "required": ["kind", "text"],
            "properties": {
                "kind": {"type": "string"},
                "text": {"type": "string"},
            },
        },
        "actions": {"type": "array"},
    },
}


BUILTIN_CAPABILITY_MANIFESTS: tuple[CapabilityManifest, ...] = (
    CapabilityManifest(
        id="material_pack.send",
        version="2026-06-16.1",
        display_name="Send material pack",
        description="Propose sending the adapter-resolved material pack for the current channel and strategy scope.",
        capability_type="action",
        domain_entities=["channel", "strategy", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=["canonical_context.selected_strategy"],
        required_artifacts=["material_pack"],
        allowed_artifacts=["material_pack"],
        forbidden_artifacts=["weekly_report", "monthly_report", "document_context"],
        required_tools=["adapter_resolve.material_pack"],
        output_schema=_ACTION_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            required_evidence_types=["material_pack_resolvable"],
            allowed_source_types=["adapter_resolve"],
            disallowed_source_types=[
                "adapter_report_scope",
                "document_mcp",
                "conversation_history",
            ],
            required_artifact_types=["material_pack"],
            allowed_artifact_types=["material_pack"],
            minimum_evidence_count=1,
        ),
        abstention_policy=AbstentionPolicy(
            abstention_reply_kinds=["unable_to_answer", "clarification", "human_handoff"],
            guidance="If the material pack cannot be resolved, abstain or hand off instead of inventing a send."
        ),
        planner_guidance=(
            "Use when the user clearly asks to send a material pack or one-pager."
        ),
        agent_guidance="Return only the typed send_material_pack action after adapter resolve evidence.",
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=["Send the material pack.", "发一下中证1000材料"],
        examples_negative=["Which products are in the weekly report?"],
        runtime_capability="material_pack",
    ),
    CapabilityManifest(
        id="weekly_report.send",
        version="2026-06-16.1",
        display_name="Send weekly report",
        description="Propose sending the adapter-resolved weekly report for the current channel or strategy scope.",
        capability_type="action",
        domain_entities=["channel", "strategy", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=["canonical_context.selected_strategy"],
        required_artifacts=["weekly_report"],
        allowed_artifacts=["weekly_report"],
        forbidden_artifacts=["material_pack", "monthly_report", "document_context"],
        required_tools=["adapter_resolve.weekly_report"],
        output_schema=_ACTION_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            required_evidence_types=["weekly_report_resolvable"],
            allowed_source_types=["adapter_resolve"],
            disallowed_source_types=[
                "adapter_report_scope",
                "document_mcp",
                "conversation_history",
            ],
            required_artifact_types=["weekly_report"],
            allowed_artifact_types=["weekly_report"],
            minimum_evidence_count=1,
        ),
        abstention_policy=AbstentionPolicy(
            abstention_reply_kinds=["unable_to_answer", "clarification", "human_handoff"],
            guidance="If the weekly report cannot be resolved for the requested scope, abstain or hand off."
        ),
        planner_guidance=(
            "Use when the user clearly asks to send the weekly report."
        ),
        agent_guidance="Return only the typed send_weekly_report action after adapter resolve evidence.",
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=["Send this week's report.", "请发一下周报"],
        examples_negative=["What products are in the weekly report?"],
        runtime_capability="weekly_report",
    ),
    CapabilityManifest(
        id="monthly_report.send",
        version="2026-06-16.1",
        display_name="Send monthly report",
        description="Propose sending the adapter-resolved monthly report for the current channel or strategy scope.",
        capability_type="action",
        domain_entities=["channel", "strategy", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=["canonical_context.selected_strategy"],
        required_artifacts=["monthly_report"],
        allowed_artifacts=["monthly_report"],
        forbidden_artifacts=["material_pack", "weekly_report", "document_context"],
        required_tools=["adapter_resolve.monthly_report"],
        output_schema=_ACTION_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            required_evidence_types=["monthly_report_resolvable"],
            allowed_source_types=["adapter_resolve"],
            disallowed_source_types=[
                "adapter_report_scope",
                "document_mcp",
                "conversation_history",
            ],
            required_artifact_types=["monthly_report"],
            allowed_artifact_types=["monthly_report"],
            minimum_evidence_count=1,
        ),
        abstention_policy=AbstentionPolicy(
            abstention_reply_kinds=["unable_to_answer", "clarification", "human_handoff"],
            guidance="If the monthly report cannot be resolved for the requested scope, abstain or hand off."
        ),
        planner_guidance=(
            "Use when the user clearly asks to send the monthly report."
        ),
        agent_guidance="Return only the typed send_monthly_report action after adapter resolve evidence.",
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=["Send the monthly report.", "发我个月报"],
        examples_negative=["What period does the weekly report cover?"],
        runtime_capability="monthly_report",
    ),
    CapabilityManifest(
        id="sales.handoff",
        version="2026-06-16.1",
        display_name="Sales handoff",
        description="Route the request to the adapter-resolved sales or support mention.",
        capability_type="handoff",
        domain_entities=["channel"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[],
        required_artifacts=["adapter_context"],
        allowed_artifacts=["adapter_context"],
        forbidden_artifacts=["material_pack", "weekly_report", "monthly_report"],
        required_tools=["adapter_resolve.sales_mention"],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            required_evidence_types=["sales_mention_resolvable"],
            allowed_source_types=["adapter_resolve"],
            required_artifact_types=["adapter_context"],
            allowed_artifact_types=["adapter_context"],
            minimum_evidence_count=1,
        ),
        abstention_policy=AbstentionPolicy(
            guidance="If sales mention cannot be resolved, return unable_to_answer."
        ),
        planner_guidance="Use when the user asks for human sales/support help.",
        agent_guidance="Mention only adapter-resolved sales/support targets.",
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=["帮我问下销售", "请销售确认"],
        examples_negative=["发一下周报"],
        runtime_capability="sales_mention",
    ),
    CapabilityManifest(
        id="general.clarification",
        version="2026-06-16.1",
        display_name="Clarification",
        description="Ask a concise clarification when the request is ambiguous.",
        capability_type="handoff",
        domain_entities=["channel"],
        required_inputs=[],
        optional_inputs=[],
        required_artifacts=[],
        allowed_artifacts=[],
        forbidden_artifacts=[],
        required_tools=[],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(fallback_policy="clarify"),
        abstention_policy=AbstentionPolicy(
            requires_abstention_when_evidence_missing=False,
            abstention_reply_kinds=["clarification"],
        ),
        planner_guidance="Use when required artifact, strategy, or request meaning is ambiguous.",
        agent_guidance="Ask one concise clarification and do not propose actions.",
        verifier_checks=_OUTPUT_ONLY_VERIFIER_CHECKS,
        examples_positive=["你说的这个是哪个策略？"],
        examples_negative=["发一下周报"],
    ),
    CapabilityManifest(
        id="general.abstention",
        version="2026-06-16.1",
        display_name="Evidence abstention",
        description="Return unable_to_answer when required evidence is missing.",
        capability_type="handoff",
        domain_entities=["channel"],
        required_inputs=[],
        optional_inputs=[],
        required_artifacts=[],
        allowed_artifacts=[],
        forbidden_artifacts=[],
        required_tools=[],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(fallback_policy="abstain"),
        abstention_policy=AbstentionPolicy(
            requires_abstention_when_evidence_missing=False,
            abstention_reply_kinds=["unable_to_answer"],
        ),
        planner_guidance="Use when the request is understood but required evidence is absent.",
        agent_guidance="State inability concisely and do not invent facts.",
        verifier_checks=_OUTPUT_ONLY_VERIFIER_CHECKS,
        examples_positive=["当前没有足够证据安全回复。"],
        examples_negative=["已发送周报"],
    ),
    CapabilityManifest(
        id="general.refusal",
        version="2026-06-16.1",
        display_name="Compliance refusal",
        description="Refuse non-compliant or unrelated requests.",
        capability_type="handoff",
        domain_entities=["channel"],
        required_inputs=[],
        optional_inputs=[],
        required_artifacts=[],
        allowed_artifacts=[],
        forbidden_artifacts=[],
        required_tools=[],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(fallback_policy="abstain"),
        abstention_policy=AbstentionPolicy(
            requires_abstention_when_evidence_missing=False,
            abstention_reply_kinds=["unable_to_answer"],
        ),
        planner_guidance="Use for requests that compliance policy requires refusing.",
        agent_guidance="Use the harness refusal text and do not propose actions.",
        verifier_checks=_OUTPUT_ONLY_VERIFIER_CHECKS,
        examples_positive=["这个产品能保本吗"],
        examples_negative=["请发周报"],
    ),
    CapabilityManifest(
        id="general.smalltalk",
        version="2026-06-16.1",
        display_name="Smalltalk",
        description="Answer greetings, thanks, identity, or capability-help questions with no business action.",
        capability_type="answer",
        domain_entities=["channel"],
        required_inputs=[],
        optional_inputs=[],
        required_artifacts=[],
        allowed_artifacts=[],
        forbidden_artifacts=[],
        required_tools=[],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(fallback_policy="ignore"),
        abstention_policy=AbstentionPolicy(
            requires_abstention_when_evidence_missing=False,
            abstention_reply_kinds=["answer"],
        ),
        planner_guidance="Use for greetings, thanks, self-introduction, or help/capability questions.",
        agent_guidance="Answer briefly from request metadata and policy only.",
        verifier_checks=_OUTPUT_ONLY_VERIFIER_CHECKS,
        examples_positive=["你是谁", "谢谢"],
        examples_negative=["发一下材料"],
    ),
    CapabilityManifest(
        id="general.no_reply",
        version="2026-06-16.1",
        display_name="No reply",
        description="Produce no reply when the harness should stay silent.",
        capability_type="handoff",
        domain_entities=["channel"],
        required_inputs=[],
        optional_inputs=[],
        required_artifacts=[],
        allowed_artifacts=[],
        forbidden_artifacts=[],
        required_tools=[],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(fallback_policy="ignore"),
        abstention_policy=AbstentionPolicy(
            requires_abstention_when_evidence_missing=False,
            abstention_reply_kinds=["no_reply"],
        ),
        planner_guidance="Use only when the message should not receive a visible reply.",
        agent_guidance="Return no_reply with empty text, mentions, and actions.",
        verifier_checks=_OUTPUT_ONLY_VERIFIER_CHECKS,
        examples_positive=["adapter-only status messages that do not need a reply"],
        examples_negative=["user asks for help"],
    ),
    CapabilityManifest(
        id="material_pack.product_list",
        version="2026-06-16.1",
        display_name="Material pack product list",
        description="Answer which products are present in an official material pack.",
        capability_type="answer",
        domain_entities=["channel", "strategy", "product", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[
            "canonical_context.selected_strategy",
            "request.available_strategies",
        ],
        required_artifacts=["material_pack"],
        allowed_artifacts=["material_pack"],
        forbidden_artifacts=["weekly_report", "monthly_report"],
        required_tools=["adapter_material_pack_content.products"],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            required_fact_types=["material_pack_product_list"],
            allowed_source_types=["adapter_material_pack_content"],
            forbidden_source_types=["adapter_report_scope", "document_mcp"],
            required_artifact_types=["material_pack"],
            allowed_artifact_types=["material_pack"],
            min_facts=1,
            notes=(
                "Report scope, document context, and conversation history cannot "
                "substitute for material-pack product-list evidence."
            ),
        ),
        abstention_policy=AbstentionPolicy(
            guidance=(
                "If material-pack product-list evidence is absent, return an "
                "unable-to-answer response instead of answering from reports."
            )
        ),
        planner_guidance=(
            "Use when the user asks what products are inside a material pack. "
            "Do not satisfy this capability from weekly or monthly report evidence."
        ),
        agent_guidance=(
            "Compose only from material-pack product-list evidence. Abstain when "
            "the material-pack content artifact was not fetched."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            "What products are in the material pack?",
            "Which products does this one-pager include?",
        ],
        examples_negative=[
            "Send me the material pack.",
            "Which products are in the weekly report?",
        ],
        runtime_capability="material_pack",
    ),
    CapabilityManifest(
        id="material_pack.open_calendar",
        version="2026-06-16.1",
        display_name="Material pack open calendar",
        description="Answer product opening or subscription calendar questions from a material pack.",
        capability_type="answer",
        domain_entities=["channel", "strategy", "product", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[
            "canonical_context.selected_strategy",
            "request.available_strategies",
        ],
        required_artifacts=["material_pack"],
        allowed_artifacts=["material_pack"],
        forbidden_artifacts=["weekly_report", "monthly_report"],
        required_tools=["adapter_material_pack_content.open_calendar"],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            required_fact_types=["material_pack_open_calendar"],
            allowed_source_types=["adapter_material_pack_content"],
            forbidden_source_types=["adapter_report_scope", "document_mcp"],
            required_artifact_types=["material_pack"],
            allowed_artifact_types=["material_pack"],
            min_facts=1,
            notes=(
                "Opening-calendar answers require material-pack content evidence; "
                "sendability evidence alone is not enough."
            ),
        ),
        abstention_policy=AbstentionPolicy(
            guidance=(
                "If the open-calendar artifact is absent, say the current content "
                "was not provided and do not infer dates."
            )
        ),
        planner_guidance=(
            "Use when the user asks when products in a material pack are open, "
            "available, or purchasable."
        ),
        agent_guidance=(
            "Only state dates or availability found in material-pack open-calendar evidence."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            "When is this product open for subscription?",
            "Which products can be bought next week from the material pack?",
        ],
        examples_negative=[
            "Send the material pack.",
            "How did products perform in the monthly report?",
        ],
        runtime_capability="material_pack",
    ),
    CapabilityManifest(
        id="weekly_report.product_performance",
        version="2026-06-16.1",
        display_name="Weekly report product performance",
        description="Answer product performance questions grounded in weekly report evidence.",
        capability_type="answer",
        domain_entities=["channel", "strategy", "product", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[
            "canonical_context.selected_strategy",
            "plan.evidence_query",
        ],
        required_artifacts=["weekly_report"],
        allowed_artifacts=["weekly_report"],
        forbidden_artifacts=["material_pack", "monthly_report"],
        required_tools=["adapter_resolve.weekly_report", "adapter_report_scope"],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            any_of_fact_types=[
                "weekly_report_resolvable",
                "report_scope_summary",
                "report_scope_match",
                "report_scope_products",
                "report_period",
            ],
            allowed_source_types=["adapter_resolve", "adapter_report_scope"],
            forbidden_source_types=["document_mcp"],
            required_artifact_types=["weekly_report"],
            allowed_artifact_types=["weekly_report"],
            min_facts=1,
        ),
        abstention_policy=AbstentionPolicy(
            guidance=(
                "If weekly-report scope or period evidence is absent, abstain or "
                "ask for clarification instead of using document or material evidence."
            )
        ),
        planner_guidance=(
            "Use for product performance, period, or product-list questions about "
            "a weekly report."
        ),
        agent_guidance=(
            "Use only weekly-report resolve and report-scope facts. Do not fill gaps "
            "from material packs or internal document context."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            "Which products are in the weekly report?",
            "What period does this weekly report cover?",
        ],
        examples_negative=[
            "What products are in the material pack?",
            "Send the monthly report.",
        ],
        runtime_capability="weekly_report",
    ),
    CapabilityManifest(
        id="monthly_report.product_performance",
        version="2026-06-16.1",
        display_name="Monthly report product performance",
        description="Answer product performance questions grounded in monthly report evidence.",
        capability_type="answer",
        domain_entities=["channel", "strategy", "product", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[
            "canonical_context.selected_strategy",
            "plan.evidence_query",
        ],
        required_artifacts=["monthly_report"],
        allowed_artifacts=["monthly_report"],
        forbidden_artifacts=["material_pack", "weekly_report"],
        required_tools=["adapter_resolve.monthly_report", "adapter_report_scope"],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            any_of_fact_types=[
                "monthly_report_resolvable",
                "report_scope_summary",
                "report_scope_match",
                "report_scope_products",
                "report_period",
            ],
            allowed_source_types=["adapter_resolve", "adapter_report_scope"],
            forbidden_source_types=["document_mcp"],
            required_artifact_types=["monthly_report"],
            allowed_artifact_types=["monthly_report"],
            min_facts=1,
        ),
        abstention_policy=AbstentionPolicy(
            guidance=(
                "If monthly-report scope or period evidence is absent, abstain or "
                "ask for clarification instead of using weekly or material evidence."
            )
        ),
        planner_guidance=(
            "Use for product performance, period, or product-list questions about "
            "a monthly report."
        ),
        agent_guidance=(
            "Use only monthly-report resolve and report-scope facts. Do not fill gaps "
            "from weekly reports or material packs."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            "Which products are in the monthly report?",
            "What month does this report cover?",
        ],
        examples_negative=[
            "Send the weekly report.",
            "What is in the material pack?",
        ],
        runtime_capability="monthly_report",
    ),
    CapabilityManifest(
        id="channel.strategy_summary",
        version="2026-06-16.1",
        display_name="Channel strategy summary",
        description="Summarize strategy information for the current channel from approved document evidence.",
        capability_type="summary",
        domain_entities=["channel", "strategy", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[
            "canonical_context.selected_strategy",
            "request.available_strategies",
        ],
        required_artifacts=["document_context"],
        allowed_artifacts=["document_context"],
        forbidden_artifacts=["material_pack", "weekly_report", "monthly_report"],
        required_tools=["document_mcp.get_documents", "approved_static_knowledge.query"],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            any_of_fact_types=["document_context"],
            allowed_source_types=["document_mcp", "approved_static_knowledge"],
            forbidden_source_types=["adapter_report_scope"],
            required_artifact_types=["document_context"],
            allowed_artifact_types=["document_context"],
            min_facts=1,
        ),
        abstention_policy=AbstentionPolicy(
            guidance=(
                "If approved document evidence is absent, say there is not enough "
                "document evidence to summarize the strategy."
            )
        ),
        planner_guidance=(
            "Use for channel or strategy summary questions that require approved "
            "company/product document evidence."
        ),
        agent_guidance=(
            "Summarize only what appears in trusted document or approved static evidence."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            "Can you summarize this channel's strategy?",
            "How should I introduce this strategy?",
        ],
        examples_negative=[
            "Send the weekly report.",
            "Which products were generated in the report?",
        ],
        runtime_capability="document_context",
    ),
    CapabilityManifest(
        id="channel.product_summary",
        version="2026-06-16.1",
        display_name="Channel product summary",
        description="Summarize product information for the current channel from approved document evidence.",
        capability_type="summary",
        domain_entities=["channel", "product", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[
            "canonical_context.selected_strategy",
            "request.available_strategies",
        ],
        required_artifacts=["document_context"],
        allowed_artifacts=["document_context"],
        forbidden_artifacts=["material_pack", "weekly_report", "monthly_report"],
        required_tools=["document_mcp.get_documents", "approved_static_knowledge.query"],
        output_schema=_REPLY_OUTPUT_SCHEMA,
        evidence_contract=EvidenceContract(
            any_of_fact_types=["document_context"],
            allowed_source_types=["document_mcp", "approved_static_knowledge"],
            forbidden_source_types=["adapter_report_scope"],
            required_artifact_types=["document_context"],
            allowed_artifact_types=["document_context"],
            min_facts=1,
        ),
        abstention_policy=AbstentionPolicy(
            guidance=(
                "If approved document evidence is absent, say there is not enough "
                "document evidence to summarize products."
            )
        ),
        planner_guidance=(
            "Use for product-summary questions about the current channel when the "
            "answer must come from approved product/company documents."
        ),
        agent_guidance=(
            "Summarize only trusted document or approved static evidence and avoid "
            "using report-generation facts as product descriptions."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            "Can you summarize the products for this channel?",
            "What products can we introduce here?",
        ],
        examples_negative=[
            "Which products are in the weekly report?",
            "When is the material-pack product open?",
        ],
        runtime_capability="document_context",
    ),
)
