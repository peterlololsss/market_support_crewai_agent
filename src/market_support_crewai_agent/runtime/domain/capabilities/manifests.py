from __future__ import annotations

from market_support_crewai_agent.runtime.domain.capabilities.registry import (
    AbstentionPolicy,
    CapabilityManifest,
    EvidenceContract,
    VerifierPrimitive,
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

_STANDARD_VERIFIER_CHECKS: list[VerifierPrimitive] = [
    "output_schema",
    "required_runtime_input_present",
    "required_evidence_present",
    "evidence_artifact_type_allowed",
    "forbidden_source_not_used",
    "abstention_correctness",
]
_OUTPUT_ONLY_VERIFIER_CHECKS: list[VerifierPrimitive] = ["output_schema"]

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
        description="Propose sending the adapter-resolved material pack for the current channel and optional material-pack option.",
        capability_type="action",
        domain_entities=["channel", "material_pack_option", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[
            "available_artifacts material_pack.options",
        ],
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
            "Use when the user clearly asks to send a material pack, one-pager, or open calendar. "
            "Also use for general product availability, distributed-products, "
            "available-products-list, or broad 'what products do you have' "
            "requests when the user does not explicitly ask about products inside "
            "a weekly or monthly report; product availability belongs to the "
            "material-pack artifact. "
            "Also use for strategy intro, product highlight, open calendar, or "
            "performance material only when the wording clearly asks to send/provide "
            "an artifact and no more specific supported action exists. Also use for "
            "broad historical performance-summary requests where a material pack is "
            "the supported collateral instead of a weekly report. "
            "material_pack.options are material-pack scope labels, not a general "
            "strategy catalog. If the user explicitly targets one material_pack.options "
            "value, copy that exact value into domain_scope.material_pack_option for adapter "
            "resolve. "
            "If material_pack.options has explicit values and the user did not select "
            "one, ask a material_pack_option clarification before sending. Empty "
            "options means an unscoped material-pack send is valid."
        ),
        agent_guidance="Return only the typed send_material_pack action after adapter resolve evidence.",
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic clear material-pack send request.',
            'Synthetic collateral one-pager send request.',
            'Synthetic product-availability collateral request.',
        ],
        examples_negative=[
            'Synthetic report-content question.',
            'Synthetic document-backed strategy definition question.',
        ],
        runtime_capability="material_pack",
    ),
    CapabilityManifest(
        id="weekly_report.send",
        version="2026-06-16.1",
        display_name="Send weekly report",
        description="Propose sending the adapter-resolved weekly report for the current channel.",
        capability_type="action",
        domain_entities=["channel", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[],
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
            guidance="If the weekly report cannot be resolved, abstain or hand off."
        ),
        planner_guidance=(
            "Use when the user clearly asks to send the weekly report. Treat the "
            "weekly report as this week's valuation/performance artifact, not as "
            "the owner of every FAQ containing performance words. Also use "
            "for named public product/strategy current performance, drawdown, NAV, "
            "or return metric follow-up, "
            "and when the user clearly asks to send or provide an official "
            "performance report/material and weekly_report is the closest "
            "supported action. "
            "Do not answer unsupported numbers directly; send the weekly report "
            "unless the request is compliance-blocked. Do not use for evergreen "
            "document FAQ facts such as factor/excess-return contribution mix, "
            "收益来源占比, T0 mechanics, fee/到手收益 explanations, holdings, "
            "exposure, or strategy-explanation why questions. For performance-report sends, "
            "add risk_flags=[\"weekly_report_rationale_required\"]."
        ),
        agent_guidance="Return only the typed send_weekly_report action after adapter resolve evidence.",
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic weekly report send request.',
            'Synthetic current product metric request requiring report artifact.',
            'Synthetic official performance material send request.',
        ],
        examples_negative=[
            'Synthetic evergreen performance explanation question.',
            'Synthetic report product-list question.',
        ],
        runtime_capability="weekly_report",
    ),
    CapabilityManifest(
        id="monthly_report.send",
        version="2026-06-16.1",
        display_name="Send monthly report",
        description="Propose sending the adapter-resolved monthly report for the current channel.",
        capability_type="action",
        domain_entities=["channel", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[],
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
            guidance="If the monthly report cannot be resolved, abstain or hand off."
        ),
        planner_guidance=(
            "Use when the user clearly asks to send the monthly report."
        ),
        agent_guidance="Return only the typed send_monthly_report action after adapter resolve evidence.",
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic monthly report send request.',
        ],
        examples_negative=[
            'Synthetic weekly-report period question.',
        ],
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
        planner_guidance="Use when the user asks for human sales/support help, add-friend/add-WeChat/private-chat routing, or a named person.",
        agent_guidance="Mention only adapter-resolved sales/support targets.",
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic request to contact sales.',
            'Synthetic named human routing request.',
        ],
        examples_negative=[
            'Synthetic artifact send request.',
        ],
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
        planner_guidance=(
            "Use only when user wording leaves the requested artifact type or "
            "material-pack option ambiguous. Prefer no outbound action unless the user "
            "clearly asks to send."
        ),
        agent_guidance="Ask one concise clarification and do not propose actions.",
        verifier_checks=_OUTPUT_ONLY_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic missing material-pack option clarification.',
        ],
        examples_negative=[
            'Synthetic clear report send request.',
        ],
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
        examples_positive=[
            'Synthetic unable-to-confirm response when required evidence is absent.',
        ],
        examples_negative=[
            'Synthetic completed-send status.',
        ],
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
        examples_positive=[
            'Synthetic guaranteed-return request.',
        ],
        examples_negative=[
            'Synthetic ordinary report send request.',
        ],
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
        examples_positive=[
            'Synthetic assistant identity question.',
            'Synthetic thanks message.',
        ],
        examples_negative=[
            'Synthetic business artifact request.',
        ],
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
        examples_positive=[
            'Synthetic adapter-only status message that should stay silent.',
        ],
        examples_negative=[
            'Synthetic user help request.',
        ],
    ),
    CapabilityManifest(
        id="material_pack.open_calendar",
        version="2026-06-16.1",
        display_name="Material pack open calendar",
        description="Answer product opening or subscription calendar questions from a material pack.",
        capability_type="answer",
        domain_entities=["channel", "material_pack_option", "product", "artifact"],
        required_inputs=["request.dist_channel_name"],
        optional_inputs=[
            "available_artifacts material_pack.options",
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
            'Synthetic product open-calendar question about material-pack content.',
        ],
        examples_negative=[
            'Synthetic material-pack send request.',
            'Synthetic monthly-report performance question.',
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
                "If weekly-report scope or period evidence is absent, abstain "
                "instead of using document or material evidence."
            )
        ),
        planner_guidance=(
            "Use for product performance, period, or product-list questions about "
            "a weekly report only when the user explicitly mentions the weekly "
            "report/report scope. Do not use for general distributed-products or "
            "available-products-list requests; those belong to material_pack.send "
            "when allowed. For report product-list or shorthand product-presence "
            "questions, set evidence_query exactly to report_scope_products so "
            "the bounded report product list is fetched. Do not use for "
            "strategy/company scale, capacity, "
            "headcount, founding date, holdings profile, factors, product intro, "
            "or other evergreen document facts unless the user explicitly asks "
            "about a weekly report."
        ),
        agent_guidance=(
            "Use only weekly-report resolve and report-scope facts. Do not fill gaps "
            "from material packs or internal document context."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic weekly-report product-list question.',
            'Synthetic weekly-report period question.',
        ],
        examples_negative=[
            'Synthetic material-pack product question.',
            'Synthetic monthly-report send request.',
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
                "If monthly-report scope or period evidence is absent, abstain "
                "instead of using weekly or material evidence."
            )
        ),
        planner_guidance=(
            "Use for product performance, period, or product-list questions about "
            "a monthly report only when the user explicitly mentions the monthly "
            "report/report scope. Do not use for general distributed-products or "
            "available-products-list requests; those belong to material_pack.send "
            "when allowed. For report product-list or shorthand product-presence "
            "questions, set evidence_query exactly to report_scope_products so "
            "the bounded report product list is fetched. Do not use for "
            "strategy/company scale, capacity, "
            "headcount, founding date, holdings profile, factors, product intro, "
            "or other evergreen document facts unless the user explicitly asks "
            "about a monthly report."
        ),
        agent_guidance=(
            "Use only monthly-report resolve and report-scope facts. Do not fill gaps "
            "from weekly reports or material packs."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic monthly-report product-list question.',
            'Synthetic monthly-report period question.',
        ],
        examples_negative=[
            'Synthetic weekly-report send request.',
            'Synthetic material-pack content question.',
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
            "domain_context.strategies",
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
            "company/product document evidence, including scale/AUM, capacity, "
            "headcount, founding date, holdings profile, factor lineup, strategy "
            "intro, product differences, exposure, excess-return sources, redemption, "
            "subscription, dividends, NAV disclosure timing, hedging cost, basis, "
            "index valuation/position facts, company-viewpoint/衍复观点 questions, "
            "fee/到手收益 explanations, T0 strategy mechanics, document-backed "
            "why/explanation questions, and FAQ style operating questions."
        ),
        agent_guidance=(
            "Summarize only what appears in trusted document or approved static evidence."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic channel strategy summary question.',
            'Synthetic company or strategy FAQ question.',
            'Synthetic document-backed operating-rule question.',
        ],
        examples_negative=[
            'Synthetic weekly-report send request.',
            'Synthetic report-generation artifact question.',
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
            "domain_context.strategies",
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
            "answer must come from approved product/company documents. Do not use "
            "for general distributed-products or available-products-list requests; "
            "those belong to material_pack.send when allowed."
        ),
        agent_guidance=(
            "Summarize only trusted document or approved static evidence and avoid "
            "using report-generation facts as product descriptions."
        ),
        verifier_checks=_STANDARD_VERIFIER_CHECKS,
        examples_positive=[
            'Synthetic channel product summary question.',
            'Synthetic product document FAQ question.',
        ],
        examples_negative=[
            'Synthetic weekly-report product-list question.',
            'Synthetic material-pack open-calendar question.',
        ],
        runtime_capability="document_context",
    ),
)
