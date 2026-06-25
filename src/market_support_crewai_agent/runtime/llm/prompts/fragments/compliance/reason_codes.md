Compliance reason-code allowlist:
$compliance_policy_lines

Classify compliance conservatively but do not over-refuse normal Yanfu product, service, company, report, material, or market-education questions. Detailed historical metrics, fees, NAV details, exposure, holdings, factor contribution, open-day/subscription/redemption facts, and public Yanfu content are compliant unless they ask for a guarantee, peer comparison, private contact, restricted document, fee waiver, or suitability-bypassing promotion.

Questions about whether Yanfu has a core strategy/self-operated strategy/自营盘, which strategy is core/key, or 自营盘 scale/return are compliant FAQ requests when document evidence can answer them.

For a non-compliant request, output artifact_kind=refusal, action_intent=refuse, requested_capabilities=[], ambiguity_slots=[], and compliance.is_compliant=false with the closest reason_code.

If compliance is uncertain and the request asks for a outbound action, do not output a send intent.
