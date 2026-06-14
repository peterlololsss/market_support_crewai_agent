Compliance reason-code allowlist:
$compliance_policy_lines

For a non-compliant request, output artifact_kind=refusal, action_intent=refuse, requested_capabilities=[], ambiguity_slots=[], and compliance.is_compliant=false with the closest reason_code.

If compliance is uncertain and the request asks for a side effect, do not output a send intent.
