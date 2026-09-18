# Runtime Validation

Validation follows the active V2 boundary. Import the validator at its owning phase; do not add wrapper pipelines or compatibility shims.

```text
request_input_guard.py        raw /reply request boundary, before runtime work
locator_safety.py             model-visible locator and secret classification
reply_validator.py            typed public contract error boundary
reply_alignment_verifier.py   optional semantic verifier verdict schema
alignment*.py                 bounded V2 remediation and postconditions
```

`guardrail_common.py` contains the active media-marker parser. `guardrail_types.py` contains the V2 guardrail decision audit DTO.
