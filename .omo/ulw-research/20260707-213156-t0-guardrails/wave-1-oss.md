# Wave 1 OSS Axis

Sources:
- https://docs.langchain.com/oss/python/langchain/middleware
- https://docs.nvidia.com/nemo/guardrails/
- https://github.com/NousResearch/hermes-agent/blob/009b42d008b81c18af39414dded9ecdf06082d93/agent/tool_guardrails.py
- https://github.com/microsoft/semantic-kernel/blob/35ba23e1b3092271c778ca057afe1a796e16e70e/python/semantic_kernel/filters/kernel_filters_extension.py

Key findings:
- Modern frameworks use middleware/filter/rail registries rather than one class per rule.
- Hermes uses a pure side-effect-free controller that returns decisions; runtime decides how to act on them.
- The closest fit for this repo is a small data-driven input rule table at the existing input-policy seam.

EXPAND:
- None needed for current T0 fix.
