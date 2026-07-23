from __future__ import annotations

from market_support_crewai_agent.runtime.context.pending import (
    PendingOutboundConfirmation,
)
from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
)


def pending_confirmation_resolution_issue(
    output: DirectComposerOutput,
    pending: PendingOutboundConfirmation | None,
) -> str | None:
    if pending is None:
        return None

    resolution = output.pending_confirmation_resolution
    mode = output.response_mode
    if resolution == "not_applicable":
        return "pending_confirmation_resolution_required"
    if resolution == "confirm" and mode != "execute_prepared_outbound_message":
        return "pending_confirmation_resolution_mode_mismatch"
    if resolution == "correct" and mode != "prepare_outbound_message":
        return "pending_confirmation_resolution_mode_mismatch"
    if resolution == "cancel" and mode != "smalltalk":
        return "pending_confirmation_resolution_mode_mismatch"
    if resolution == "ambiguous" and mode != "clarify":
        return "pending_confirmation_resolution_mode_mismatch"
    if resolution == "topic_switch" and mode == "execute_prepared_outbound_message":
        return "pending_confirmation_resolution_mode_mismatch"
    return None
