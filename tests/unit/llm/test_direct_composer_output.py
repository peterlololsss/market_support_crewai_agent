from __future__ import annotations

from typing import Literal

import pytest

from market_support_crewai_agent.runtime.llm.direct_composer_output import (
    DirectComposerOutput,
    DirectTargetDraft,
)
from market_support_crewai_agent.schemas import PrimaryReply


@pytest.mark.parametrize(
    "response_mode",
    ["clarify", "prepare_outbound_message"],
)
def test_partial_outbound_target_remains_structured_while_content_is_requested(
    response_mode: Literal["clarify", "prepare_outbound_message"],
) -> None:
    output = DirectComposerOutput(
        response_mode=response_mode,
        reply=PrimaryReply(
            kind="clarification",
            text="请问要发送什么内容？",
            mentions=[],
        ),
        target=DirectTargetDraft(kind="channel", name="兴业银行"),
    )

    assert output.target == DirectTargetDraft(kind="channel", name="兴业银行")
    assert output.content is None


def test_company_info_request_cannot_carry_outbound_fields() -> None:
    with pytest.raises(ValueError, match="direct_non_action_outbound_forbidden"):
        DirectComposerOutput(
            response_mode="request_company_info",
            reply=PrimaryReply(
                kind="unable_to_answer",
                text="需要查询内部资料。",
                mentions=[],
            ),
            target=DirectTargetDraft(kind="channel", name="兴业银行"),
        )
