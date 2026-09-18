from __future__ import annotations

from json import dumps
from types import TracebackType
from typing import final

import pytest
from pydantic import JsonValue

from market_support_crewai_agent.runtime.identity import VerifiedRequestEnvelopeV1
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.prompts import invocation_journal
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply
from tests.helpers.planning import make_plan_spec


@final
class _DirectProviderResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def raise_for_status(self) -> None:
        return None

    @property
    def content(self) -> bytes:
        return dumps(
            {"choices": [{"message": {"content": self._text}}]},
            ensure_ascii=False,
        ).encode("utf-8")


def install_direct_provider_http_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    envelope: VerifiedRequestEnvelopeV1,
    requests: list[str],
    events: list[str],
    journals: list[TurnLlmInvocationJournalV1],
) -> None:
    import httpx

    outputs: dict[str, PlanSpec | ComposerReplyOutput | ReplyAlignmentVerdict] = {
        "PlanSpec": make_plan_spec(
            envelope.request,
            artifact_kind="smalltalk",
            action_intent="none",
            user_need="reply to a greeting",
        ),
        "ComposerReplyOutput": ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(kind="answer", text="你好，我在。", mentions=[]),
        ),
        "ReplyAlignmentVerdict": ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            confidence=1.0,
        ),
    }

    @final
    class FakeAsyncClient:
        def __init__(self, **kwargs: JsonValue) -> None:
            del kwargs

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exc_type, exc, traceback

        async def post(
            self,
            *args: str,
            **kwargs: JsonValue,
        ) -> _DirectProviderResponse:
            del args
            body = kwargs.get("json")
            assert isinstance(body, dict)
            response_format = body.get("response_format")
            assert isinstance(response_format, dict)
            json_schema = response_format.get("json_schema")
            assert isinstance(json_schema, dict)
            schema_name = json_schema.get("name")
            assert isinstance(schema_name, str)
            messages = body.get("messages")
            assert isinstance(messages, list) and messages
            first_message = messages[0]
            assert isinstance(first_message, dict)
            content = first_message.get("content")
            assert isinstance(content, str)
            journal = invocation_journal.current_turn_llm_invocation_journal()
            assert journal is not None
            reserved = journal.rows[-1]
            assert reserved.status == "reserved"
            requests.append(content)
            events.append(
                f"direct_dispatch:{reserved.stage_kind}:{reserved.program_id}"
            )
            journals.append(journal)
            output = outputs.get(schema_name)
            assert output is not None
            return _DirectProviderResponse(output.model_dump_json())

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
