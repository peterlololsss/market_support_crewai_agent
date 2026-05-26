from __future__ import annotations

import json

from market_support_crewai_agent.runtime.conversation_store import (
    ConversationMessage,
    ConversationStore,
)
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse
from market_support_crewai_agent.settings import Settings, get_settings


class AgentRuntimeError(RuntimeError):
    """Raised when the CrewAI runtime cannot produce a valid reply."""


_DEFAULT_SETTINGS = get_settings()
_DEFAULT_CONVERSATION_STORE = ConversationStore.from_settings(_DEFAULT_SETTINGS)


async def build_reply(
    request: ReplyRequest,
    settings: Settings | None = None,
    conversation_store: ConversationStore | None = None,
) -> ReplyResponse:
    runtime = CrewAIReplyRuntime(
        settings or _DEFAULT_SETTINGS,
        conversation_store or _DEFAULT_CONVERSATION_STORE,
    )
    return await runtime.reply(request)


class CrewAIReplyRuntime:
    """CrewAI runtime boundary used by the FastAPI transport layer."""

    def __init__(
        self,
        settings: Settings,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self.settings = settings
        self.conversation_store = conversation_store or ConversationStore.from_settings(
            settings
        )

    async def reply(self, request: ReplyRequest) -> ReplyResponse:
        if not self.settings.llm_api_key:
            raise AgentRuntimeError("YANFU_LLM_API_KEY is not configured")

        agent = self._build_agent()
        history = self.conversation_store.get_recent(request.conversation_key)
        try:
            result = await agent.kickoff_async(
                _runtime_prompt(request, history),
                response_format=ReplyResponse,
            )
        except Exception as exc:
            raise AgentRuntimeError("CrewAI runtime failed") from exc

        if result.pydantic is not None:
            response = ReplyResponse.model_validate(result.pydantic)
            self.conversation_store.save_turn(
                request.conversation_key,
                request.message,
                _compact_assistant_result(response),
            )
            return response

        try:
            response = ReplyResponse.model_validate_json(result.raw)
            self.conversation_store.save_turn(
                request.conversation_key,
                request.message,
                _compact_assistant_result(response),
            )
            return response
        except ValueError as exc:
            raise AgentRuntimeError(
                "CrewAI runtime returned an invalid reply contract"
            ) from exc

    def _build_agent(self):
        from crewai import Agent, LLM

        return Agent(
            role="Market Support Reply Decision Agent",
            goal=(
                "Decide the natural-language reply and typed actions for the "
                "external WeWork adapter."
            ),
            backstory=(
                "You are the external agent brain for a market support workflow. "
                "You reason over the request context and return structured output. "
                "You never send WeWork messages directly."
            ),
            llm=LLM(
                model=self.settings.llm_model,
                provider=self.settings.llm_provider,
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
            ),
            allow_delegation=False,
            verbose=self.settings.crewai_verbose,
            max_iter=self.settings.crewai_max_iter,
            max_execution_time=self.settings.crewai_max_execution_time,
        )


def _runtime_prompt(
    request: ReplyRequest,
    history: list[ConversationMessage] | None = None,
) -> str:
    metadata = request.model_dump(
        mode="json",
        exclude={"message"},
        exclude_none=True,
    )
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    history_json = json.dumps(
        [
            {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            for message in history or []
        ],
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are handling a /reply request for a WeWork adapter. "
        "Return a ReplyResponse matching the provided response_format. "
        "The response text is for the user; actions are typed instructions for "
        "the adapter to execute later. Do not execute actions yourself. "
        "Use the request context as the source of truth.\n\n"
        "Conversation rules:\n"
        "- CrewAI has no durable application memory in this service; use only "
        "the recent_turns JSON supplied here for chat history.\n"
        "- Do not rely on trigger fields such as session_id, bot_mentioned, or "
        "trigger_reason; the gateway owns trigger detection.\n"
        "- Treat conversation_key as the isolation boundary for this history.\n\n"
        f"Request metadata JSON:\n{metadata_json}\n\n"
        f"Recent turns JSON:\n{history_json}\n\n"
        f"Current user message:\n{request.message}"
    )


def _compact_assistant_result(response: ReplyResponse) -> str:
    return json.dumps(
        response.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
