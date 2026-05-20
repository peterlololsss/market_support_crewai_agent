from __future__ import annotations

import json

from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse
from market_support_crewai_agent.settings import Settings, get_settings


class AgentRuntimeError(RuntimeError):
    """Raised when the CrewAI runtime cannot produce a valid reply."""


async def build_reply(
    request: ReplyRequest,
    settings: Settings | None = None,
) -> ReplyResponse:
    runtime = CrewAIReplyRuntime(settings or get_settings())
    return await runtime.reply(request)


class CrewAIReplyRuntime:
    """CrewAI runtime boundary used by the FastAPI transport layer."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def reply(self, request: ReplyRequest) -> ReplyResponse:
        if not self.settings.llm_api_key:
            raise AgentRuntimeError("YANFU_LLM_API_KEY is not configured")

        agent = self._build_agent()
        try:
            result = await agent.kickoff_async(
                _runtime_prompt(request),
                response_format=ReplyResponse,
            )
        except Exception as exc:
            raise AgentRuntimeError("CrewAI runtime failed") from exc

        if result.pydantic is not None:
            return ReplyResponse.model_validate(result.pydantic)

        try:
            return ReplyResponse.model_validate_json(result.raw)
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


def _runtime_prompt(request: ReplyRequest) -> str:
    request_json = json.dumps(request.model_dump(mode="json"), ensure_ascii=False)
    return (
        "You are handling a /reply request for a WeWork adapter. "
        "Return a ReplyResponse matching the provided response_format. "
        "The response text is for the user; actions are typed instructions for "
        "the adapter to execute later. Do not execute actions yourself. "
        "Use the request context as the source of truth.\n\n"
        f"Request JSON:\n{request_json}"
    )

