from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel, JsonValue

from market_support_crewai_agent.runtime.identity import VerifiedRequestEnvelopeV1
from market_support_crewai_agent.runtime.integrations.crewai.contracts import (
    CrewAIAgentAdapterV1,
    CrewAICompletionValueV1,
    CrewAIKickoffOutputV1,
)
from market_support_crewai_agent.runtime.planning.plan_spec import PlanSpec
from market_support_crewai_agent.runtime.prompts import (
    invocation_journal,
    program_models,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    TurnLlmInvocationJournalV1,
)
from market_support_crewai_agent.runtime.recall import approved_static_selector
from market_support_crewai_agent.runtime.rendering.composer_output import (
    ComposerReplyOutput,
)
from market_support_crewai_agent.runtime.service import CrewAIReplyRuntime
from market_support_crewai_agent.runtime.state.audit_records import (
    DirectAuditRecordV1,
    GroupAuditRecordV1,
)
from market_support_crewai_agent.runtime.state.transaction_coordinator import (
    ReplyStateTransactionCoordinatorV1,
)
from market_support_crewai_agent.runtime.validation.reply_alignment_verifier import (
    ReplyAlignmentVerdict,
)
from market_support_crewai_agent.schemas.reply import PrimaryReply
from market_support_crewai_agent.settings_model import Settings
from tests.helpers.crewai_adapter import (
    invoke_completion,
    make_agent_adapter,
    make_llm_adapter,
)
from tests.helpers.planning import make_plan_spec


def _provider_agent(
    *,
    stage: str,
    output: BaseModel,
    events: list[str],
    journals: list[TurnLlmInvocationJournalV1],
) -> CrewAIAgentAdapterV1:
    def complete(
        *,
        params: Mapping[str, JsonValue],
        available_functions: JsonValue | None = None,
        from_task: JsonValue | None = None,
        from_agent: JsonValue | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> CrewAICompletionValueV1:
        del params, available_functions, from_task, from_agent
        assert response_model is type(output)
        return output

    llm = make_llm_adapter(
        provider="openai",
        model=f"fake-{stage}",
        base_url="https://provider.invalid/v1",
        api_key="fake-key",
        timeout=5.0,
        temperature=0.0,
        max_tokens=1200,
        completion=complete,
    )

    def kickoff(
        prompt: str,
        response_format: type[BaseModel],
    ) -> CrewAIKickoffOutputV1:
        completion_value = invoke_completion(
            llm,
            params={
                "model": llm.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            response_model=response_format,
        )
        assert isinstance(completion_value, BaseModel)
        journal = invocation_journal.current_turn_llm_invocation_journal()
        assert journal is not None, "provider dispatch has no turn journal"
        reserved = journal.rows[-1]
        assert reserved.status == "reserved"
        assert response_format is type(output)
        events.append(f"dispatch:{stage}:{reserved.program_id}")
        journals.append(journal)
        return CrewAIKickoffOutputV1(
            raw=completion_value.model_dump_json(),
            pydantic=completion_value,
        )

    return make_agent_adapter(
        role=stage,
        llm=llm,
        on_prompt=kickoff,
    )


@dataclass(frozen=True, slots=True)
class ProviderPlannerFactory:
    planner: CrewAIAgentAdapterV1

    def build_planner_agent(self) -> CrewAIAgentAdapterV1:
        return self.planner


@dataclass(frozen=True, slots=True)
class ProviderComposerFactory:
    composer: CrewAIAgentAdapterV1

    def build_composer_agent(
        self,
        stage: Literal["knowledge_composer", "smalltalk_composer"],
    ) -> CrewAIAgentAdapterV1:
        del stage
        return self.composer


@dataclass(frozen=True, slots=True)
class ProviderAlignmentFactory:
    verifier: CrewAIAgentAdapterV1

    def build_alignment_verifier_agent(self) -> CrewAIAgentAdapterV1:
        return self.verifier


def install_registry_spy(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    load_programs = program_models.load_prompt_program_registry_v2
    load_specs = program_models.load_agent_execution_specs_v1

    def programs():
        rows = load_programs()
        events.append("registry:" + ",".join(row.program_id for row in rows))
        return rows

    def specs():
        rows = load_specs()
        events.append("specs:" + ",".join(row.program_id for row in rows))
        return rows

    monkeypatch.setattr(program_models, "load_prompt_program_registry_v2", programs)
    monkeypatch.setattr(program_models, "load_agent_execution_specs_v1", specs)


def runtime_with_provider_fakes(
    *,
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    envelope: VerifiedRequestEnvelopeV1,
    events: list[str],
    journals: list[TurnLlmInvocationJournalV1],
) -> CrewAIReplyRuntime:
    install_runtime_settings(monkeypatch, settings)
    runtime = CrewAIReplyRuntime(
        settings,
        coordinator=ReplyStateTransactionCoordinatorV1(),
    )
    planner = _provider_agent(
        stage="planner_intent",
        output=smalltalk_plan(envelope),
        events=events,
        journals=journals,
    )
    composer = _provider_agent(
        stage="smalltalk_composer",
        output=ComposerReplyOutput(
            response_mode="answer",
            reply=PrimaryReply(kind="answer", text="你好，我在。", mentions=[]),
        ),
        events=events,
        journals=journals,
    )
    verifier = _provider_agent(
        stage="alignment_verifier",
        output=ReplyAlignmentVerdict(
            aligned=True,
            safe_to_return=True,
            confidence=1.0,
        ),
        events=events,
        journals=journals,
    )
    runtime.planner_agent_factory = ProviderPlannerFactory(planner)
    runtime.composer_agent_factory = ProviderComposerFactory(composer)
    runtime.alignment_agent_factory = ProviderAlignmentFactory(verifier)
    return runtime


def install_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    monkeypatch.setattr(approved_static_selector, "get_settings", lambda: settings)


def smalltalk_plan(envelope: VerifiedRequestEnvelopeV1) -> PlanSpec:
    return make_plan_spec(
        envelope.request,
        artifact_kind="smalltalk",
        action_intent="none",
        user_need="reply to a greeting",
    )


def assert_reachability(
    *,
    events: list[str],
    journals: list[TurnLlmInvocationJournalV1],
    scene: str,
    record: GroupAuditRecordV1 | DirectAuditRecordV1,
) -> None:
    expected_ids = (
        f"planner_intent.wecom_{scene}.v1@1",
        f"smalltalk_composer.wecom_{scene}.v1@1",
        f"alignment_verifier.wecom_{scene}.v1@1",
    )
    dispatch_prefix = "dispatch:" if scene == "group" else "direct_dispatch:"
    dispatches = tuple(event for event in events if event.startswith(dispatch_prefix))
    assert tuple(event.rsplit(":", 1)[-1] for event in dispatches) == expected_ids
    assert journals and all(journal is journals[0] for journal in journals)
    rows = journals[0].rows
    assert tuple(row.invocation_ordinal for row in rows) == (1, 2, 3)
    assert tuple(row.logical_attempt for row in rows) == (1, 2, 3)
    assert tuple(row.program_id for row in rows) == expected_ids
    assert tuple(row.status for row in rows) == ("success", "success", "success")
    assert tuple(program.program_id for program in record.programs) == expected_ids
    assert tuple(program.ordinal for program in record.programs) == (1, 2, 3)
    assert any(event.startswith("registry:") for event in events)
    assert any(event.startswith("specs:") for event in events)
    for dispatch, program_id in zip(dispatches, expected_ids, strict=True):
        dispatch_index = events.index(dispatch)
        assert any(
            program_id in event
            for event in events[:dispatch_index]
            if event.startswith(("registry:", "specs:"))
        )
