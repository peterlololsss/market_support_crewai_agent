from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import singledispatch
from typing import TypeVar

import anyio
from pydantic import JsonValue

from market_support_crewai_agent.runtime.hashing import CanonicalValue
from market_support_crewai_agent.runtime.prompts.assembler import PromptProgram
from market_support_crewai_agent.runtime.prompts.profiles import prompt_profile_by_stage
from market_support_crewai_agent.runtime.prompts.program_models import (
    resolve_active_prompt_program_v2,
)

ResultT = TypeVar("ResultT")


def planner_program() -> PromptProgram:
    _, execution_spec = resolve_active_prompt_program_v2(
        stage="planner_intent",
        scene_key="wecom_group.v1",
    )
    return PromptProgram(
        profile=prompt_profile_by_stage("planner_intent"),
        program_id="planner_intent.wecom_group.v1@1",
        program_version="1",
        agent_execution_spec=execution_spec,
        scene_key="wecom_group.v1",
        scene_contract_id="scene.wecom_group.planner_intent.v1",
        scene_contract_version="2026-07-15.1",
        fragment_ids=(),
        prompt_text="{}",
        prompt_hash="hash",
        fragment_hashes={},
        layers=(),
        static_bytes=2,
        baseline_bytes=2,
        allowed_max_bytes=1026,
        mch1="mch1:" + "0" * 64,
    )


@singledispatch
def canonical_value(value: JsonValue) -> CanonicalValue:
    del value
    raise AssertionError("canonical_value_type_unsupported")


@canonical_value.register
def canonical_string(value: str) -> CanonicalValue:
    return value


@canonical_value.register
def canonical_integer(value: int) -> CanonicalValue:
    return value


@canonical_value.register
def canonical_none(value: None) -> CanonicalValue:
    return value


@canonical_value.register
def canonical_float(value: float) -> CanonicalValue:
    del value
    raise AssertionError("canonical_float_forbidden")


@canonical_value.register(list)
def canonical_list(value: list[JsonValue]) -> CanonicalValue:
    return [canonical_value(item) for item in value]


@canonical_value.register(dict)
def canonical_mapping(value: dict[str, JsonValue]) -> CanonicalValue:
    return {key: canonical_value(item) for key, item in value.items()}


def run_async(operation: Callable[[], Awaitable[ResultT]]) -> ResultT:
    return anyio.run(operation)
