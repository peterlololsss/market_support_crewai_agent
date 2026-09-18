from __future__ import annotations

from importlib.resources import files
import string
from string import Template
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from market_support_crewai_agent.runtime.prompts.profiles import (
    PromptScene,
    PromptStage,
    UserFacingPromptStage,
)
from market_support_crewai_agent.runtime.prompts.registry_catalog import (
    PROMPT_AGENT_SPECS as PROMPT_AGENT_SPECS,
    PROMPT_FRAGMENT_PACKAGE as PROMPT_FRAGMENT_PACKAGE,
    PROMPT_FRAGMENTS as PROMPT_FRAGMENTS,
    PROMPT_LAYER_ORDER as PROMPT_LAYER_ORDER,
    _DIRECT_RULES,
    _GROUP_RULES,
    _PROMPT_SCENES,
    _USER_FACING_STAGES,
)
from market_support_crewai_agent.runtime.prompts.registry_models import (
    PresentationField as PresentationField,
    PresentationRuleId as PresentationRuleId,
    PromptAgentSpec as PromptAgentSpec,
    PromptFragment as PromptFragment,
    PromptLayer as PromptLayer,
)


class PromptRegistryInvariantError(ValueError):
    def __init__(self, code: str) -> None:
        self.code: str = code
        super().__init__(code)


class ScenePresentationContractV1(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    contract_id: str
    version: Literal["2026-07-15.1"] = "2026-07-15.1"
    stage: UserFacingPromptStage
    scene: PromptScene
    audience: Literal["group", "individual"]
    allowed_presentation_fields: tuple[PresentationField, ...]
    presentation_rule_ids: tuple[PresentationRuleId, ...]
    fragment_id: str


SCENE_PRESENTATION_CONTRACTS: tuple[ScenePresentationContractV1, ...] = tuple(
    ScenePresentationContractV1(
        contract_id=f"scene.wecom_{scene}.{stage}.v1",
        stage=stage,
        scene=scene,
        audience="group" if scene == "group" else "individual",
        allowed_presentation_fields=(
            ("conversation_name", "principal_name")
            if scene == "group"
            else ("principal_name",)
        ),
        presentation_rule_ids=(_GROUP_RULES if scene == "group" else _DIRECT_RULES),
        fragment_id=f"scene.wecom_{scene}.{stage}.v1",
    )
    for stage in _USER_FACING_STAGES
    for scene in _PROMPT_SCENES
)


class PromptRegistry:
    def __init__(
        self,
        fragments: tuple[PromptFragment, ...] = PROMPT_FRAGMENTS,
        agent_specs: tuple[PromptAgentSpec, ...] = PROMPT_AGENT_SPECS,
        scene_contracts: tuple[
            ScenePresentationContractV1, ...
        ] = SCENE_PRESENTATION_CONTRACTS,
    ) -> None:
        self._fragments: tuple[PromptFragment, ...] = fragments
        self._agent_specs: dict[str, PromptAgentSpec] = {
            spec.id: spec for spec in agent_specs
        }
        self._scene_contracts: tuple[ScenePresentationContractV1, ...] = scene_contracts
        self._validate_scene_contracts()

    def fragment_by_id(
        self, fragment_id: str, stage: PromptStage | None = None
    ) -> PromptFragment:
        matches = [
            fragment
            for fragment in self._fragments
            if fragment.id == fragment_id and (stage is None or fragment.stage == stage)
        ]
        if not matches:
            raise PromptRegistryInvariantError("unknown_prompt_fragment")
        if len(matches) > 1 and stage is None:
            raise PromptRegistryInvariantError("prompt_fragment_stage_required")
        return matches[0]

    def scene_contract(
        self, stage: UserFacingPromptStage, scene: PromptScene
    ) -> ScenePresentationContractV1:
        matches = [
            contract
            for contract in self._scene_contracts
            if contract.stage == stage and contract.scene == scene
        ]
        if len(matches) != 1:
            raise PromptRegistryInvariantError("scene_contract_cardinality_invalid")
        return matches[0]

    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(fragment.id for fragment in self._fragments))

    def agent_spec_by_id(self, agent_id: str) -> PromptAgentSpec:
        try:
            return self._agent_specs[agent_id]
        except KeyError as exc:
            raise PromptRegistryInvariantError("unknown_prompt_agent_spec") from exc

    def agent_spec_ids(self) -> tuple[str, ...]:
        return tuple(self._agent_specs)

    def _validate_scene_contracts(self) -> None:
        if len(self._scene_contracts) != 8:
            raise PromptRegistryInvariantError("scene_contract_count_invalid")
        keys = {(contract.stage, contract.scene) for contract in self._scene_contracts}
        if len(keys) != 8:
            raise PromptRegistryInvariantError("scene_contract_key_duplicate")
        payloads: dict[tuple[UserFacingPromptStage, PromptScene], str] = {}
        for contract in self._scene_contracts:
            expected_id = f"scene.wecom_{contract.scene}.{contract.stage}.v1"
            if (
                contract.contract_id != expected_id
                or contract.fragment_id != expected_id
            ):
                raise PromptRegistryInvariantError("scene_contract_id_mismatch")
            if contract.scene == "group":
                expected_shape = (
                    "group",
                    ("conversation_name", "principal_name"),
                    _GROUP_RULES,
                )
            else:
                expected_shape = ("individual", ("principal_name",), _DIRECT_RULES)
            if (
                contract.audience,
                contract.allowed_presentation_fields,
                contract.presentation_rule_ids,
            ) != expected_shape:
                raise PromptRegistryInvariantError(
                    "scene_contract_presentation_mismatch"
                )
            fragment = self.fragment_by_id(contract.fragment_id, contract.stage)
            payload = load_prompt_fragment_text(fragment)
            payloads[(contract.stage, contract.scene)] = payload
            if not payload.endswith("\n"):
                raise PromptRegistryInvariantError("scene_fragment_final_lf_required")
            words = set(
                payload.casefold()
                .translate(str.maketrans({char: " " for char in string.punctuation}))
                .split()
            )
            forbidden_fields = {
                "source_type",
                "read_capabilities",
                "outbound_actions",
                "mention_types",
                "recall_mode",
                "validator_override",
            }
            normalized = payload.casefold()
            if words & {"capability", "tool", "grants"} or any(
                field in normalized for field in forbidden_fields
            ):
                raise PromptRegistryInvariantError("scene_fragment_authority_forbidden")
        stages: set[UserFacingPromptStage] = {
            contract.stage for contract in self._scene_contracts
        }
        for stage in stages:
            if payloads[(stage, "group")] == payloads[(stage, "direct")]:
                raise PromptRegistryInvariantError("scene_fragment_bytes_not_distinct")


def load_prompt_fragment_text(fragment: PromptFragment) -> str:
    if (
        fragment.template_name.startswith("/")
        or ".." in fragment.template_name.split("/")
        or not fragment.template_name.endswith(".md")
    ):
        raise PromptRegistryInvariantError("prompt_fragment_template_invalid")
    resource = files(PROMPT_FRAGMENT_PACKAGE).joinpath(fragment.template_name)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptRegistryInvariantError("prompt_fragment_template_missing") from exc


PROMPT_REGISTRY = PromptRegistry()


def fragment_by_id(
    fragment_id: str, stage: PromptStage | None = None
) -> PromptFragment:
    return PROMPT_REGISTRY.fragment_by_id(fragment_id, stage)


def scene_contract_by_stage(
    stage: UserFacingPromptStage, scene: PromptScene
) -> ScenePresentationContractV1:
    return PROMPT_REGISTRY.scene_contract(stage, scene)


def render_prompt_fragment(fragment_id: str, stage: PromptStage, **context: str) -> str:
    fragment = fragment_by_id(fragment_id, stage)
    return (
        Template(load_prompt_fragment_text(fragment)).safe_substitute(context).strip()
    )


def prompt_agent_spec_by_id(agent_id: str) -> PromptAgentSpec:
    return PROMPT_REGISTRY.agent_spec_by_id(agent_id)
