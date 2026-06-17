from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from market_support_crewai_agent.schemas import StrictModel

CapabilityType = Literal["action", "answer", "summary", "handoff"]
VerifierPrimitive = Literal[
    "output_schema",
    "required_evidence_present",
    "evidence_artifact_type_allowed",
    "required_runtime_input_present",
    "forbidden_source_not_used",
    "abstention_correctness",
]


class EvidenceContract(StrictModel):
    required_evidence_types: list[str] = Field(default_factory=list)
    required_fact_types: list[str] = Field(default_factory=list)
    any_of_fact_types: list[str] = Field(default_factory=list)
    allowed_source_types: list[str] = Field(default_factory=list)
    disallowed_source_types: list[str] = Field(default_factory=list)
    forbidden_source_types: list[str] = Field(default_factory=list)
    required_artifact_types: list[str] = Field(default_factory=list)
    allowed_artifact_types: list[str] = Field(default_factory=list)
    required_scope_match: list[str] = Field(default_factory=list)
    minimum_evidence_count: int = Field(default=0, ge=0)
    min_facts: int = Field(default=0, ge=0)
    allow_history: bool = False
    history_constraints: dict[str, Any] = Field(default_factory=dict)
    fallback_policy: str = "abstain"
    citation_required: bool = False
    provenance_required: bool = True
    citation_requirements: list[str] = Field(default_factory=list)
    stale_data_policy: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_equivalent_field_names(cls, value):
        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        if (
            "required_evidence_types" not in payload
            and "required_fact_types" in payload
        ):
            payload["required_evidence_types"] = payload["required_fact_types"]
        if (
            "required_fact_types" not in payload
            and "required_evidence_types" in payload
        ):
            payload["required_fact_types"] = payload["required_evidence_types"]
        if (
            "disallowed_source_types" not in payload
            and "forbidden_source_types" in payload
        ):
            payload["disallowed_source_types"] = payload["forbidden_source_types"]
        if (
            "forbidden_source_types" not in payload
            and "disallowed_source_types" in payload
        ):
            payload["forbidden_source_types"] = payload["disallowed_source_types"]
        if "minimum_evidence_count" not in payload and "min_facts" in payload:
            payload["minimum_evidence_count"] = payload["min_facts"]
        if "min_facts" not in payload and "minimum_evidence_count" in payload:
            payload["min_facts"] = payload["minimum_evidence_count"]
        return payload


class AbstentionPolicy(StrictModel):
    requires_abstention_when_evidence_missing: bool = True
    abstention_reply_kinds: list[str] = Field(
        default_factory=lambda: ["unable_to_answer", "clarification"]
    )
    guidance: str = ""


class CapabilityManifest(StrictModel):
    id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$",
    )
    version: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    capability_type: CapabilityType
    domain_entities: list[str]
    required_inputs: list[str]
    optional_inputs: list[str]
    required_artifacts: list[str]
    allowed_artifacts: list[str]
    forbidden_artifacts: list[str]
    required_tools: list[str]
    output_schema: dict[str, Any]
    evidence_contract: EvidenceContract
    abstention_policy: AbstentionPolicy
    planner_guidance: str = Field(min_length=1)
    agent_guidance: str = Field(min_length=1)
    verifier_checks: list[VerifierPrimitive]
    examples_positive: list[str]
    examples_negative: list[str]
    runtime_capability: str | None = None

    @model_validator(mode="after")
    def validate_manifest_shape(self):
        if not self.domain_entities:
            raise ValueError("domain_entities must not be empty")
        if not self.verifier_checks:
            raise ValueError("verifier_checks must not be empty")
        if "output_schema" in self.verifier_checks and not self.output_schema:
            raise ValueError("output_schema check requires output_schema")
        required = set(self.required_artifacts)
        allowed = set(self.allowed_artifacts)
        forbidden = set(self.forbidden_artifacts)
        if required & forbidden:
            raise ValueError("required_artifacts cannot also be forbidden")
        if allowed & forbidden:
            raise ValueError("allowed_artifacts cannot also be forbidden")
        if not required <= allowed:
            raise ValueError("required_artifacts must be included in allowed_artifacts")
        if "required_evidence_present" in self.verifier_checks:
            evidence = self.evidence_contract
            if (
                not evidence.required_evidence_types
                and not evidence.required_fact_types
                and not evidence.any_of_fact_types
                and evidence.minimum_evidence_count <= 0
                and evidence.min_facts <= 0
            ):
                raise ValueError(
                    "required_evidence_present requires required facts, any-of facts, or min_facts"
                )
        return self

    def to_planner_card(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": self.version,
            "display_name": self.display_name,
            "description": self.description,
            "capability_type": self.capability_type,
            "domain_entities": list(self.domain_entities),
            "required_inputs": list(self.required_inputs),
            "optional_inputs": list(self.optional_inputs),
            "required_artifacts": list(self.required_artifacts),
            "allowed_artifacts": list(self.allowed_artifacts),
            "forbidden_artifacts": list(self.forbidden_artifacts),
            "required_tools": list(self.required_tools),
            "evidence": {
                "required_fact_types": list(self.evidence_contract.required_fact_types),
                "required_evidence_types": list(
                    self.evidence_contract.required_evidence_types
                ),
                "any_of_fact_types": list(self.evidence_contract.any_of_fact_types),
                "forbidden_source_types": list(
                    self.evidence_contract.forbidden_source_types
                ),
                "disallowed_source_types": list(
                    self.evidence_contract.disallowed_source_types
                ),
                "allowed_artifact_types": list(
                    self.evidence_contract.allowed_artifact_types
                ),
                "required_scope_match": list(
                    self.evidence_contract.required_scope_match
                ),
                "minimum_evidence_count": (
                    self.evidence_contract.minimum_evidence_count
                ),
                "allow_history": self.evidence_contract.allow_history,
            },
            "abstain_when_missing_evidence": (
                self.abstention_policy.requires_abstention_when_evidence_missing
            ),
            "abstention_guidance": self.abstention_policy.guidance,
            "planner_guidance": self.planner_guidance,
            "examples_positive": list(self.examples_positive),
            "examples_negative": list(self.examples_negative),
            "verifier_checks": list(self.verifier_checks),
            "runtime_capability": self.runtime_capability,
        }

    def to_verifier_contract(self) -> dict[str, object]:
        return {
            "id": self.id,
            "version": self.version,
            "output_schema": self.output_schema,
            "required_inputs": list(self.required_inputs),
            "required_artifacts": list(self.required_artifacts),
            "allowed_artifacts": list(self.allowed_artifacts),
            "forbidden_artifacts": list(self.forbidden_artifacts),
            "evidence_contract": self.evidence_contract.model_dump(mode="json"),
            "abstention_policy": self.abstention_policy.model_dump(mode="json"),
            "verifier_checks": list(self.verifier_checks),
        }


class CandidateCapabilityResolver(Protocol):
    def resolve(
        self,
        registry: CapabilityRegistry,
        userRequest: object | None,
        runtimeContext: Mapping[str, object] | object | None,
    ) -> tuple[CapabilityManifest, ...]: ...


class RuntimeContextCapabilityResolver:
    """Resolve manifest candidates from explicit runtime context only.

    This resolver intentionally does not inspect message text or maintain
    semantic keyword lists.
    """

    def resolve(
        self,
        registry: CapabilityRegistry,
        userRequest: object | None,
        runtimeContext: Mapping[str, object] | object | None,
    ) -> tuple[CapabilityManifest, ...]:
        del userRequest
        allowed_ids = _context_values(runtimeContext, "allowed_capability_ids")
        if allowed_ids:
            return _unique(
                registry.get(capability_id)
                for capability_id in allowed_ids
                if registry.find(capability_id) is not None
            )

        requested_ids = _context_values(runtimeContext, "requested_capability_ids")
        if requested_ids:
            return _unique(
                registry.get(capability_id)
                for capability_id in requested_ids
                if registry.find(capability_id) is not None
            )

        allowed_runtime = _context_values(
            runtimeContext,
            "policy_allowed_capabilities",
            "allowed_capabilities",
        )
        if allowed_runtime:
            allowed = set(allowed_runtime)
            return tuple(
                manifest
                for manifest in registry.list()
                if manifest.runtime_capability in allowed
                or manifest.runtime_capability is None
            )

        return registry.list()


@dataclass
class CapabilityRegistry:
    _resolver: CandidateCapabilityResolver = field(
        default_factory=RuntimeContextCapabilityResolver
    )
    _manifests: dict[str, CapabilityManifest] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def __init__(
        self,
        manifests: Sequence[CapabilityManifest | Mapping[str, object]] = (),
        resolver: CandidateCapabilityResolver | None = None,
    ) -> None:
        self._resolver = resolver or RuntimeContextCapabilityResolver()
        self._manifests = {}
        self._order = []
        for manifest in manifests:
            self.register(manifest)

    def register(
        self,
        manifest: CapabilityManifest | Mapping[str, object],
    ) -> CapabilityManifest:
        normalized = self.validateManifest(manifest)
        if normalized.id in self._manifests:
            raise ValueError(f"duplicate capability manifest id: {normalized.id}")
        self._manifests[normalized.id] = normalized
        self._order.append(normalized.id)
        return normalized

    def get(self, id: str) -> CapabilityManifest:
        try:
            return self._manifests[id]
        except KeyError as exc:
            raise KeyError(f"unknown capability manifest id: {id}") from exc

    def find(self, id: str) -> CapabilityManifest | None:
        return self._manifests.get(id)

    def list(self) -> tuple[CapabilityManifest, ...]:
        return tuple(self._manifests[id] for id in self._order)

    def resolveCandidateCapabilities(
        self,
        userRequest: object | None,
        runtimeContext: Mapping[str, object] | object | None = None,
    ) -> tuple[CapabilityManifest, ...]:
        return self._resolver.resolve(self, userRequest, runtimeContext)

    def validateManifest(
        self,
        manifest: CapabilityManifest | Mapping[str, object],
    ) -> CapabilityManifest:
        return (
            manifest
            if isinstance(manifest, CapabilityManifest)
            else CapabilityManifest.model_validate(manifest)
        )

    def resolve_candidate_capabilities(
        self,
        user_request: object | None,
        runtime_context: Mapping[str, object] | object | None = None,
    ) -> tuple[CapabilityManifest, ...]:
        return self.resolveCandidateCapabilities(user_request, runtime_context)

    def validate_manifest(
        self,
        manifest: CapabilityManifest | Mapping[str, object],
    ) -> CapabilityManifest:
        return self.validateManifest(manifest)


def _context_values(
    runtime_context: Mapping[str, object] | object | None,
    *keys: str,
) -> tuple[str, ...]:
    for key in keys:
        value: object | None
        if runtime_context is None:
            value = None
        elif isinstance(runtime_context, Mapping):
            value = runtime_context.get(key)
        else:
            value = getattr(runtime_context, key, None)
        values = _string_values(value)
        if values:
            return values
    return ()


def _string_values(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    if isinstance(value, frozenset | set | tuple | list):
        return tuple(str(item) for item in value if str(item))
    return ()


def _unique(values) -> tuple[CapabilityManifest, ...]:
    seen: set[str] = set()
    output: list[CapabilityManifest] = []
    for value in values:
        if value.id in seen:
            continue
        seen.add(value.id)
        output.append(value)
    return tuple(output)
