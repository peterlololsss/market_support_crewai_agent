from __future__ import annotations

import hashlib
import json

from market_support_crewai_agent.runtime.hashing import canonical_json_bytes, hph1
from market_support_crewai_agent.runtime.policy.capabilities import ManifestRefV1
from market_support_crewai_agent.runtime.prompts.auxiliary_contracts import (
    ApprovedKnowledgeCandidateViewV1,
    ApprovedKnowledgeSelectorInputV1,
)
from market_support_crewai_agent.runtime.prompts.invocation_journal import (
    SelectorCacheIdentityV1,
    selector_input_hash,
)
from market_support_crewai_agent.runtime.prompts.program_models import (
    DirectProviderSynthesisV1,
    ProviderJsonSchemaFormatV1,
    ProviderTargetConfigV1,
    resolve_active_prompt_program_v2,
    synthesize_direct_provider_messages,
)
from market_support_crewai_agent.runtime.prompts.provider_transport import (
    build_provider_transport_envelope,
)
from market_support_crewai_agent.runtime.prompts.provider_target import (
    build_provider_target_from_settings,
)


def test_hash_domains_are_distinct_for_prompt_synthesis_and_transport() -> None:
    # Given: one registered direct synthesis and OpenAI-compatible target.
    synthesis = _synthesis("value")
    target = _target()

    # When: each governed domain is hashed.
    envelope = build_provider_transport_envelope(synthesis, target)
    execution_spec = synthesis.system_payload.agent_execution_spec
    hashes = {
        hph1("registered instruction", execution_spec),
        synthesis.osh1(),
        envelope.poh1(),
        envelope.prh1(),
    }

    # Then: each prefix and framed digest domain is distinct.
    assert len(hashes) == 4
    expected_harness_payload = {
        "agent_execution_spec": execution_spec.model_dump(
            mode="json",
            exclude_none=False,
            exclude_defaults=False,
        ),
        "prompt_text": "registered instruction",
    }
    assert hph1("registered instruction", execution_spec) == (
        "hph1:"
        + hashlib.sha256(
            b"harness-prompt.v1\0" + canonical_json_bytes(expected_harness_payload)
        ).hexdigest()
    )
    assert synthesis.osh1().startswith("osh1:")
    assert envelope.poh1().startswith("poh1:")
    assert envelope.prh1().startswith("prh1:")
    assert json.loads(synthesis.messages[0].content) == (
        synthesis.system_payload.model_dump(mode="json")
    )
    assert json.loads(synthesis.messages[1].content) == (
        synthesis.user_payload.model_dump(mode="json")
    )


def test_hph1_changes_when_any_packaged_agent_instruction_field_changes() -> None:
    # Given: one packaged execution spec and mutations for every instruction field.
    synthesis = _synthesis("value")
    execution_spec = synthesis.system_payload.agent_execution_spec
    baseline = hph1("registered instruction", execution_spec)
    mutations = {
        "role": "Mutated role",
        "goal": "Mutated goal",
        "backstory": "Mutated backstory",
        "task_template": "Mutated task template",
        "expected_output_template": "Mutated expected output",
        "agent_spec_version": 2,
    }

    # When: each packaged instruction field changes independently.
    changed_hashes = {
        field: hph1(
            "registered instruction",
            execution_spec.model_copy(update={field: value}),
        )
        for field, value in mutations.items()
    }

    # Then: every instruction mutation changes the harness fingerprint.
    assert set(changed_hashes) == set(mutations)
    assert all(value != baseline for value in changed_hashes.values())


def test_direct_request_hash_uses_provider_visible_prompt_domain() -> None:
    # Given: the complete typed envelope that direct transport sends to a provider.
    envelope = build_provider_transport_envelope(_synthesis("value"), _target())
    payload = canonical_json_bytes(
        envelope.model_dump(
            mode="json",
            exclude_none=False,
            exclude_defaults=False,
        )
    )

    # When: the direct request is fingerprinted.
    actual = envelope.prh1()

    # Then: direct and SDK transports share the frozen provider-visible domain.
    expected = hashlib.sha256(b"provider-visible-prompt.v1\0" + payload).hexdigest()
    legacy = hashlib.sha256(b"provider-request.v1\0" + payload).hexdigest()
    assert actual == f"prh1:{expected}"
    assert actual != f"prh1:{legacy}"


def test_schema_hashes_ignore_recursive_documentation_only_changes() -> None:
    # Given: schemas that differ only in recursive documentation keywords.
    baseline_schema = {
        "type": "object",
        "title": "Baseline",
        "description": "Baseline docs",
        "properties": {
            "value": {
                "type": "string",
                "title": "Value",
                "examples": ["alpha"],
                "$comment": "baseline",
            },
            "title": {"type": "string"},
        },
    }
    changed_docs_schema = {
        "type": "object",
        "title": "Changed",
        "description": "Changed docs",
        "properties": {
            "value": {
                "type": "string",
                "title": "Changed value",
                "examples": ["beta"],
                "$comment": "changed",
            },
            "title": {"type": "string"},
        },
    }
    baseline = _synthesis("alpha", output_schema=baseline_schema)
    changed_docs = _synthesis("alpha", output_schema=changed_docs_schema)
    target = _target()

    # When: canonical and provider schema fingerprints are computed.
    baseline_envelope = build_provider_transport_envelope(baseline, target)
    changed_docs_envelope = build_provider_transport_envelope(changed_docs, target)

    # Then: docs are removed recursively while a property named title is preserved.
    assert baseline.osh1() == changed_docs.osh1()
    assert baseline_envelope.poh1() == changed_docs_envelope.poh1()
    canonical = baseline.system_payload.output_schema.json_schema
    assert canonical == {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "title": {"type": "string"},
        },
    }


def test_schema_hashes_change_when_structural_schema_changes() -> None:
    # Given: two schemas with different machine-consumed constraints.
    baseline = _synthesis("alpha", output_schema={"type": "string"})
    changed = _synthesis(
        "alpha",
        output_schema={"type": "string", "minLength": 1},
    )
    target = _target()

    # When/Then: structural changes alter canonical and transported schema hashes.
    assert baseline.osh1() != changed.osh1()
    assert (
        build_provider_transport_envelope(
            baseline,
            target,
        ).poh1()
        != build_provider_transport_envelope(changed, target).poh1()
    )


def test_schema_hashes_do_not_change_when_only_user_data_changes() -> None:
    # Given: two requests with identical output schemas and different user data.
    baseline = _synthesis("alpha")
    changed_user_data = _synthesis("beta")
    target = _target()

    # When: canonical and transported output-schema domains are fingerprinted.
    baseline_envelope = build_provider_transport_envelope(baseline, target)
    changed_envelope = build_provider_transport_envelope(changed_user_data, target)

    # Then: user data changes neither schema-owned digest.
    assert baseline.osh1() == changed_user_data.osh1()
    assert baseline_envelope.poh1() == changed_envelope.poh1()


def test_output_hash_frames_only_exact_returned_text_bytes() -> None:
    # Given: an available provider output whose wrapper has additional metadata.
    from market_support_crewai_agent.runtime.prompts.program_models import (
        ProviderOutputCaptureV1,
    )

    capture = ProviderOutputCaptureV1(
        status="available_text",
        text='{"ok":true}',
        byte_count=11,
    )

    # When/Then: out1 is the raw text-byte frame, not the capture-model frame.
    expected = hashlib.sha256(b'provider-output-text.v1\0{"ok":true}').hexdigest()
    assert capture.out1() == f"out1:{expected}"


def test_direct_synthesis_exposes_only_typed_system_user_and_message_fields() -> None:
    # Given/When: the direct synthesis contract is inspected independently.
    field_names = set(DirectProviderSynthesisV1.model_fields)

    # Then: the simplified stage/system/user/schema bag is not the wire contract.
    assert field_names == {
        "contract_version",
        "system_payload",
        "user_payload",
        "messages",
    }


def test_selector_cache_hash_changes_on_any_input_byte_change() -> None:
    # Given: two selector payloads that differ only by a candidate value.
    target = _target()
    first = SelectorCacheIdentityV1(
        selector_input_json='{"candidates":["a"],"stage":"approved_knowledge_selector"}',
        program_id="approved_knowledge_selector.scene_neutral.v1@1",
        program_version="2026-07-18.1",
        provider_target=target.identity(),
    )
    second = first.model_copy(
        update={
            "selector_input_json": '{"candidates":["b"],"stage":"approved_knowledge_selector"}'
        }
    )

    # When/Then: exact `sih1` compatibility does not reuse different inputs.
    assert selector_input_hash(first).startswith("sih1:")
    assert selector_input_hash(first) != selector_input_hash(second)


def test_selector_cache_hash_covers_program_and_complete_provider_target() -> None:
    target = _target(model="model-a")
    identity = SelectorCacheIdentityV1(
        selector_input_json='{"query":"Alpha"}',
        program_id="document_product_selector.scene_neutral.v1@1",
        program_version="2026-07-18.1",
        provider_target=target.identity(),
    )

    assert selector_input_hash(identity) != selector_input_hash(
        identity.model_copy(update={"program_version": "2026-07-18.2"})
    )
    assert selector_input_hash(identity) != selector_input_hash(
        identity.model_copy(
            update={
                "provider_target": target.model_copy(
                    update={"model_name": "model-b"}
                ).identity()
            }
        )
    )
    assert selector_input_hash(identity) != selector_input_hash(
        identity.model_copy(
            update={
                "provider_target": target.model_copy(
                    update={"normalized_endpoint": "https://llm.example/v2"}
                ).identity()
            }
        )
    )
    assert selector_input_hash(identity) != selector_input_hash(
        identity.model_copy(
            update={
                "provider_target": _target(
                    provider="gemini",
                    model="gemini-3-flash-preview",
                ).identity()
            }
        )
    )


def _target(
    *,
    provider: str = "openai",
    model: str = "deepseek-v4-pro",
    base_url: str = "https://llm.example/v1",
) -> ProviderTargetConfigV1:
    return build_provider_target_from_settings(
        provider=provider,
        target_slot="selector",
        model=model,
        base_url=base_url,
        api_key_configured=True,
        timeout_seconds=20.0,
        temperature=0.0,
        max_tokens=1200,
    )


def _synthesis(
    user_query: str,
    *,
    output_schema: dict[str, object] | None = None,
) -> DirectProviderSynthesisV1:
    program, execution_spec = resolve_active_prompt_program_v2(
        stage="approved_knowledge_selector",
        scene_key="scene_neutral.v1",
    )
    manifest_ref = ManifestRefV1(
        manifest_id="answer_internal_company_knowledge",
        manifest_version="2026-07-18.1",
    )
    stage_input = ApprovedKnowledgeSelectorInputV1(
        user_query=user_query,
        evidence_query="",
        candidates=(
            ApprovedKnowledgeCandidateViewV1(
                entry_id="company_profile",
                question="company profile",
                title="company profile",
                semantic_purpose="company facts",
                manifest_ref=manifest_ref,
                fact_type="document_context",
            ),
        ),
        max_entries=1,
        max_images=0,
        selected_manifest_ref=manifest_ref,
    )
    return synthesize_direct_provider_messages(
        program=program,
        execution_spec=execution_spec,
        prompt_text="registered instruction",
        stage_input=stage_input,
        output_schema=ProviderJsonSchemaFormatV1(
            name="SelectorOutput",
            json_schema=output_schema or {"type": "object"},
        ),
    )
