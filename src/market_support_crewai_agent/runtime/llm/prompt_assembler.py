from __future__ import annotations

import hashlib
from dataclasses import dataclass

from market_support_crewai_agent.runtime.domain.compliance_policy import (
    compliance_policy_prompt_lines,
)
from market_support_crewai_agent.runtime.llm.prompt_context import (
    PromptAssemblyContext,
    render_prompt_context,
)
from market_support_crewai_agent.runtime.llm.prompt_fragments import render_prompt_fragment
from market_support_crewai_agent.runtime.llm.prompt_profiles import PromptProfile


@dataclass(frozen=True)
class PromptProgram:
    profile: PromptProfile
    fragment_ids: tuple[str, ...]
    prompt_text: str
    prompt_hash: str
    fragment_hashes: dict[str, str]


def assemble_prompt_program(
    ctx: PromptAssemblyContext,
    profile: PromptProfile,
    fragment_ids: tuple[str, ...],
) -> PromptProgram:
    ordered_fragment_ids = _dedupe(fragment_ids)
    template_context = _template_context()
    sections: list[str] = []
    fragment_hashes: dict[str, str] = {}
    for fragment_id in ordered_fragment_ids:
        fragment_text = render_prompt_fragment(
            fragment_id,
            ctx.stage,
            **template_context,
        )
        fragment_hashes[fragment_id] = _sha256(fragment_text)
        sections.append(
            '<prompt_fragment id="{}">\n{}\n</prompt_fragment>'.format(
                fragment_id,
                fragment_text,
            )
        )
    sections.append(
        "<runtime_context>\n{}\n</runtime_context>".format(
            render_prompt_context(ctx)
        )
    )
    prompt_text = "\n\n".join(sections)
    return PromptProgram(
        profile=profile,
        fragment_ids=ordered_fragment_ids,
        prompt_text=prompt_text,
        prompt_hash=_sha256(prompt_text),
        fragment_hashes=fragment_hashes,
    )


def _template_context() -> dict[str, str]:
    return {
        "compliance_policy_lines": "\n".join(compliance_policy_prompt_lines()),
    }


def _dedupe(fragment_ids: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for fragment_id in fragment_ids:
        if fragment_id in seen:
            continue
        seen.add(fragment_id)
        output.append(fragment_id)
    return tuple(output)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
