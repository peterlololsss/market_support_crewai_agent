from __future__ import annotations

import asyncio
import json
import re
from typing import Literal, Protocol

from pydantic import Field

from market_support_crewai_agent.runtime.domain.canonicalization import CanonicalContext
from market_support_crewai_agent.runtime.orchestration.decision import ResponseDirective
from market_support_crewai_agent.runtime.evidence import EvidenceFact
from market_support_crewai_agent.runtime.domain.planning import ExecutionPlan
from market_support_crewai_agent.schemas import ReplyRequest, ReplyResponse, StrictModel
from market_support_crewai_agent.settings import Settings, get_settings

AlignmentIssueCode = Literal[
    "wrong_image_marker",
    "missing_image_marker",
    "image_marker_not_supported",
]
AlignmentRemediation = Literal["none", "recompose", "refetch_document_context"]

_IMAGE_MARKER_RE = re.compile(r"%%([\w\d_.-]+\.png)%%")


class AlignmentIssue(StrictModel):
    code: AlignmentIssueCode
    message: str
    metadata: dict[str, object] = Field(default_factory=dict)


class AlignmentVerification(StrictModel):
    valid: bool = True
    issues: tuple[AlignmentIssue, ...] = ()
    remediation: AlignmentRemediation = "none"
    rationale: str = Field(default="", max_length=800)


class SemanticAlignmentVerifier(Protocol):
    async def verify(
        self,
        *,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        directive: ResponseDirective,
        evidence_facts: list[EvidenceFact],
        response: ReplyResponse,
    ) -> AlignmentVerification: ...


class NoopSemanticAlignmentVerifier:
    async def verify(
        self,
        *,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        directive: ResponseDirective,
        evidence_facts: list[EvidenceFact],
        response: ReplyResponse,
    ) -> AlignmentVerification:
        del request, canonical_context, plan, directive, evidence_facts, response
        return AlignmentVerification(valid=True)


class CrewAISemanticAlignmentVerifier:
    """Final semantic judge for image-marker alignment."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def verify(
        self,
        *,
        request: ReplyRequest,
        canonical_context: CanonicalContext,
        plan: ExecutionPlan,
        directive: ResponseDirective,
        evidence_facts: list[EvidenceFact],
        response: ReplyResponse,
    ) -> AlignmentVerification:
        supported_markers = supported_image_markers(evidence_facts)
        reply_markers = tuple(_image_markers(response.reply.text))
        support_result = _deterministic_support_check(reply_markers, supported_markers)
        if not support_result.valid:
            return support_result

        if not reply_markers and not supported_markers:
            return AlignmentVerification(valid=True)
        if not self.settings.llm_api_key:
            return AlignmentVerification(valid=True)

        prompt = _alignment_prompt(
            request=request,
            canonical_context=canonical_context,
            plan=plan,
            directive=directive,
            evidence_facts=evidence_facts,
            response=response,
            supported_markers=supported_markers,
            reply_markers=reply_markers,
        )
        return await _run_crewai_alignment(
            prompt,
            self.settings,
            timeout_seconds=self.settings.llm_timeout_seconds,
        )


def supported_image_markers(evidence_facts: list[EvidenceFact]) -> tuple[str, ...]:
    markers: list[str] = []
    seen: set[str] = set()
    for fact in evidence_facts:
        if not _is_trusted_document_context(fact):
            continue
        for marker in fact.metadata.get("image_markers") or ():
            marker_text = str(marker)
            if marker_text and marker_text not in seen:
                seen.add(marker_text)
                markers.append(marker_text)
        for filename in _image_markers(str(fact.value or "")):
            marker_text = f"%%{filename}%%"
            if marker_text and marker_text not in seen:
                seen.add(marker_text)
                markers.append(marker_text)
    return tuple(markers)


def _deterministic_support_check(
    reply_markers: tuple[str, ...],
    supported_markers: tuple[str, ...],
) -> AlignmentVerification:
    unsupported = [
        marker for marker in reply_markers if f"%%{marker}%%" not in supported_markers
    ]
    if not unsupported:
        return AlignmentVerification(valid=True)
    return AlignmentVerification(
        valid=False,
        remediation="refetch_document_context",
        issues=tuple(
            AlignmentIssue(
                code="image_marker_not_supported",
                message="reply includes an image marker that was not selected into trusted evidence",
                metadata={"filename": marker},
            )
            for marker in unsupported
        ),
    )


def _alignment_prompt(
    *,
    request: ReplyRequest,
    canonical_context: CanonicalContext,
    plan: ExecutionPlan,
    directive: ResponseDirective,
    evidence_facts: list[EvidenceFact],
    response: ReplyResponse,
    supported_markers: tuple[str, ...],
    reply_markers: tuple[str, ...],
) -> str:
    payload = {
        "current_user_message": request.message,
        "canonical_context": canonical_context.to_prompt_dict(),
        "execution_plan": plan.model_dump(mode="json", exclude_none=True),
        "response_directive": directive.model_dump(mode="json", exclude_none=True),
        "reply": response.model_dump(mode="json", exclude_none=True),
        "supported_image_markers": supported_markers,
        "reply_image_filenames": reply_markers,
        "trusted_image_evidence": _trusted_image_evidence(evidence_facts),
    }
    return (
        "You are the final semantic alignment verifier for image markers in a "
        "deterministic support reply harness.\n\n"
        "Judge only whether image markers in the final reply are semantically "
        "responsive to the current user request and supported by trusted evidence.\n"
        "Do not select image markers directly. Do not create new image markers, "
        "filenames, facts, actions, or reply text.\n\n"
        "Reject with wrong_image_marker when the reply includes an approved and "
        "supported marker, but that image is not responsive to the current request.\n"
        "Reject with missing_image_marker when the user explicitly requested an "
        "available image, QR code, chart, or diagram and trusted selected evidence "
        "supports the marker, but reply.text omitted it.\n"
        "Reject with image_marker_not_supported when reply.text includes a marker "
        "that is not in supported_image_markers.\n"
        "Use remediation='recompose' for wrong_image_marker or missing_image_marker. "
        "Use remediation='refetch_document_context' only for unsupported markers.\n"
        "If no issue exists, return valid=true and remediation='none'.\n\n"
        "Verifier input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}"
    )


def _trusted_image_evidence(evidence_facts: list[EvidenceFact]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for fact in evidence_facts:
        if not _is_trusted_document_context(fact):
            continue
        markers = tuple(fact.metadata.get("image_markers") or ())
        if not markers and not _image_markers(str(fact.value or "")):
            continue
        output.append(
            {
                "source_type": fact.source_type,
                "source_id": fact.source_id,
                "value": fact.value,
                "image_markers": markers,
                "image_asset_ids": tuple(fact.metadata.get("image_asset_ids") or ()),
                "image_asset_semantic_purposes": dict(
                    fact.metadata.get("image_asset_semantic_purposes") or {}
                ),
                "title": fact.metadata.get("title", ""),
            }
        )
    return output


async def _run_crewai_alignment(
    prompt: str,
    settings: Settings,
    *,
    timeout_seconds: float,
) -> AlignmentVerification:
    from crewai import Agent, LLM

    agent = Agent(
        role="Final Image Alignment Verifier",
        goal="Judge image-marker semantic alignment for a validated support reply.",
        backstory=(
            "You are a bounded verifier. You do not compose the reply and you do "
            "not choose image assets."
        ),
        llm=LLM(
            model=settings.llm_model,
            provider=settings.llm_provider,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0,
            max_tokens=min(settings.llm_max_tokens, 1200),
            timeout=settings.llm_timeout_seconds,
        ),
        allow_delegation=False,
        verbose=settings.crewai_verbose,
        max_iter=1,
        max_execution_time=settings.crewai_max_execution_time,
        max_retry_limit=settings.crewai_max_retry_limit,
        planning=False,
    )
    result = await asyncio.wait_for(
        agent.kickoff_async(prompt, response_format=AlignmentVerification),
        timeout=timeout_seconds,
    )
    if result.pydantic is not None:
        return AlignmentVerification.model_validate(result.pydantic)
    return AlignmentVerification.model_validate_json(result.raw)


def _image_markers(text: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for match in _IMAGE_MARKER_RE.finditer(str(text or "")):
        filename = match.group(1)
        if filename in seen:
            continue
        seen.add(filename)
        output.append(filename)
    return output


def _is_trusted_document_context(fact: EvidenceFact) -> bool:
    if fact.fact_type != "document_context" or not fact.value:
        return False
    if fact.source_type == "document_mcp":
        return True
    return (
        fact.source_type == "approved_static_knowledge"
        and fact.metadata.get("approved_static_knowledge") is True
        and fact.metadata.get("content_is_data_only") is True
    )
