from __future__ import annotations

from typing import Final, assert_never

from market_support_crewai_agent.runtime.context import report_view_projection
from market_support_crewai_agent.runtime.context.evidence_view_models import (
    EvidenceBooleanValueViewV1,
    EvidenceContentValueViewV1,
    EvidenceFactViewV1,
    EvidenceIntegerValueViewV1,
    EvidenceNullValueViewV1,
    EvidenceNumberValueViewV1,
    EvidenceScopeViewV1,
    EvidenceStringValueViewV1,
    EvidenceValueViewV1,
)
from market_support_crewai_agent.runtime.context.grounding_projection_context import (
    GroundingProjectionContextV1,
)
from market_support_crewai_agent.runtime.context.view_errors import (
    ContextViewInvariantError,
)
from market_support_crewai_agent.runtime.evidence.canonical_identity import (
    DistributionEvidenceScopeIdentityV1,
    UnscopedEvidenceScopeIdentityV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_models import (
    CanonicalEvidenceFactTypeV1,
    CanonicalEvidenceFactV1,
)
from market_support_crewai_agent.runtime.evidence.canonical_values import (
    EvidenceBooleanValueV1,
    EvidenceContentValueV1,
    EvidenceIntegerValueV1,
    EvidenceNullValueV1,
    EvidenceNumberValueV1,
    EvidenceStringValueV1,
    EvidenceValueCanonicalV1,
)
from market_support_crewai_agent.schemas.conversation import (
    DistributionScopeV1,
    UnscopedScopeV1,
)

_ANSWER_CONTENT_FACT_TYPES: Final = frozenset({"document_context"})


def project_evidence_fact_view_v1(
    fact: CanonicalEvidenceFactV1,
    context: GroundingProjectionContextV1,
) -> EvidenceFactViewV1:
    if len(fact.public_urls) > 8:
        raise ContextViewInvariantError("evidence_public_url_limit_exceeded")
    if any(context.locator_safety.public_url(url) is None for url in fact.public_urls):
        raise ContextViewInvariantError("evidence_public_url_not_safe")
    report_payload = report_view_projection.project_report_payload_view_v1(
        fact.report_payload
    )
    value = (
        None
        if report_payload is not None
        else _project_evidence_value(fact.value, fact.fact_type)
    )
    return EvidenceFactViewV1(
        evidence_id=fact.evidence_id,
        fact_type=fact.fact_type,
        source_type=fact.source_type,
        artifact_type=fact.artifact_type,
        value=value,
        report_payload=report_payload,
        scope=_project_evidence_scope(fact, context),
        observed_age_seconds=_observed_age_seconds(fact, context),
        provenance_available=True,
        citation_available=bool(fact.public_urls),
        public_urls=fact.public_urls,
    )


def _project_evidence_value(
    value: EvidenceValueCanonicalV1,
    fact_type: CanonicalEvidenceFactTypeV1,
) -> EvidenceValueViewV1:
    match value:
        case EvidenceNullValueV1():
            return EvidenceNullValueViewV1()
        case EvidenceBooleanValueV1():
            return EvidenceBooleanValueViewV1(value=value.value)
        case EvidenceIntegerValueV1():
            return EvidenceIntegerValueViewV1(value=value.value)
        case EvidenceNumberValueV1():
            return EvidenceNumberValueViewV1(value=value.value.root)
        case EvidenceStringValueV1():
            return EvidenceStringValueViewV1(value=value.value)
        case EvidenceContentValueV1():
            limit = 1_000_000 if fact_type in _ANSWER_CONTENT_FACT_TYPES else 6_000
            if len(value.text) > limit:
                raise ContextViewInvariantError(
                    "evidence_content_projection_limit_exceeded"
                )
            return EvidenceContentValueViewV1(
                media_type=value.media_type,
                charset=value.charset,
                text=value.text,
            )
        case unreachable:
            assert_never(unreachable)


def _project_evidence_scope(
    fact: CanonicalEvidenceFactV1,
    context: GroundingProjectionContextV1,
) -> EvidenceScopeViewV1:
    match fact.scope:
        case DistributionEvidenceScopeIdentityV1():
            match context.business_scope_authority.scope:
                case DistributionScopeV1():
                    if (
                        fact.scope.business_scope_ref
                        != context.business_scope_authority.business_scope_ref
                        or fact.scope.channel_kind
                        != context.business_scope_authority.scope.channel_type
                    ):
                        raise ContextViewInvariantError(
                            "evidence_scope_authority_mismatch"
                        )
                    return EvidenceScopeViewV1(
                        kind="distribution",
                        channel_kind=fact.scope.channel_kind,
                        channel_match=True,
                        artifact_type=fact.artifact_type,
                        material_option=fact.scope.material_option,
                        period=fact.scope.period,
                        report_date=fact.scope.report_date,
                        product_count=len(fact.scope.product_ids),
                    )
                case UnscopedScopeV1():
                    raise ContextViewInvariantError(
                        "evidence_scope_authority_kind_mismatch"
                    )
                case unreachable:
                    assert_never(unreachable)
        case UnscopedEvidenceScopeIdentityV1():
            match context.business_scope_authority.scope:
                case UnscopedScopeV1():
                    return EvidenceScopeViewV1(
                        kind="unscoped",
                        artifact_type=fact.artifact_type,
                        material_option=fact.scope.material_option,
                        period=fact.scope.period,
                        report_date=fact.scope.report_date,
                        product_count=None,
                    )
                case DistributionScopeV1():
                    raise ContextViewInvariantError(
                        "evidence_scope_authority_kind_mismatch"
                    )
                case unreachable:
                    assert_never(unreachable)
        case unreachable:
            assert_never(unreachable)


def _observed_age_seconds(
    fact: CanonicalEvidenceFactV1,
    context: GroundingProjectionContextV1,
) -> int | None:
    if (
        fact.observed_at_epoch_seconds is None
        or context.evaluation_epoch_seconds is None
    ):
        return None
    return max(
        0,
        context.evaluation_epoch_seconds - fact.observed_at_epoch_seconds,
    )
