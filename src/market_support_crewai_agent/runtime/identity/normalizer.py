from __future__ import annotations

import hashlib

from market_support_crewai_agent.runtime.identity.models import (
    AuthenticatedAdapterCallerV1,
    ConversationStateKey,
    KernelReplyRequestV1,
    PolicyRequestKernelV1,
    VerifiedConversationIdentityV1,
    VerifiedRequestEnvelopeV1,
)
from market_support_crewai_agent.runtime.identity.state_key import state_key_ref
from market_support_crewai_agent.schemas.conversation import ReplyRequestV2
from market_support_crewai_agent.schemas.feedback import ActionFeedbackRequestV2


def normalize_reply_request_v2(
    request: ReplyRequestV2,
    *,
    adapter_namespace: str,
) -> VerifiedRequestEnvelopeV1:
    subject_ref = (
        request.identity.group_ref
        if request.identity.scene == "group"
        else request.identity.direct_thread_ref
    )
    identity = VerifiedConversationIdentityV1(
        scene=request.identity.scene,
        tenant_ref=request.identity.tenant_ref,
        subject_ref=subject_ref,
        principal_ref=request.identity.principal_ref,
    )
    state_key = ConversationStateKey(
        surface="wecom",
        adapter_namespace=adapter_namespace,
        tenant_ref=identity.tenant_ref,
        scene=identity.scene,
        subject_ref=identity.subject_ref,
        principal_ref=identity.principal_ref,
    )
    return VerifiedRequestEnvelopeV1(
        caller=AuthenticatedAdapterCallerV1(adapter_namespace=adapter_namespace),
        request=KernelReplyRequestV1(
            request_id=request.request_id,
            replay_eligible=True,
            message=request.message,
            context_id=request.context_id,
            identity=identity,
            presentation=request.presentation,
            business_scope=request.business_scope,
            grants=request.grants,
        ),
        state_key=state_key,
        state_key_ref=state_key_ref(state_key),
    )


def policy_request_kernel_v1(
    envelope: VerifiedRequestEnvelopeV1,
) -> PolicyRequestKernelV1:
    return PolicyRequestKernelV1(
        adapter_namespace=envelope.caller.adapter_namespace,
        replay_eligible=envelope.request.replay_eligible,
        message=envelope.request.message,
        context_id=envelope.request.context_id,
        identity=envelope.request.identity,
        presentation=envelope.request.presentation,
        business_scope=envelope.request.business_scope,
        grants=envelope.request.grants,
    )


def request_kernel_hash(envelope: VerifiedRequestEnvelopeV1) -> str:
    payload = policy_request_kernel_v1(envelope).model_dump(mode="json")
    import json

    digest = hashlib.sha256(
        b"request-kernel.v1\0"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"rqh1:{digest}"


def feedback_state_key_v2(
    feedback: ActionFeedbackRequestV2,
    *,
    adapter_namespace: str,
) -> ConversationStateKey:
    subject_ref = (
        feedback.identity.group_ref
        if feedback.identity.scene == "group"
        else feedback.identity.direct_thread_ref
    )
    return ConversationStateKey(
        surface="wecom",
        adapter_namespace=adapter_namespace,
        tenant_ref=feedback.identity.tenant_ref,
        scene=feedback.identity.scene,
        subject_ref=subject_ref,
        principal_ref=feedback.identity.principal_ref,
    )
