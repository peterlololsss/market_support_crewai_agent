from dataclasses import replace

from market_support_crewai_agent.runtime.state.coordinator_errors import (
    CoordinatorError,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    CompleteIssuedResponseRecordV1,
    IssuedEffectV1,
    PreparedEffectTransitionV1,
)
from market_support_crewai_agent.schemas.feedback import (
    ActionExecutionFeedbackV2,
    ActionFeedbackRequestV2,
    MaterialPackFeedbackArtifact,
    MonthlyReportFeedbackArtifact,
    WeeklyReportFeedbackArtifact,
)
from market_support_crewai_agent.schemas.reply import ReplyResponse


def derive_issued_effects(response: ReplyResponse) -> tuple[IssuedEffectV1, ...]:
    effects: list[IssuedEffectV1] = []
    if response.reply.kind != "no_reply" and response.reply.text:
        effects.append(
            IssuedEffectV1(
                "send_text", "send_text", None, None, None, None, None, None, None, ()
            )
        )
    for mention in response.reply.mentions:
        if mention.type == "sales":
            effects.append(
                IssuedEffectV1(
                    "mention:sales",
                    "mention_sales",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    (),
                )
            )
    for action in response.actions:
        effects.append(
            IssuedEffectV1(
                effect_key=f"action:{action.action_id}",
                effect_type=action.type,
                action_id=action.action_id,
                resolve_ref=action.resolve_ref,
                material_pack_option=getattr(action, "material_pack_option", None),
                period=getattr(action, "period", None),
                report_date=getattr(action, "report_date", None),
                status=None,
                artifact_ref=None,
                status_history=(),
            )
        )
    return tuple(effects)


def prepare_effect_transitions(
    record: CompleteIssuedResponseRecordV1,
    feedback: ActionFeedbackRequestV2,
) -> tuple[PreparedEffectTransitionV1, ...]:
    _validate_execution_set(feedback)
    transitions: list[PreparedEffectTransitionV1] = []
    for execution in feedback.executions:
        effect_key = (
            f"action:{execution.action_id}"
            if execution.action_id is not None
            else (
                "mention:sales"
                if execution.action_type == "mention_sales"
                else "send_text"
            )
        )
        effect = next(
            (item for item in record.effects if item.effect_key == effect_key), None
        )
        if effect is None or effect.effect_type != execution.action_type:
            raise CoordinatorError("feedback_effect_mismatch")
        _validate_effect_metadata(effect, execution)
        artifact_ref = (
            execution.artifact.artifact_ref if execution.artifact is not None else None
        )
        _validate_status_transition(
            effect.status, effect.artifact_ref, execution.status, artifact_ref
        )
        transitions.append(
            PreparedEffectTransitionV1(
                effect_key=effect.effect_key,
                effect_type=effect.effect_type,
                expected_status=effect.status,
                requested_status=execution.status,
                expected_artifact_ref=effect.artifact_ref,
                requested_artifact_ref=artifact_ref,
                is_noop=effect.status == execution.status
                and effect.artifact_ref == artifact_ref,
            )
        )
    return tuple(transitions)


def apply_effect_transitions(
    effects: tuple[IssuedEffectV1, ...],
    transitions: tuple[PreparedEffectTransitionV1, ...],
) -> tuple[IssuedEffectV1, ...]:
    by_key = {transition.effect_key: transition for transition in transitions}
    updated: list[IssuedEffectV1] = []
    for effect in effects:
        transition = by_key.get(effect.effect_key)
        if transition is None or transition.is_noop:
            updated.append(effect)
            continue
        updated.append(
            replace(
                effect,
                status=transition.requested_status,
                artifact_ref=transition.requested_artifact_ref,
                status_history=(*effect.status_history, transition.requested_status),
            )
        )
    return tuple(updated)


def _validate_execution_set(feedback: ActionFeedbackRequestV2) -> None:
    effect_keys: set[str] = set()
    for execution in feedback.executions:
        match execution.action_type, execution.action_id, execution.artifact:
            case (
                ("send_material_pack" | "send_weekly_report" | "send_monthly_report"),
                str() as action_id,
                MaterialPackFeedbackArtifact()
                | WeeklyReportFeedbackArtifact()
                | MonthlyReportFeedbackArtifact(),
            ):
                effect_key = f"action:{action_id}"
            case "send_text", None, None:
                effect_key = "send_text"
            case "mention_sales", None, None:
                effect_key = "mention:sales"
            case _:
                raise CoordinatorError("feedback_effect_mismatch")
        if effect_key in effect_keys:
            raise CoordinatorError("feedback_effect_duplicate")
        effect_keys.add(effect_key)


def _validate_effect_metadata(
    effect: IssuedEffectV1,
    execution: ActionExecutionFeedbackV2,
) -> None:
    match execution.action_type, execution.artifact:
        case "send_material_pack", MaterialPackFeedbackArtifact(
            option=option, resolve_ref=resolve_ref
        ):
            matches = (
                resolve_ref == effect.resolve_ref
                and option == effect.material_pack_option
                and effect.period is None
                and effect.report_date is None
            )
        case "send_weekly_report", WeeklyReportFeedbackArtifact(
            period=period, report_date=report_date, resolve_ref=resolve_ref
        ):
            matches = (
                resolve_ref == effect.resolve_ref
                and period == effect.period
                and report_date == effect.report_date
                and effect.material_pack_option is None
            )
        case "send_monthly_report", MonthlyReportFeedbackArtifact(
            period=period, report_date=report_date, resolve_ref=resolve_ref
        ):
            matches = (
                resolve_ref == effect.resolve_ref
                and period == effect.period
                and report_date == effect.report_date
                and effect.material_pack_option is None
            )
        case "send_text" | "mention_sales", None:
            matches = True
        case _:
            matches = False
    if not matches:
        raise CoordinatorError("feedback_effect_metadata_mismatch")


def _validate_status_transition(
    current: str | None,
    current_artifact: str | None,
    requested: str,
    requested_artifact: str | None,
) -> None:
    if current == requested and current_artifact == requested_artifact:
        return
    if current is None:
        return
    if current in {"failed", "skipped"} and requested == "executed":
        return
    raise CoordinatorError("feedback_effect_conflict")
