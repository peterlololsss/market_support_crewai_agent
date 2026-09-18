from market_support_crewai_agent.runtime.identity import ConversationStateKey
from market_support_crewai_agent.runtime.state.coordinator_state import (
    CoordinatorStateRootV1,
)
from market_support_crewai_agent.runtime.state.effect_records import (
    ActionLedgerRecordV2,
    CompleteIssuedResponseRecordV1,
    PreparedEffectTransitionV1,
)


def derive_ledger_updates(
    *,
    root: CoordinatorStateRootV1,
    record: CompleteIssuedResponseRecordV1,
    transitions: tuple[PreparedEffectTransitionV1, ...],
    received_at_epoch_ms: int,
    expires_at_monotonic_ns: int,
) -> dict[tuple[ConversationStateKey, str, str], ActionLedgerRecordV2]:
    transition_by_key = {
        transition.effect_key: transition for transition in transitions
    }
    updates: dict[tuple[ConversationStateKey, str, str], ActionLedgerRecordV2] = {}
    for effect in record.effects:
        transition = transition_by_key.get(effect.effect_key)
        if transition is None or transition.is_noop or effect.action_id is None:
            continue
        match effect.effect_type:
            case "send_material_pack" | "send_weekly_report" | "send_monthly_report":
                key = (record.state_key, record.response.response_id, effect.action_id)
                previous = root.action_ledger_records.get(key)
                match previous:
                    case ActionLedgerRecordV2(status_revision=previous_revision):
                        status_revision = previous_revision + 1
                    case None:
                        status_revision = 1
                updates[key] = ActionLedgerRecordV2(
                    state_key=record.state_key,
                    response_id=record.response.response_id,
                    action_id=effect.action_id,
                    action_type=effect.effect_type,
                    status=transition.requested_status,
                    artifact_ref=transition.requested_artifact_ref,
                    status_revision=status_revision,
                    received_at_epoch_ms=received_at_epoch_ms,
                    expires_at_monotonic_ns=expires_at_monotonic_ns,
                    pol1=record.pol1,
                    par1=record.par1,
                    grh1=record.grh1,
                    bsh1=record.bsh1,
                )
            case "send_text" | "mention_sales":
                continue
    return updates


def admitted_action_ledger_rows(
    root: CoordinatorStateRootV1,
    *,
    state_key: ConversationStateKey,
    par1: str,
    grh1: str,
    bsh1: str,
) -> tuple[ActionLedgerRecordV2, ...]:
    rows: dict[tuple[ConversationStateKey, str, str], ActionLedgerRecordV2] = {}
    for row in root.action_ledger_records.values():
        if (
            row.state_key != state_key
            or row.par1 != par1
            or row.grh1 != grh1
            or row.bsh1 != bsh1
        ):
            continue
        effect_identity = _issued_effect_identity(root, row)
        if effect_identity is not None:
            rows[effect_identity] = row
    return tuple(
        sorted(
            rows.values(),
            key=lambda row: (
                row.received_at_epoch_ms,
                row.response_id,
                row.action_id,
            ),
            reverse=True,
        )[:20]
    )


def _issued_effect_identity(
    root: CoordinatorStateRootV1,
    row: ActionLedgerRecordV2,
) -> tuple[ConversationStateKey, str, str] | None:
    issued_key = root.response_index.get(row.response_id)
    if issued_key is None or issued_key[0] != row.state_key:
        return None
    record = root.issued_records.get(issued_key)
    if (
        record is None
        or record.phase != "complete"
        or record.state_key != row.state_key
        or record.response.response_id != row.response_id
    ):
        return None
    matches = tuple(
        effect
        for effect in record.effects
        if effect.action_id == row.action_id and effect.effect_type == row.action_type
    )
    if len(matches) != 1:
        return None
    return (row.state_key, row.response_id, matches[0].effect_key)
