"""Exact Colonist trade lifecycle state for replay mode."""

from __future__ import annotations

from copy import deepcopy
from typing import Final

from cle.replay.colonist.types import ActionHint, TradeLedgerRecord
from cle.replay.contracts import ReplayRuntimeState

TRADE_OFFER_TYPES: Final[set[str]] = {"OFFER_TRADE", "COUNTER_OFFER"}
TRADE_RESPONSE_TYPES: Final[set[str]] = {
    "ACCEPT_TRADE",
    "REJECT_TRADE",
    "CLEAR_TRADE_RESPONSE",
}


def ensure_replay_trade_ledger(state: ReplayRuntimeState) -> None:
    """Initialize exact trade storage on older ServerState instances."""
    if not hasattr(state, "replay_trade_ledger"):
        state.replay_trade_ledger = {}


def _record_from_hint(action_hint: ActionHint) -> TradeLedgerRecord:
    offered = tuple(action_hint.get("offered") or (0, 0, 0, 0, 0))
    wanted = tuple(action_hint.get("wanted") or (0, 0, 0, 0, 0))
    trade_tuple = action_hint.get("trade_tuple") or ()
    offered_any = trade_tuple[10] if len(trade_tuple) > 10 else 0
    wanted_any = trade_tuple[11] if len(trade_tuple) > 11 else 0
    responses = {
        str(player_id): "accepted" if response == 1 else "rejected"
        for player_id, response in action_hint.get("player_responses", {}).items()
        if response in (1, 2)
    }
    return {
        "trade_id": action_hint.get("trade_id"),
        "trade_num": action_hint.get("trade_num", 0),
        "creator": action_hint.get("player", action_hint.get("creator")),
        "offered": offered,
        "wanted": wanted,
        "offered_any": offered_any,
        "wanted_any": wanted_any,
        "is_flexible": bool(offered_any or wanted_any),
        "is_counter_offer": bool(action_hint.get("is_counter_offer")),
        "counter_offer_to": action_hint.get("counter_offer_to"),
        "responses": responses,
    }


def _closures_from_hint(action_hint: ActionHint) -> list[ActionHint]:
    closures = action_hint.get("closed_trades")
    if closures is not None:
        return closures
    if action_hint.get("type") == "CLOSE_TRADE":
        return [action_hint]
    return []


def apply_replay_trade_event(
    state: ReplayRuntimeState,
    action_hint: ActionHint,
) -> dict[str, list[object]]:
    """Apply one parsed action to the exact trade-ID ledger."""
    ensure_replay_trade_ledger(state)
    ledger = state.replay_trade_ledger
    action_type = action_hint.get("type")
    changes: dict[str, list[object]] = {"opened": [], "responded": [], "closed": []}

    if action_type in TRADE_OFFER_TYPES:
        trade_id = action_hint.get("trade_id")
        if trade_id is not None:
            existing_responses = deepcopy(
                ledger.get(trade_id, {}).get("responses", {})
            )
            record = _record_from_hint(action_hint)
            if "player_responses" not in action_hint:
                record["responses"] = existing_responses
            ledger[trade_id] = record
            changes["opened"].append(trade_id)

    if action_type in TRADE_RESPONSE_TYPES:
        trade_id = action_hint.get("trade_id")
        if trade_id is not None:
            if trade_id not in ledger:
                ledger[trade_id] = _record_from_hint(action_hint)
                ledger[trade_id]["creator"] = action_hint.get("creator")
            responder = action_hint.get("player")
            if action_type == "CLEAR_TRADE_RESPONSE":
                response = None
                ledger[trade_id]["responses"].pop(str(responder), None)
            else:
                response = (
                    "accepted" if action_type == "ACCEPT_TRADE" else "rejected"
                )
                ledger[trade_id]["responses"][str(responder)] = response
            changes["responded"].append({
                "trade_id": trade_id,
                "player": responder,
                "response": response,
            })

    for closure in _closures_from_hint(action_hint):
        trade_id = closure.get("trade_id")
        if trade_id is None:
            continue
        closed_record: TradeLedgerRecord | None = ledger.pop(trade_id, None)
        changes["closed"].append({
            "trade_id": trade_id,
            "reason": closure.get("reason"),
            "existed": closed_record is not None,
        })

    return changes


def replay_trade_ledger_payload(state: ReplayRuntimeState) -> list[TradeLedgerRecord]:
    """Return JSON-safe active replay trades in source insertion order."""
    ensure_replay_trade_ledger(state)
    return [deepcopy(record) for record in state.replay_trade_ledger.values()]
