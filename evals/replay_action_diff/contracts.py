"""Schema versions, action families, the decision-point record, and narrowing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Required, TypedDict

from cle.game_engine.game import GameEngine
from cle.players.contracts import PlayerContext
from cle.replay.contracts import ParsedActions, ReplayArchive, ReplayRuntimeState
from cle.replay.runtime.access import get_game_engine
from evals.json_types import JsonDict, JsonList, JsonValue, as_dict, as_list

SCHEMA_VERSION = "replay-action-diff-v2"
COMPARISON_PARSER_VERSION = "contract-aware-action-parser-v4"
SELECTION_CONTRACT = "validated-player-choice-v1"
COMPOUND_ACTIONS = {
    "PLAY_MONOPOLY": "MONOPOLY_RESOURCE",
    "PLAY_YEAR_OF_PLENTY": "YEAR_OF_PLENTY_RESOURCES",
}
COMPOUND_FOLLOWUPS = set(COMPOUND_ACTIONS.values())
ASYNC_TRADE_RESPONSES = {"ACCEPT_TRADE", "REJECT_TRADE"}
# Historical comparisons stay index-only; exact new payloads do not relabel old decisions.
COARSE_ACTIONS = {
    "OFFER_TRADE": "trade terms are not represented by the indexed meta-action",
    "COUNTER_OFFER": "counter-offer terms are not represented by the indexed meta-action",
    "DISCARD": "the indexed interface omits the human's controlled card selection",
}
LIFECYCLE_ACTIONS = {
    "CLEAR_TRADE_RESPONSE": "trade lifecycle update, not an indexed decision",
    "CLOSE_TRADE": "trade lifecycle closure, not an indexed decision",
}


class HumanMatch(TypedDict, total=False):
    """How a recorded human choice mapped onto the indexed legal menu."""

    status: Required[str]
    reason: Required[str | None]
    action_index: Required[int | None]
    normalization: str
    candidate_indices: list[int]


@dataclass
class DecisionPoint:
    """One exact human decision and its shared provider-safe player context."""

    record: JsonDict
    context: PlayerContext


def require_engine(state: ReplayRuntimeState) -> GameEngine:
    """Return the replay's live engine; a missing one is a caller bug."""
    engine = get_game_engine(state)
    if engine is None:
        raise RuntimeError("Replay action diff has no replay loaded: no game engine")
    return engine


def require_archive(state: ReplayRuntimeState) -> ReplayArchive:
    """Return the loaded replay archive; a missing one is a caller bug."""
    if state.replay_data is None:
        raise RuntimeError("Replay action diff has no replay loaded: no replay archive")
    return state.replay_data


def replay_parsed_actions(state: ReplayRuntimeState) -> ParsedActions:
    """Read the archive's parsed rows; an absent key raises as it always did."""
    value = require_archive(state)["parsed_actions"]
    if not isinstance(value, ParsedActions):
        raise TypeError(f"parsed_actions is not indexable: {type(value).__name__}")
    return value


def to_json(value: object, label: str) -> JsonValue:
    """Copy a replay-sourced value into the JSON shape ``json.dumps`` writes."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [to_json(item, f"{label} entry") for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_json(item, f"{label}.{key}") for key, item in value.items()}
    raise TypeError(f"{label} is not JSON serializable: {type(value).__name__}")


def object_or_empty(value: JsonValue, label: str) -> JsonDict:
    """Read ``value or {}`` as an object; a truthy non-object still fails."""
    return as_dict(value, label) if value else {}


def list_or_empty(value: JsonValue, label: str) -> JsonList:
    """Read ``value or []`` as a list; a truthy non-list still fails."""
    return as_list(value, label) if value else []
