"""Per-decision state features derived from a live game state."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, Mapping, Sequence

from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState
from evals.decision_buckets.models import DecisionStateFeatures
from evals.decision_buckets.taxonomy import (
    CONSEQUENTIAL_ACTIONS,
    EARLY_ROUNDS,
    LATE_PUBLIC_VP,
)
from evals.json_types import JsonValue


def decision_state_features(
    game_state: GameState,
    actor_color: Color,
    legal_actions: Iterable[object],
) -> dict[str, JsonValue]:
    """Capture only state evidence needed for deterministic scenario tagging."""
    colors = tuple(game_state.colors)
    actor_index = game_state.color_to_index[actor_color]
    actor_key = f"P{actor_index}"
    public_vp = {
        _color_name(color): int(
            game_state.player_state.get(
                f"P{game_state.color_to_index[color]}_VICTORY_POINTS",
                0,
            )
        )
        for color in colors
    }
    opponent_vp = [
        score for color, score in public_vp.items() if color != _color_name(actor_color)
    ]
    action_types = tuple(sorted({_action_type_name(action) for action in legal_actions}))
    completed_turns = int(game_state.num_turns)
    payload = DecisionStateFeatures(
        completed_turns=completed_turns,
        table_round=(completed_turns // len(colors)) + 1,
        player_count=len(colors),
        actor_color=_color_name(actor_color),
        turn_owner_color=_color_name(colors[game_state.current_turn_index]),
        actor_is_turn_owner=game_state.current_turn_index == actor_index,
        actor_public_vp=int(game_state.player_state.get(f"{actor_key}_VICTORY_POINTS", 0)),
        actor_actual_vp=int(
            game_state.player_state.get(f"{actor_key}_ACTUAL_VICTORY_POINTS", 0)
        ),
        opponent_max_public_vp=max(opponent_vp, default=0),
        max_public_vp=max(public_vp.values(), default=0),
        public_vp=public_vp,
        actor_longest_road_length=int(
            game_state.player_state.get(f"{actor_key}_LONGEST_ROAD_LENGTH", 0)
        ),
        actor_played_knights=int(
            game_state.player_state.get(f"{actor_key}_PLAYED_KNIGHT", 0)
        ),
        actor_has_longest_road=bool(
            game_state.player_state.get(f"{actor_key}_HAS_ROAD", False)
        ),
        actor_has_largest_army=bool(
            game_state.player_state.get(f"{actor_key}_HAS_ARMY", False)
        ),
        available_action_types=action_types,
    )
    return payload.model_dump(mode="json", by_alias=True)


def _coerce_features(raw: object) -> DecisionStateFeatures | None:
    if not isinstance(raw, Mapping):
        return None
    return DecisionStateFeatures.model_validate(dict(raw))


def _decision_stage(
    phase: str,
    features: DecisionStateFeatures | None,
) -> Literal["setup", "early", "mid", "late", "unknown"]:
    if phase in {"initial_placement", "initial_settlement_1", "initial_settlement_2", "initial_road"}:
        return "setup"
    if features is None:
        return "unknown"
    if features.max_public_vp >= LATE_PUBLIC_VP:
        return "late"
    if features.completed_turns < EARLY_ROUNDS * features.player_count:
        return "early"
    return "mid"


def _consequential_counts_by_turn(
    records: Sequence[Mapping[str, object]],
) -> Counter[tuple[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    seen: set[tuple[tuple[str, int], int]] = set()
    for record in records:
        features = _coerce_features(record.get("state_features"))
        turn_key = _turn_key(record, features)
        replay_index = record.get("replay_index")
        action_type = _record_action_type(record)
        if turn_key is None or not isinstance(replay_index, int):
            continue
        if action_type not in CONSEQUENTIAL_ACTIONS:
            continue
        identity = (turn_key, replay_index)
        if identity in seen:
            continue
        seen.add(identity)
        counts[turn_key] += 1
    return counts


def _turn_key(
    record: Mapping[str, object],
    features: DecisionStateFeatures | None,
) -> tuple[str, int] | None:
    actor = record.get("actor")
    actor_color = actor.get("engine_color") if isinstance(actor, Mapping) else None
    if (
        not isinstance(actor_color, str)
        or features is None
        or not features.actor_is_turn_owner
    ):
        return None
    return actor_color, features.completed_turns


def _is_award_race(
    action_type: str,
    available_types: set[str],
    features: DecisionStateFeatures | None,
) -> bool:
    if features is None:
        return False
    road_relevant = action_type in {"BUILD_ROAD", "PLAY_ROAD_BUILDING"} or bool(
        available_types.intersection({"BUILD_ROAD", "PLAY_ROAD_BUILDING"})
    )
    army_relevant = action_type == "PLAY_KNIGHT_CARD" or "PLAY_KNIGHT_CARD" in available_types
    return (
        road_relevant
        and (features.actor_longest_road_length >= 4 or features.actor_has_longest_road)
    ) or (
        army_relevant
        and (features.actor_played_knights >= 2 or features.actor_has_largest_army)
    )


def _record_action_type(record: Mapping[str, object]) -> str:
    return str(record.get("effective_action_type") or record.get("source_action_type") or "")


def _action_type_name(action: object) -> str:
    action_type = getattr(action, "action_type", None)
    value = getattr(action_type, "value", None)
    if isinstance(value, str):
        return value
    return str(action_type).replace("ActionType.", "").replace("AT.", "")


def _color_name(color: object) -> str:
    name = getattr(color, "name", None)
    if isinstance(name, str):
        return name
    value = getattr(color, "value", None)
    return str(value if value is not None else color)
