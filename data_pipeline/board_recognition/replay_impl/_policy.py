"""Canonical action ordering and the weighted datagen policy."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from enum import Enum

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from data_pipeline.board_recognition.replay_impl._config import (
    TRADE_ACTIONS,
    ReplayDatasetBuildError,
)
from data_pipeline.json_types import JsonValue


def _canonical_payload_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _canonical_action_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _canonical_action_value(value.value)
    if isinstance(value, (list, tuple)):
        return [_canonical_action_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_action_value(item) for item in value]
        return sorted(normalized, key=_canonical_payload_bytes)
    if isinstance(value, dict):
        return {
            str(key): _canonical_action_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    to_payload = getattr(value, "to_payload", None)
    if callable(to_payload):
        return _canonical_action_value(to_payload())
    raise TypeError(f"unsupported legal-action value for canonical ordering: {value!r}")


def legal_action_sort_key(action: Action) -> tuple[str, str, bytes]:
    return (
        action.color.value,
        action.action_type.value,
        _canonical_payload_bytes(_canonical_action_value(action.value)),
    )


def _weighted_choice(
    actions: Sequence[Action],
    policy_rng: random.Random,
) -> Action:
    weighted = []
    weights = {
        ActionType.BUILD_CITY: 20,
        ActionType.BUILD_SETTLEMENT: 8,
        ActionType.BUILD_ROAD: 5,
        ActionType.BUY_DEVELOPMENT_CARD: 2,
        ActionType.MARITIME_TRADE: 2,
        ActionType.END_TURN: 1,
    }
    for action in actions:
        weighted.extend([action] * weights.get(action.action_type, 1))
    return policy_rng.choice(weighted)


def choose_legal_datagen_action(engine: GameEngine, policy_rng: random.Random) -> Action:
    """Choose one advertised action without force or handcrafted state mutation."""

    actions = sorted(
        (
            action
            for action in engine.state.playable_actions
            if action.action_type not in TRADE_ACTIONS
        ),
        key=legal_action_sort_key,
    )
    if not actions:
        raise ReplayDatasetBuildError("engine datagen policy has no non-domestic legal action")
    prompt = engine.state.current_prompt
    if prompt != ActionPrompt.PLAY_TURN or engine.state.is_road_building:
        return policy_rng.choice(actions)
    roll = next((action for action in actions if action.action_type == ActionType.ROLL), None)
    if roll is not None:
        pre_roll_development = [
            action
            for action in actions
            if action.action_type
            in {
                ActionType.PLAY_KNIGHT_CARD,
                ActionType.PLAY_YEAR_OF_PLENTY,
                ActionType.PLAY_MONOPOLY,
                ActionType.PLAY_ROAD_BUILDING,
            }
        ]
        if pre_roll_development and policy_rng.random() < 0.15:
            return policy_rng.choice(pre_roll_development)
        return roll
    return _weighted_choice(actions, policy_rng)

