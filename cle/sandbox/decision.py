"""Shared perspective-safe decision-context construction."""

from __future__ import annotations

from typing import Any

from cle.players.contracts import PlayerContext
from game_engine.game import GameEngine
from game_engine.models.enums import Action, ActionType
from game_engine.models.player import Color


def build_decision_context(
    game_engine: GameEngine,
    actor: Color | None = None,
    advertised_actions: tuple[Action, ...] | None = None,
    *,
    context_revision: int | None = None,
) -> PlayerContext:
    """Build the one general player context used by live and replay agents."""
    actor = actor or game_engine.state.current_color()
    observation = game_engine.observe(actor)
    legal_actions = advertised_actions or tuple(game_engine.state.playable_actions)
    observation.valid_actions = list(legal_actions)
    if not legal_actions:
        raise ValueError(f"No legal actions are available for {actor}")
    revision = game_engine.revision if context_revision is None else context_revision
    return PlayerContext(
        context_id=f"{game_engine.id}:{revision}:{actor.value}",
        actor=actor,
        turn_number=game_engine.state.num_turns,
        phase=observation.current_phase,
        observation=observation,
        events=game_engine.project_game_events(actor),
        legal_actions=legal_actions,
        prompt_key=decision_prompt_key(observation, legal_actions),
    )


def decision_prompt_key(observation: Any, legal_actions: tuple[Action, ...]) -> str:
    """Select authored phase guidance from only visible state and exact actions."""
    action_types = {action.action_type for action in legal_actions}
    if observation.current_phase == "initial_placement":
        if ActionType.BUILD_SETTLEMENT in action_types:
            return (
                "initial_settlement_1"
                if len(observation.my_settlements) == 0
                else "initial_settlement_2"
            )
        if ActionType.BUILD_ROAD in action_types:
            return "initial_road"
        return "initial_placement"
    if observation.current_phase == "discarding" or ActionType.DISCARD in action_types:
        return "discarding"
    if (
        observation.current_phase == "moving_robber"
        or ActionType.MOVE_ROBBER in action_types
        or ActionType.STEAL in action_types
    ):
        return "robber"
    return "main_game"
