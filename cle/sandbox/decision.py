"""Shared perspective-safe decision-context construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from cle.players.contracts import PlayerContext
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color


def build_decision_context(
    game_engine: GameEngine,
    actor: Color | None = None,
    advertised_actions: tuple[Action, ...] | None = None,
    *,
    context_revision: int | None = None,
    allow_terminal: bool = False,
) -> PlayerContext:
    """Build one player context, optionally previewing an ongoing source replay.

    allow_terminal bypasses only score-based termination; callers must establish
    that source rows remain. It never exposes another actor's implicit menu.
    """
    if not isinstance(allow_terminal, bool):
        raise ValueError("allow_terminal must be a boolean")
    terminal = game_engine.winning_color() is not None
    if terminal and not allow_terminal:
        raise ValueError("Cannot build a decision context for a terminal game")
    actor = game_engine.state.current_color() if actor is None else actor
    cutoff = game_engine.revision - 1
    observation = game_engine.observe(actor)
    if advertised_actions is not None:
        legal_actions = advertised_actions
    elif allow_terminal and terminal and actor == game_engine.state.current_color():
        legal_actions = tuple(game_engine.state.playable_actions)
    else:
        legal_actions = tuple(observation.valid_actions)
    if not legal_actions:
        raise ValueError(f"No legal actions are available for {actor}")
    if any(action.color != actor for action in legal_actions):
        raise ValueError(f"Advertised actions must belong to {actor}")
    observation.valid_actions = list(legal_actions)
    revision = game_engine.revision if context_revision is None else context_revision
    visible = tuple(event for event in game_engine.project_events(actor) if event.sequence <= cutoff)
    return deepcopy(PlayerContext(
        context_id=f"{game_engine.id}:{revision}:{actor.value}",
        actor=actor,
        turn_number=game_engine.state.num_turns,
        phase=observation.current_phase,
        observation=observation,
        events=tuple(event for event in visible if event.event_type != "MESSAGE_SENT"),
        legal_actions=legal_actions,
        prompt_key=decision_prompt_key(observation, legal_actions),
        recent_messages=tuple(
            event for event in visible if event.event_type == "MESSAGE_SENT"
        )[-game_engine.communication_limits.recent_message_window :],
        active_commitments=tuple(
            item for item in game_engine.active_commitments(actor)
            if item.created_sequence <= cutoff and item.source_message_sequence <= cutoff
        ),
        discard_count=(
            sum(observation.my_resources.values()) // 2
            if any(action.action_type == ActionType.DISCARD for action in legal_actions)
            else 0
        ),
        visible_through_sequence=cutoff,
        visible_messages=tuple(
            event for event in visible if event.event_type == "MESSAGE_SENT"
        ),
    ))


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
            return "initial_road_1" if len(observation.my_roads) == 0 else "initial_road_2"
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
