"""Direct execution for discards, steals and robber moves."""

from __future__ import annotations

from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.replay.colonist.constants import ENGINE_RESOURCES
from cle.replay.contracts import ReplayPayload, mapping_field
from cle.replay.runtime.action_matcher import _colonist_xy_to_engine_coord
from cle.replay.runtime.audit import record_replay_issue

from ..context import engine_of, record_forced_overlay, seat_index, seating_map
from .outcome import DirectContext

__all__ = ["discard", "move_robber", "steal"]


def discard(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    game = engine_of(state)
    cards = ctx.action_hint.get("cards")
    colonist_player = ctx.action_hint.get("player")
    player_idx = seat_index(seating_map(state), colonist_player)

    if player_idx is not None and cards:
        discarded: list[str] = []
        for i, count in enumerate(cards):
            discarded.extend([ENGINE_RESOURCES[i]] * count)

        player_color = game.state.colors[player_idx]
        print(f"[Replay] Executing DISCARD directly: player={player_color}, cards={discarded}")
        game.state.current_player_index = player_idx
        discard_action = Action(player_color, ActionType.DISCARD, discarded)
        game.step(discard_action, force=True)
        return ctx.finish("ok", 1)

    print(f"[Replay] Skipping DISCARD: player={colonist_player}, cards={cards}")
    return ctx.finish("skipped", 0, "Skipped DISCARD (invalid params)")


def steal(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    game = engine_of(state)
    action_hint = ctx.action_hint
    thief = action_hint.get("player")
    victim = action_hint.get("victim")
    stolen_resource = action_hint.get("stolen_resource")
    seating = seating_map(state)
    thief_idx = seat_index(seating, thief)
    victim_idx = seat_index(seating, victim)

    if thief_idx is not None and victim_idx is not None and stolen_resource is not None:
        thief_color = game.state.colors[thief_idx]
        victim_color = game.state.colors[victim_idx]
        print(
            f"[Replay] Executing STEAL directly: "
            f"{thief_color} steals {stolen_resource} from {victim_color}"
        )
        game.state.current_player_index = thief_idx
        game.state.current_turn_index = thief_idx
        steal_action = Action(thief_color, ActionType.STEAL, (victim_color, stolen_resource))
        game.step(steal_action, force=True)
        return ctx.finish("ok", 1)

    print(f"[Replay] Skipping STEAL: thief={thief}, victim={victim}, resource={stolen_resource}")
    return ctx.finish("skipped", 0, "Skipped STEAL (invalid params)")


def move_robber(ctx: DirectContext) -> ReplayPayload:
    state = ctx.state
    game = engine_of(state)
    action_hint = ctx.action_hint
    tile_info = mapping_field(action_hint, "tile_info")
    colonist_player = action_hint.get("player")
    player_idx = seat_index(seating_map(state), colonist_player)
    x = tile_info.get("x")
    y = tile_info.get("y")

    if player_idx is not None and isinstance(x, int) and isinstance(y, int):
        player_color = game.state.colors[player_idx]
        target_coord = _colonist_xy_to_engine_coord(x, y)
        if (
            target_coord == game.state.board.robber_coordinate
            and game.state.current_prompt != ActionPrompt.MOVE_ROBBER
        ):
            record_replay_issue(
                state,
                kind="observed_replay_state",
                action_hint=action_hint,
                message="Observed unchanged robber location from Colonist replay",
                severity="info",
                details={"target_coord": target_coord, "tile_info": tile_info},
            )
            return ctx.finish(
                "already_satisfied",
                0,
                "Robber location already matches Colonist replay",
            )

        print(
            f"[Replay] Forcing MOVE_ROBBER directly: "
            f"player={player_color}, coord={target_coord}"
        )
        robber_action = Action(player_color, ActionType.MOVE_ROBBER, target_coord)
        game.step(robber_action, force=True)
        record_forced_overlay(
            state,
            action_hint,
            "Forced MOVE_ROBBER from Colonist tile coordinate",
            details={"target_coord": target_coord, "tile_info": tile_info},
        )
        return ctx.finish("ok", 1, "Forced MOVE_ROBBER from Colonist replay")

    print(
        f"[Replay] Skipping MOVE_ROBBER: player={colonist_player}, "
        f"tile_info={tile_info}"
    )
    return ctx.finish("skipped", 0, "Skipped MOVE_ROBBER (invalid params)")
