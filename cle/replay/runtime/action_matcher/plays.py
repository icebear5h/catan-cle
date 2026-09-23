"""Match Colonist development-card, robber and steal rows."""

from __future__ import annotations

from cle.game_engine.models.enums import Action, ActionType
from cle.replay.colonist.constants import COLONIST_RESOURCE
from cle.replay.contracts import mapping_field

from .context import (
    UNDECIDED,
    Decision,
    MatchContext,
    colonist_xy_to_engine_coord,
    decide,
    engine_seat,
)

__all__ = ["PLAY_MATCHERS"]


def _buy_development_card(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "BUY_DEVELOPMENT_CARD" not in action_str:
        return UNDECIDED
    colonist_player = ctx.action_hint.get("player")
    if colonist_player is not None and ctx.replay_data is not None and ctx.game is not None:
        expected_color = engine_seat(ctx.game, ctx.replay_data, colonist_player)
        if expected_color is not None and action.color != expected_color:
            return UNDECIDED

    card_type = ctx.action_hint.get("card_type")
    if card_type and card_type != "UNKNOWN":
        return decide(Action(action.color, ActionType.BUY_DEVELOPMENT_CARD, card_type))
    return decide(action)


def _play_knight(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "PLAY_KNIGHT_CARD" not in action_str:
        return UNDECIDED
    return decide(action)


def _play_road_building(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "PLAY_ROAD_BUILDING" not in action_str:
        return UNDECIDED
    return decide(action)


def _play_year_of_plenty(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "PLAY_YEAR_OF_PLENTY" not in action_str:
        return UNDECIDED
    return decide(None)


def _play_monopoly(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "PLAY_MONOPOLY" not in action_str:
        return UNDECIDED
    return decide(None)


def _year_of_plenty_resources(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "PLAY_YEAR_OF_PLENTY" not in action_str:
        return UNDECIDED
    resources = ctx.action_hint.get("resources", [])
    action_resources = action.value if hasattr(action, "value") else None
    if len(resources) >= 2:
        if action_resources and isinstance(action_resources, tuple):
            if action_resources[0] == resources[0] and action_resources[1] == resources[1]:
                return decide(action)
    elif len(resources) == 1:
        if action_resources and isinstance(action_resources, tuple):
            if action_resources[0] == resources[0]:
                return decide(action)
    return UNDECIDED


def _monopoly_resource(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "PLAY_MONOPOLY" not in action_str:
        return UNDECIDED
    resource = ctx.action_hint.get("resource")
    if resource:
        action_resource = action.value if hasattr(action, "value") else None
        if action_resource == resource:
            return decide(action)
    return UNDECIDED


def _move_robber(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "MOVE_ROBBER" not in action_str:
        return UNDECIDED
    tile_info = mapping_field(ctx.action_hint, "tile_info")
    colonist_resource_type = tile_info.get("resourceType")
    colonist_dice_number = tile_info.get("diceNumber")
    colonist_x = tile_info.get("x")
    colonist_y = tile_info.get("y")

    target_coord = None
    if isinstance(colonist_x, int) and isinstance(colonist_y, int):
        target_coord = colonist_xy_to_engine_coord(colonist_x, colonist_y)

    target_resource = (
        COLONIST_RESOURCE.get(colonist_resource_type)
        if isinstance(colonist_resource_type, int)
        else None
    )

    action_coord = action.value if hasattr(action, "value") else None
    if action_coord and ctx.game:
        if target_coord is not None:
            if action_coord == target_coord:
                return decide(action)
            return UNDECIDED

        engine_tile = ctx.game.state.board.map.land_tiles.get(action_coord)
        if engine_tile:
            engine_resource = engine_tile.resource
            engine_number = engine_tile.number

            if target_resource is None:
                if engine_resource is None:
                    return decide(action)
            elif engine_resource == target_resource and engine_number == colonist_dice_number:
                return decide(action)
    return UNDECIDED


def _steal(action: Action, action_str: str, ctx: MatchContext) -> Decision:
    if "STEAL" not in action_str:
        return UNDECIDED
    victim = ctx.action_hint.get("victim")
    stolen_resource = ctx.action_hint.get("stolen_resource")
    action_value = action.value if hasattr(action, "value") else None

    if action_value and isinstance(action_value, tuple) and len(action_value) >= 1:
        action_victim = action_value[0]

        if action_victim is not None and victim is not None and ctx.replay_data:
            try:
                expected_color = engine_seat(ctx.engine, ctx.replay_data, victim)
                if expected_color is not None and action_victim == expected_color:
                    if stolen_resource:
                        return decide(
                            Action(
                                action.color,
                                action.action_type,
                                (action_victim, stolen_resource),
                            )
                        )
                    return decide(action)
            except (IndexError, TypeError):
                pass
    return UNDECIDED


PLAY_MATCHERS = {
    "BUY_DEVELOPMENT_CARD": _buy_development_card,
    "PLAY_KNIGHT_CARD": _play_knight,
    "PLAY_ROAD_BUILDING": _play_road_building,
    "PLAY_YEAR_OF_PLENTY": _play_year_of_plenty,
    "PLAY_MONOPOLY": _play_monopoly,
    "YEAR_OF_PLENTY_RESOURCES": _year_of_plenty_resources,
    "MONOPOLY_RESOURCE": _monopoly_resource,
    "MOVE_ROBBER": _move_robber,
    "STEAL": _steal,
}
