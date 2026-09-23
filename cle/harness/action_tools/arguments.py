"""Exact argument validation and actual-board spatial token resolution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Literal, TypeAlias, TypedDict, cast, overload

from cle.game_engine.models.coordinate_system import Coordinate
from cle.game_engine.models.enums import Action, FastResource
from cle.game_engine.models.player import Color
from cle.game_engine.trading import ResourceBundle, TradeOffer
from cle.harness import action_tools
from cle.players.contracts import PlayerContext

SpatialCategory: TypeAlias = Literal["tile", "node", "edge"]
SpatialValues: TypeAlias = dict[str, Coordinate] | dict[str, int] | dict[str, tuple[int, int]]
IndexedActions: TypeAlias = list[tuple[int, Action]]
ConfirmationPriority: TypeAlias = Literal["ANY"] | tuple[Color, ...]


class ChoiceParameters(TypedDict, total=False):
    confirm_if_accepted_by: ConfirmationPriority
    knight_destination: Coordinate
    discard_cards: tuple[str, ...]
    trade_offer: TradeOffer


def _require_arguments(
    arguments: Mapping[str, object], *required: str, optional: tuple[str, ...] = (),
) -> None:
    if set(arguments) - set(required) - set(optional) or set(required) - set(arguments):
        raise ValueError(
            f"Expected arguments {', '.join(required) or '(none)'}"
            + (f"; optional: {', '.join(optional)}" if optional else "")
        )


def _resource(value: object) -> FastResource:
    if not isinstance(value, str) or value.upper() not in action_tools.RESOURCE_NAMES:
        raise ValueError("Resource must be WOOD, BRICK, SHEEP, WHEAT, or ORE")
    return cast(FastResource, value.upper())


def _bundle(value: object) -> ResourceBundle:
    if not isinstance(value, dict):
        raise ValueError("Resources must be a named resource-count object")
    counts: dict[str, int] = {}
    for key, count in value.items():
        resource = action_tools._resource(key)
        if resource in counts:
            raise ValueError(f"Duplicate resource: {resource}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Named resource counts must be positive integers")
        counts[resource] = count
    return cast(ResourceBundle, tuple(counts.get(resource, 0) for resource in action_tools.RESOURCE_NAMES))


def _card_counts(cards: Iterable[str | None]) -> ResourceBundle:
    counts = Counter(cards)
    return cast(ResourceBundle, tuple(counts[resource] for resource in action_tools.RESOURCE_NAMES))


def _color(value: object) -> Color:
    if not isinstance(value, str):
        raise ValueError("Player color must be a string")
    try:
        return Color(value.upper())
    except ValueError as exc:
        raise ValueError(f"Unknown player color: {value!r}") from exc


def _offer_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("offer_id must be an exact nonempty offer ID string")
    return value


@overload
def _spatial_values(context: PlayerContext, category: Literal["tile"]) -> dict[str, Coordinate]: ...


@overload
def _spatial_values(context: PlayerContext, category: Literal["node"]) -> dict[str, int]: ...


@overload
def _spatial_values(
    context: PlayerContext, category: Literal["edge"],
) -> dict[str, tuple[int, int]]: ...


@overload
def _spatial_values(context: PlayerContext, category: str) -> SpatialValues: ...


def _spatial_values(context: PlayerContext, category: str) -> SpatialValues:
    board_map = context.observation.board_map
    if board_map is None:
        raise ValueError("Spatial tools require the observation's actual board map")
    if category == "tile":
        return {
            action_tools.tile_token(tile.id): coordinate
            for coordinate, tile in board_map.land_tiles.items()
            if coordinate != context.observation.robber_position
        }
    if category == "node":
        return {
            action_tools.node_token(node): node
            for tile in board_map.land_tiles.values()
            for node in tile.nodes.values()
        }
    return {
        action_tools.edge_token(edge): action_tools.canonical_edge(edge)
        for tile in board_map.land_tiles.values()
        for edge in tile.edges.values()
    }
