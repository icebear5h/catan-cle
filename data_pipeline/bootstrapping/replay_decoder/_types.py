"""Result shapes the decoder hands back to callers."""

from typing import TypeAlias, TypedDict

MoveDetail: TypeAlias = int | tuple[int, int]
Move: TypeAlias = dict[str, str | int]


class ActionInfo(TypedDict):
    """What a single replay event did, as `apply_event` reports it."""
    delta_s: float
    action_type: str | None
    player: int | None
    details: dict[str, MoveDetail]


class PlayerSnapshot(TypedDict):
    resources: list[int]
    victory_points: dict[str, int]
    bank_trade_ratios: dict[int, int]


class BuildingSnapshot(TypedDict):
    owner: int | None
    type: str


class RoadSnapshot(TypedDict):
    owner: int | None


class StateSnapshot(TypedDict):
    current_player: int
    action_state: str
    completed_turns: int
    dice: tuple[int, int]
    robber_tile: int
    players: dict[int, PlayerSnapshot]
    buildings: dict[int, BuildingSnapshot]
    roads: dict[int, RoadSnapshot]


class ReplayStep(TypedDict):
    event_idx: int
    state: StateSnapshot
    valid_moves: list[Move]
    action_taken: ActionInfo | None


class TileSnapshot(TypedDict):
    x: int
    y: int
    type: str
    dice_number: int


class PortSnapshot(TypedDict):
    x: int
    y: int
    z: int
    type: str
    ratio: int


class BoardState(TypedDict):
    tiles: dict[int, TileSnapshot]
    ports: dict[int, PortSnapshot]
    corner_tiles: dict[int, list[int]]


__all__ = [
    "ActionInfo",
    "BoardState",
    "BuildingSnapshot",
    "Move",
    "MoveDetail",
    "PlayerSnapshot",
    "PortSnapshot",
    "ReplayStep",
    "RoadSnapshot",
    "StateSnapshot",
    "TileSnapshot",
]
