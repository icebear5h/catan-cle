"""Board and game-state dataclasses reconstructed from a Colonist replay."""

from dataclasses import dataclass, field

from data_pipeline.bootstrapping.replay_decoder._enums import (
    ActionState,
    BuildingType,
    PortType,
    ResourceType,
    TileType,
)


@dataclass
class HexTile:
    index: int
    x: int
    y: int
    tile_type: TileType
    dice_number: int

    def produces_resource(self) -> ResourceType | None:
        """Return resource type this tile produces, or None for desert"""
        if self.tile_type == TileType.DESERT:
            return None
        # tile_type 1-5 maps to resource 1-5
        return ResourceType(self.tile_type)


@dataclass
class Port:
    edge_id: int
    x: int
    y: int
    z: int
    port_type: PortType

    def trade_ratio(self) -> int:
        return 3 if self.port_type == PortType.GENERIC else 2


@dataclass
class Corner:
    index: int
    x: int
    y: int
    z: int
    owner: int | None = None
    building_type: BuildingType = BuildingType.NONE


@dataclass
class Edge:
    index: int
    x: int
    y: int
    z: int
    owner: int | None = None
    road_type: int = 0  # 1 = road


@dataclass
class PlayerState:
    color: int
    is_connected: bool = True
    resources: list[int] = field(default_factory=list)  # List of resource type ints
    dev_cards: list[int] = field(default_factory=list)
    dev_cards_used: list[int] = field(default_factory=list)
    victory_points: dict[str, int] = field(default_factory=dict)
    bank_trade_ratios: dict[int, int] = field(default_factory=dict)  # resource -> ratio
    settlements_remaining: int = 5
    cities_remaining: int = 4
    roads_remaining: int = 15

    def resource_count(self, resource: ResourceType) -> int:
        return self.resources.count(resource)

    def total_resources(self) -> int:
        return len(self.resources)


@dataclass
class GameState:
    # Board layout (static)
    tiles: dict[int, HexTile] = field(default_factory=dict)
    ports: dict[int, Port] = field(default_factory=dict)
    corners: dict[int, Corner] = field(default_factory=dict)
    edges: dict[int, Edge] = field(default_factory=dict)

    # Dynamic state
    players: dict[int, PlayerState] = field(default_factory=dict)
    current_player: int = 0
    action_state: ActionState = ActionState.SETUP_PLACE_SETTLEMENT
    completed_turns: int = 0
    robber_tile: int = 0
    dice_value: tuple[int, int] = (0, 0)

    # Bank state
    bank_resources: dict[int, int] = field(default_factory=dict)
    bank_dev_cards: list[int] = field(default_factory=list)

    # Longest road / largest army
    longest_road_player: int | None = None
    longest_road_length: int = 0
    largest_army_player: int | None = None
    largest_army_size: int = 0


__all__ = ["Corner", "Edge", "GameState", "HexTile", "PlayerState", "Port"]
