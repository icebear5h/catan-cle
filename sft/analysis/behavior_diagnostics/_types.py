"""Behavior-level evaluation summaries and checkpoint retention matrices."""

from __future__ import annotations

from dataclasses import dataclass

from sft.json_types import JsonDict as JsonDict


@dataclass(frozen=True)
class Behavior:
    """One stable, human-readable capability slice."""

    name: str
    description: str


BEHAVIORS = (
    Behavior("pair.positive", "occupied target on an adjacent-pair image"),
    Behavior("pair.road_positive", "occupied road target on an adjacent-pair image"),
    Behavior("pair.building_positive", "occupied building target on an adjacent-pair image"),
    Behavior("pair.empty", "all empty-location queries on an adjacent-pair image"),
    Behavior("pair.adjacent_empty", "empty location touching a pair member"),
    Behavior("pair.far_empty", "empty location far from both pair members"),
    Behavior("pair.localization", "piece-to-atlas localization on an adjacent-pair image"),
    Behavior("pair.edge_edge_positive", "occupied target on an edge-edge pair"),
    Behavior("single.positive", "occupied target on a single-piece image"),
    Behavior("single.road_positive", "occupied road target on a single-piece image"),
    Behavior("single.building_positive", "occupied building target on a single-piece image"),
    Behavior("single.empty", "all empty-location queries on a single-piece image"),
    Behavior("single.adjacent_empty", "empty location touching a single piece"),
    Behavior("single.far_empty", "empty location far from a single piece"),
    Behavior("single.localization", "piece-to-atlas localization on a single-piece image"),
    Behavior("tile.resource", "tile resource classification"),
    Behavior("tile.number", "tile number classification"),
    Behavior("tile.inverse", "tile description-to-atlas localization"),
    Behavior("full_board.node_occupancy", "node occupancy on production board renders"),
    Behavior("full_board.node_occupied", "occupied-node recall on production board renders"),
    Behavior("full_board.node_empty", "empty-node accuracy on production board renders"),
    Behavior("full_board.edge_owner", "edge owner on production board renders"),
    Behavior("full_board.road_occupied", "occupied-road recall on production board renders"),
    Behavior("full_board.edge_empty", "empty-edge accuracy on production board renders"),
    Behavior("full_board.tile_resource", "tile resource on production board renders"),
    Behavior("full_board.tile_number", "tile number on production board renders"),
    Behavior("full_board.tile_inverse", "inverse tile lookup on production board renders"),
    Behavior("full_board.tile_robber", "tile-level robber presence on production board renders"),
    Behavior("full_board.robber", "robber presence and localization on production board renders"),
    Behavior("full_board.port", "port classification on production board renders"),
    Behavior("full_board.inverse_node", "inverse node lookup on production board renders"),
    Behavior("full_board.inverse_edge", "inverse edge lookup on production board renders"),
    Behavior("full_board.inverse_port", "inverse port lookup on production board renders"),
    Behavior("full_board.spatial_grounding", "graph-relation queries on production board renders"),
)
BEHAVIOR_DESCRIPTIONS = {item.name: item.description for item in BEHAVIORS}
