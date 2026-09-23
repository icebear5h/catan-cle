"""Board state plus the static node/edge graph shared by every map.

``Board`` keeps ``cle.game_engine.models.board`` as its module path so pickled
snapshots written before the package split still resolve.
"""

from cle.game_engine.models.board.core import Board
from cle.game_engine.models.board.graph import (
    STATIC_GRAPH,
    NodeGraph,
    base_map,
    get_edges,
    get_node_distances,
    mini_map,
    sorted_edge,
)
from cle.game_engine.models.board.roads import (
    RoadComponents,
    RoadState,
    longest_acyclic_path,
    road_components_and_lengths,
)
from cle.game_engine.models.enums import CITY, SETTLEMENT, FastBuildingType
from cle.game_engine.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    NUM_NODES,
    CatanMap,
    NodeId,
)
from cle.game_engine.models.player import Color

__all__ = [
    "BASE_MAP_TEMPLATE",
    "CITY",
    "MINI_MAP_TEMPLATE",
    "NUM_NODES",
    "SETTLEMENT",
    "STATIC_GRAPH",
    "Board",
    "CatanMap",
    "Color",
    "FastBuildingType",
    "NodeGraph",
    "NodeId",
    "RoadComponents",
    "RoadState",
    "base_map",
    "get_edges",
    "get_node_distances",
    "longest_acyclic_path",
    "mini_map",
    "road_components_and_lengths",
    "sorted_edge",
]
