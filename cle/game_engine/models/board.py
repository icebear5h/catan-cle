import pickle
import copy
from collections import defaultdict
from typing import Any, Set, Dict, Tuple, List
import functools

import networkx as nx  # type: ignore

from cle.game_engine.models.player import Color
from cle.game_engine.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    NUM_NODES,
    CatanMap,
    NodeId,
)
from cle.game_engine.models.enums import FastBuildingType, SETTLEMENT, CITY


# Used to find relationships between nodes and edges
base_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
mini_map = CatanMap.from_template(MINI_MAP_TEMPLATE)
STATIC_GRAPH = nx.Graph()
for tile in base_map.tiles.values():
    STATIC_GRAPH.add_nodes_from(tile.nodes.values())
    STATIC_GRAPH.add_edges_from(tile.edges.values())


@functools.lru_cache(1)
def get_node_distances():
    return nx.floyd_warshall(STATIC_GRAPH)


@functools.lru_cache(3)  # None, range(54), range(24)
def get_edges(land_nodes=None):
    return list(STATIC_GRAPH.subgraph(land_nodes or range(NUM_NODES)).edges())


class Board:
    """Encapsulates all state information regarding the board.

    Attributes:
        buildings (Dict[NodeId, Tuple[Color, FastBuildingType]]): Mapping from
            node id to building (if there is a building there).
        roads (Dict[EdgeId, Color]): Mapping from edge
            to Color (if there is a road there). Contains inverted
            edges as well for ease of querying.
        connected_components (Dict[Color, List[Set[NodeId]]]): Cache
            datastructure to speed up maintaining longest road computation.
            To be queried by Color. Value is a list of node sets.
        board_buildable_ids (Set[NodeId]): Cache of buildable node ids in board.
        road_color (Color): Color of player with longest road.
        road_length (int): Number of roads of longest road
        robber_coordinate (Coordinate): Coordinate where robber is.
    """

    def __init__(self, catan_map=None, initialize=True):
        self.buildable_subgraph: Any = None
        self.buildable_edges_cache = {}
        self.player_port_resources_cache = {}
        if initialize:
            self.map: CatanMap = catan_map or CatanMap.from_template(
                BASE_MAP_TEMPLATE
            )  # Static State (no need to copy)

            self.buildings: Dict[NodeId, Tuple[Color, FastBuildingType]] = dict()
            self.roads = dict()  # (node_id, node_id) => color

            # color => int{}[] (list of node_id sets) one per component
            #   nodes in sets are incidental (might not be owned by player)
            self.connected_components: Any = defaultdict(list)
            self.board_buildable_ids = set(self.map.land_nodes)
            self.road_lengths = defaultdict(int)
            self.road_color = None
            self.road_length = 0

            # assumes there is at least one desert:
            self.robber_coordinate = filter(
                lambda coordinate: self.map.land_tiles[coordinate].resource is None,
                self.map.land_tiles.keys(),
            ).__next__()

            # Cache buildable subgraph
            self.buildable_subgraph = STATIC_GRAPH.subgraph(self.map.land_nodes)

    def build_settlement(self, color, node_id, initial_build_phase=False):
        """Adds a settlement, and ensures is a valid place to build.

        Args:
            color (Color): player's color
            node_id (int): where to build
            initial_build_phase (bool, optional):
                Whether this is part of initial building phase, so as to skip
                connectedness validation. Defaults to True.
        """
        buildable = self.buildable_node_ids(
            color, initial_build_phase=initial_build_phase
        )
        if node_id not in buildable:
            raise ValueError(
                "Invalid Settlement Placement: not connected and not initial-placement"
            )

        if node_id in self.buildings:
            raise ValueError("Invalid Settlement Placement: a building exists there")

        self.buildings[node_id] = (color, SETTLEMENT)

        self.board_buildable_ids.discard(node_id)
        for n in STATIC_GRAPH.neighbors(node_id):
            self.board_buildable_ids.discard(n)

        self.player_port_resources_cache = {}  # Reset port resources
        return self.recompute_road_state()

    def dfs_walk(self, node_id, color):
        """Generates set of nodes that are "connected" to given node.

        Args:
            node_id (int): Where to start search/walk.
            color (Color): Player color asking

        Returns:
            Set[int]: Nodes that are "connected" to this one
                by roads of the color player.
        """
        agenda = [node_id]  # assuming node_id is owned.
        visited = set()

        while len(agenda) != 0:
            n = agenda.pop()
            visited.add(n)

            if self.is_enemy_node(n, color):
                continue  # end of the road

            neighbors = [v for v in STATIC_GRAPH.neighbors(n) if v not in visited]
            expandable = [v for v in neighbors if self.roads.get((n, v), None) == color]
            agenda.extend(expandable)

        return visited

    def _get_connected_component_index(self, node_id, color):
        for i, component in enumerate(self.connected_components[color]):
            if node_id in component:
                return i

    def build_road(self, color, edge):
        buildable = self.buildable_edges(color)
        inverted_edge = (edge[1], edge[0])
        if edge not in buildable and inverted_edge not in buildable:
            raise ValueError("Invalid Road Placement")

        self.roads[edge] = color
        self.roads[inverted_edge] = color

        return self.recompute_road_state()

    def recompute_road_state(self):
        """Rebuild road components, edge-simple lengths, and award ownership.

        Enemy vertices can be endpoints shared by multiple components, but
        never join roads through that vertex. Also used after replay overlays.
        """
        previous_road_color = self.road_color
        colors = (
            set(self.connected_components)
            | set(self.road_lengths)
            | set(self.roads.values())
            | {owner for owner, _ in self.buildings.values()}
        )
        components = defaultdict(list)
        lengths = defaultdict(int)
        for color in sorted(colors, key=lambda item: item.value):
            remaining = {
                tuple(sorted(edge)) for edge, owner in self.roads.items() if owner == color
            }
            road_nodes = {node for edge in remaining for node in edge}
            while remaining:
                agenda = [min(remaining)]
                nodes = set()
                while agenda:
                    edge = agenda.pop()
                    if edge not in remaining:
                        continue
                    remaining.remove(edge)
                    nodes.update(edge)
                    for node in edge:
                        if self.is_enemy_node(node, color):
                            continue
                        for neighbor in STATIC_GRAPH.neighbors(node):
                            candidate = tuple(sorted((node, neighbor)))
                            if candidate in remaining:
                                agenda.append(candidate)
                components[color].append(nodes)
            components[color].extend(
                {node}
                for node, (owner, _) in self.buildings.items()
                if owner == color and node not in road_nodes
            )
            lengths[color] = max(
                (len(longest_acyclic_path(self, nodes, color)) for nodes in components[color]),
                default=0,
            )

        self.connected_components = components
        self.road_lengths = lengths
        self.road_length = max(lengths.values(), default=0)
        leaders = [
            color for color, length in lengths.items() if length == self.road_length and length >= 5
        ]
        if previous_road_color in leaders:
            self.road_color = previous_road_color
        else:
            self.road_color = leaders[0] if len(leaders) == 1 else None
        self.buildable_edges_cache = {}
        return previous_road_color, self.road_color, self.road_lengths

    def build_city(self, color, node_id):
        building = self.buildings.get(node_id, None)
        if building is None or building[0] != color or building[1] != SETTLEMENT:
            raise ValueError("Invalid City Placement: no player settlement there")

        self.buildings[node_id] = (color, CITY)

    def buildable_node_ids(self, color: Color, initial_build_phase=False):
        if initial_build_phase:
            return sorted(list(self.board_buildable_ids))

        subgraphs = self.find_connected_components(color)
        nodes = set().union(*subgraphs)
        return sorted(list(nodes.intersection(self.board_buildable_ids)))

    def buildable_edges(self, color: Color):
        """List of (n1,n2) tuples. Edges are in n1 < n2 order."""
        if color in self.buildable_edges_cache:
            return self.buildable_edges_cache[color]

        expandable = set()

        expandable_nodes = {
            node
            for component in self.connected_components[color]
            for node in component
            if not self.is_enemy_node(node, color)
        }

        candidate_edges = self.buildable_subgraph.edges(expandable_nodes)
        for edge in candidate_edges:
            if self.get_edge_color(edge) is None:
                expandable.add(tuple(sorted(edge)))

        self.buildable_edges_cache[color] = list(expandable)
        return self.buildable_edges_cache[color]

    def get_player_port_resources(self, color):
        """Yields resources (None for 3:1) of ports owned by color"""
        if color in self.player_port_resources_cache:
            return self.player_port_resources_cache[color]

        resources = set()
        for resource, node_ids in self.map.port_nodes.items():
            if any(self.is_friendly_node(node_id, color) for node_id in node_ids):
                resources.add(resource)

        self.player_port_resources_cache[color] = resources
        return resources

    def find_connected_components(self, color: Color):
        """
        Returns:
            nx.Graph[]: connected subgraphs. subgraphs
                might include nodes that color doesnt own (on the way and on ends),
                just to make it is "closed" and easier for buildable_nodes to operate.
        """
        return self.connected_components[color]

    def continuous_roads_by_player(self, color: Color):
        paths = []
        components = self.find_connected_components(color)
        for component in components:
            paths.append(longest_acyclic_path(self, component, color))
        return paths

    def copy(self):
        board = Board(self.map, initialize=False)
        board.map = self.map  # reuse since its immutable
        board.buildings = self.buildings.copy()
        board.roads = self.roads.copy()
        board.connected_components = pickle.loads(
            pickle.dumps(self.connected_components)
        )
        board.board_buildable_ids = self.board_buildable_ids.copy()
        board.road_lengths = self.road_lengths.copy()
        board.road_color = self.road_color
        board.road_length = self.road_length

        board.robber_coordinate = self.robber_coordinate
        board.buildable_subgraph = self.buildable_subgraph
        board.buildable_edges_cache = copy.deepcopy(self.buildable_edges_cache)
        board.player_port_resources_cache = copy.deepcopy(
            self.player_port_resources_cache
        )
        return board

    # ===== Helper functions
    def get_node_color(self, node_id):
        # using try-except instead of .get for performance
        try:
            return self.buildings[node_id][0]
        except KeyError:
            return None

    def get_edge_color(self, edge):
        # using try-except instead of .get for performance
        try:
            return self.roads[edge]
        except KeyError:
            return None

    def is_enemy_node(self, node_id, color):
        node_color = self.get_node_color(node_id)
        return node_color is not None and node_color != color

    def is_enemy_road(self, edge, color):
        edge_color = self.get_edge_color(edge)
        return edge_color is not None and self.get_edge_color(edge) != color

    def is_friendly_node(self, node_id, color):
        return self.get_node_color(node_id) == color

    def is_friendly_road(self, edge, color):
        return self.get_edge_color(edge) == color


def longest_acyclic_path(board: Board, node_set: Set[int], color: Color):
    """Return a longest trail: vertices may repeat, undirected edges may not."""
    longest = []
    for start_node in sorted(node_set):
        agenda: List[Tuple[int, Any]] = [(start_node, [])]
        while len(agenda) > 0:
            node, path_thus_far = agenda.pop()
            if len(path_thus_far) > len(longest):
                longest = path_thus_far
            if path_thus_far and board.is_enemy_node(node, color):
                continue
            for neighbor_node in STATIC_GRAPH.neighbors(node):
                if neighbor_node not in node_set:
                    continue
                edge = tuple(sorted((node, neighbor_node)))
                if board.is_friendly_road(edge, color) and edge not in path_thus_far:
                    agenda.append((neighbor_node, path_thus_far + [edge]))
    return longest
