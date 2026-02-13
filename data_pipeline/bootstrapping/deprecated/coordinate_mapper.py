#!/usr/bin/env python3
"""
Coordinate Mapper: Maps Colonist.io coordinates to our Catanatron engine.

Strategy:
1. Initialize our engine with a known board (tournament map with fixed tile positions)
2. For each node, record which tiles (by resource+number) it touches
3. Parse Colonist initial state to get their tile layout
4. Match nodes by their adjacent tiles' signatures
5. Build bidirectional mapping

This works because each node has a unique "signature" based on its adjacent tiles.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.models.map import (
    CatanMap,
    BASE_MAP_TEMPLATE,
    LandTile,
    NodeId,
    EdgeId,
    Coordinate,
    NUM_NODES,
    NUM_EDGES,
)
from engine.models.enums import FastResource, WOOD, BRICK, SHEEP, WHEAT, ORE


# Colonist resource enum -> our FastResource
COLONIST_TO_ENGINE_RESOURCE = {
    1: WOOD,
    2: BRICK,
    3: SHEEP,
    4: WHEAT,
    5: ORE,
    0: None,  # Desert
}

# Reverse mapping
ENGINE_TO_COLONIST_RESOURCE = {v: k for k, v in COLONIST_TO_ENGINE_RESOURCE.items()}


@dataclass
class TileSignature:
    """Unique signature for a tile based on resource and number."""
    resource: Optional[Any]  # FastResource or string
    number: Optional[int]

    def __hash__(self):
        return hash((str(self.resource), self.number))

    def __eq__(self, other):
        return str(self.resource) == str(other.resource) and self.number == other.number

    def __repr__(self):
        r = str(self.resource) if self.resource else "DESERT"
        return f"{r}:{self.number}"


@dataclass
class NodeSignature:
    """
    Signature for a node based on its adjacent tiles.

    A node can touch up to 3 tiles. The signature is the sorted set of
    tile signatures, which should be unique for each node on the board.
    """
    tiles: Tuple[TileSignature, ...]

    def __hash__(self):
        return hash(self.tiles)

    def __eq__(self, other):
        return self.tiles == other.tiles

    def __repr__(self):
        return f"Node({', '.join(str(t) for t in self.tiles)})"


@dataclass
class CoordinateMapping:
    """Complete coordinate mapping between Colonist and our engine."""
    # Colonist node ID -> our node ID
    colonist_to_engine_node: Dict[int, int] = field(default_factory=dict)

    # Our node ID -> Colonist node ID
    engine_to_colonist_node: Dict[int, int] = field(default_factory=dict)

    # Colonist edge ID -> our edge tuple (node1, node2)
    colonist_to_engine_edge: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Our edge tuple -> Colonist edge ID
    engine_to_colonist_edge: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Colonist tile ID -> our tile ID
    colonist_to_engine_tile: Dict[int, int] = field(default_factory=dict)

    # Mapping confidence (0-1)
    confidence: float = 0.0

    def save(self, path: str):
        """Save mapping to JSON."""
        data = {
            "colonist_to_engine_node": self.colonist_to_engine_node,
            "engine_to_colonist_node": self.engine_to_colonist_node,
            "colonist_to_engine_edge": {
                k: list(v) for k, v in self.colonist_to_engine_edge.items()
            },
            "colonist_to_engine_tile": self.colonist_to_engine_tile,
            "confidence": self.confidence,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CoordinateMapping":
        """Load mapping from JSON."""
        with open(path) as f:
            data = json.load(f)
        mapping = cls()
        mapping.colonist_to_engine_node = {
            int(k): v for k, v in data["colonist_to_engine_node"].items()
        }
        mapping.engine_to_colonist_node = {
            int(k): v for k, v in data["engine_to_colonist_node"].items()
        }
        mapping.colonist_to_engine_edge = {
            int(k): tuple(v) for k, v in data["colonist_to_engine_edge"].items()
        }
        mapping.colonist_to_engine_tile = {
            int(k): v for k, v in data.get("colonist_to_engine_tile", {}).items()
        }
        mapping.confidence = data.get("confidence", 0.0)
        return mapping


def get_engine_node_signatures(catan_map: CatanMap) -> Dict[NodeId, NodeSignature]:
    """
    Get signature for each node in our engine's map.

    Returns dict mapping node_id -> NodeSignature
    """
    signatures = {}

    for node_id in catan_map.land_nodes:
        # Get adjacent tiles for this node
        adjacent = catan_map.adjacent_tiles.get(node_id, [])

        tile_sigs = []
        for tile in adjacent:
            sig = TileSignature(tile.resource, tile.number)
            tile_sigs.append(sig)

        # Sort for consistent ordering
        tile_sigs.sort(key=lambda s: (str(s.resource), s.number or 0))
        signatures[node_id] = NodeSignature(tuple(tile_sigs))

    return signatures


def parse_colonist_initial_state(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse Colonist's initial state to extract tile layout.

    Returns dict with:
    - tiles: {tile_id: {resource, number}}
    - node_to_tiles: {node_id: [tile_ids]}
    """
    data = raw_data.get("data", raw_data)
    initial_state = data.get("initialState", {})

    result = {
        "tiles": {},
        "node_to_tiles": defaultdict(list),
    }

    # Parse tile states
    map_state = initial_state.get("mapState", {})
    tile_states = map_state.get("tileStates", {})

    for tile_id, tile_data in tile_states.items():
        tile_id = int(tile_id)
        resource_enum = tile_data.get("type", 0)
        number = tile_data.get("number")

        result["tiles"][tile_id] = {
            "resource": COLONIST_TO_ENGINE_RESOURCE.get(resource_enum),
            "number": number,
        }

    return result


def get_colonist_node_signatures_from_events(
    raw_data: Dict[str, Any]
) -> Dict[int, Set[Tuple[Optional[FastResource], Optional[int]]]]:
    """
    Infer Colonist node -> tile relationships from settlement placements
    and resource distributions.

    When a settlement is placed, we can see which resources it produces
    from subsequent dice rolls.
    """
    data = raw_data.get("data", raw_data)
    events = data.get("eventHistory", {}).get("events", [])

    # Track which resources each node produces on each dice roll
    node_resources: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))

    # Parse events to find settlement placements and resource gains
    settlements = {}  # node_id -> owner

    for event in events:
        state_change = event.get("stateChange", {})
        map_state = state_change.get("mapState", {})

        # Track settlement placements
        for corner_id, corner_data in map_state.get("tileCornerStates", {}).items():
            if corner_data and "owner" in corner_data:
                settlements[int(corner_id)] = corner_data["owner"]

        # Track resource gains on dice rolls
        dice_state = state_change.get("diceState", {})
        if dice_state.get("diceThrown"):
            dice_sum = dice_state.get("dice1", 0) + dice_state.get("dice2", 0)

            # See which players gained which resources
            player_states = state_change.get("playerStates", {})
            for player_id, p_state in player_states.items():
                resource_cards = p_state.get("resourceCards", {})
                cards = resource_cards.get("cards", [])
                if cards:
                    # This player gained resources on this roll
                    # Find their settlements
                    for node_id, owner in settlements.items():
                        if owner == int(player_id):
                            for card in cards:
                                resource = COLONIST_TO_ENGINE_RESOURCE.get(card)
                                if resource:
                                    node_resources[node_id][dice_sum].add(card)

    return node_resources


def build_mapping_from_replay(raw_data: Dict[str, Any]) -> CoordinateMapping:
    """
    Build coordinate mapping by analyzing a Colonist replay.

    This uses the tile layout from the initial state and matches
    nodes based on their production signatures.
    """
    mapping = CoordinateMapping()

    # Parse Colonist data
    colonist_state = parse_colonist_initial_state(raw_data)

    # TODO: Get Colonist tile->node relationships from their data
    # For now, we'll need to build this from multiple replays or
    # use a heuristic based on standard board layout

    # Create our engine map with the SAME tile configuration
    # This is the key insight: if we can match tile layouts, we can match nodes

    # For standard 4P games, the tile positions are fixed, only resources/numbers shuffle
    # So we need to:
    # 1. Extract Colonist's tile layout (resource + number for each position)
    # 2. Initialize our map with the same layout
    # 3. Then node IDs should correspond (assuming same traversal order)

    # This is still approximate - we need more data to verify
    mapping.confidence = 0.5

    return mapping


def analyze_colonist_replay(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a Colonist replay to understand its coordinate system.

    Returns analysis dict with discovered information.
    """
    data = raw_data.get("data", raw_data)
    events = data.get("eventHistory", {}).get("events", [])
    initial_state = data.get("initialState", {})

    analysis = {
        "node_ids_seen": set(),
        "edge_ids_seen": set(),
        "tile_ids_seen": set(),
        "settlement_placements": [],
        "road_placements": [],
        "initial_state_keys": list(initial_state.keys()) if initial_state else [],
        "map_state_keys": [],
    }

    # Parse initial state
    if initial_state:
        map_state = initial_state.get("mapState", {})
        analysis["map_state_keys"] = list(map_state.keys())

        tile_states = map_state.get("tileStates", {})
        for tile_id in tile_states:
            analysis["tile_ids_seen"].add(int(tile_id))

    # Parse events
    for event in events:
        state_change = event.get("stateChange", {})
        map_state = state_change.get("mapState", {})

        for corner_id, corner_data in map_state.get("tileCornerStates", {}).items():
            corner_int = int(corner_id)
            analysis["node_ids_seen"].add(corner_int)
            if corner_data and "owner" in corner_data:
                analysis["settlement_placements"].append({
                    "node_id": corner_int,
                    "owner": corner_data["owner"],
                    "building_type": corner_data.get("buildingType"),
                })

        for edge_id, edge_data in map_state.get("tileEdgeStates", {}).items():
            edge_int = int(edge_id)
            analysis["edge_ids_seen"].add(edge_int)
            if edge_data and "owner" in edge_data:
                analysis["road_placements"].append({
                    "edge_id": edge_int,
                    "owner": edge_data["owner"],
                })

    # Convert sets to sorted lists
    analysis["node_ids_seen"] = sorted(analysis["node_ids_seen"])
    analysis["edge_ids_seen"] = sorted(analysis["edge_ids_seen"])
    analysis["tile_ids_seen"] = sorted(analysis["tile_ids_seen"])

    return analysis


def dump_engine_coordinates():
    """
    Dump our engine's coordinate system for analysis.

    Creates a map and prints node/edge assignments.
    """
    catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)

    print("=== ENGINE COORDINATE DUMP ===\n")

    print(f"Total land nodes: {len(catan_map.land_nodes)}")
    print(f"Total tiles: {len(catan_map.land_tiles)}")

    print("\n--- TILES ---")
    for coord, tile in sorted(catan_map.land_tiles.items(), key=lambda x: x[1].id):
        resource = str(tile.resource) if tile.resource else "DESERT"
        print(f"Tile {tile.id}: {resource}:{tile.number} at {coord}")
        print(f"  Nodes: {dict(tile.nodes)}")
        print(f"  Edges: {dict(tile.edges)}")

    print("\n--- NODE SIGNATURES ---")
    signatures = get_engine_node_signatures(catan_map)
    for node_id in sorted(signatures.keys()):
        sig = signatures[node_id]
        print(f"Node {node_id}: {sig}")

    print("\n--- PORT NODES ---")
    for resource, nodes in catan_map.port_nodes.items():
        r_name = resource.name if resource else "3:1"
        print(f"{r_name} port: nodes {sorted(nodes)}")


def extract_node_resources_from_replay(raw_data: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Extract resource signatures for nodes from 2nd settlement placements.

    During initial placement, the 2nd settlement gives starting resources
    (one from each adjacent non-desert tile). This reveals the node's
    adjacent resources.

    Returns: {colonist_node_id: [resource_names]}
    """
    data = raw_data.get("data", raw_data)
    events = data.get("eventHistory", {}).get("events", [])

    # Track settlements by player
    player_settlements: Dict[int, List[int]] = defaultdict(list)

    # Track resources gained after each settlement
    node_resources: Dict[int, List[str]] = {}

    RESOURCE_MAP = {1: "WOOD", 2: "BRICK", 3: "SHEEP", 4: "WHEAT", 5: "ORE"}

    for event in events:
        state_change = event.get("stateChange", {})
        map_state = state_change.get("mapState", {})

        # Track settlement placements
        corner_states = map_state.get("tileCornerStates", {})
        for corner_id, corner_data in corner_states.items():
            if corner_data and corner_data.get("buildingType") == 1:  # Settlement
                owner = corner_data["owner"]
                corner_int = int(corner_id)
                player_settlements[owner].append(corner_int)

                # Check if this is their 2nd settlement (gives starting resources)
                if len(player_settlements[owner]) == 2:
                    # Look for resources in this same event
                    player_states = state_change.get("playerStates", {})
                    p_state = player_states.get(str(owner), {})
                    resource_cards = p_state.get("resourceCards", {})
                    cards = resource_cards.get("cards", [])
                    if cards:
                        resources = [RESOURCE_MAP.get(c, f"?{c}") for c in cards]
                        node_resources[corner_int] = resources
                        print(f"  Node {corner_int} (player {owner}): {resources}")

    return node_resources


def match_nodes_by_resource_signature(
    colonist_resources: Dict[int, List[str]],
    engine_signatures: Dict[int, NodeSignature],
) -> Dict[int, int]:
    """
    Match Colonist nodes to engine nodes by resource signature.

    Returns: {colonist_node_id: engine_node_id}
    """
    matches = {}

    for colonist_node, resources in colonist_resources.items():
        # Sort resources for comparison (count occurrences)
        resource_count = defaultdict(int)
        for r in resources:
            if r and not r.startswith("?"):
                resource_count[r] += 1

        # Find matching engine node
        for engine_node, sig in engine_signatures.items():
            # Get resources from engine node (ignoring numbers)
            engine_count = defaultdict(int)
            for t in sig.tiles:
                if t.resource:
                    r_str = str(t.resource)
                    engine_count[r_str] += 1

            if dict(resource_count) == dict(engine_count):
                matches[colonist_node] = engine_node
                print(f"  Matched: Colonist {colonist_node} -> Engine {engine_node}")
                print(f"    Resources: {dict(resource_count)}")
                break
        else:
            print(f"  No match for Colonist {colonist_node}: {dict(resource_count)}")

    return matches


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Coordinate Mapper")
    parser.add_argument("command", choices=["dump-engine", "analyze", "build-mapping", "match-nodes"],
                       help="Command to run")
    parser.add_argument("--replay", type=str, help="Path to Colonist replay JSON")
    parser.add_argument("--output", type=str, help="Output path for mapping")

    args = parser.parse_args()

    if args.command == "dump-engine":
        dump_engine_coordinates()

    elif args.command == "match-nodes":
        if not args.replay:
            print("Error: --replay required")
            sys.exit(1)

        with open(args.replay) as f:
            raw_data = json.load(f)

        print("=== EXTRACTING NODE RESOURCES FROM REPLAY ===\n")
        colonist_resources = extract_node_resources_from_replay(raw_data)

        print(f"\n=== MATCHING TO ENGINE NODES ===\n")
        catan_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
        engine_sigs = get_engine_node_signatures(catan_map)

        matches = match_nodes_by_resource_signature(colonist_resources, engine_sigs)

        print(f"\n=== RESULTS ===")
        print(f"Matched {len(matches)} out of {len(colonist_resources)} nodes")
        print(f"Mapping: {matches}")

    elif args.command == "analyze":
        if not args.replay:
            print("Error: --replay required for analyze command")
            sys.exit(1)

        with open(args.replay) as f:
            raw_data = json.load(f)

        analysis = analyze_colonist_replay(raw_data)

        print("=== COLONIST REPLAY ANALYSIS ===\n")
        print(f"Node IDs seen: {len(analysis['node_ids_seen'])}")
        print(f"  Range: {min(analysis['node_ids_seen'])} - {max(analysis['node_ids_seen'])}")
        print(f"  IDs: {analysis['node_ids_seen'][:20]}...")

        print(f"\nEdge IDs seen: {len(analysis['edge_ids_seen'])}")
        print(f"  Range: {min(analysis['edge_ids_seen'])} - {max(analysis['edge_ids_seen'])}")
        print(f"  IDs: {analysis['edge_ids_seen'][:20]}...")

        print(f"\nTile IDs seen: {len(analysis['tile_ids_seen'])}")
        if analysis['tile_ids_seen']:
            print(f"  Range: {min(analysis['tile_ids_seen'])} - {max(analysis['tile_ids_seen'])}")

        print(f"\nInitial state keys: {analysis['initial_state_keys']}")
        print(f"Map state keys: {analysis['map_state_keys']}")

        print(f"\nFirst 5 settlement placements:")
        for p in analysis["settlement_placements"][:5]:
            print(f"  Node {p['node_id']} by player {p['owner']}")

        print(f"\nFirst 5 road placements:")
        for p in analysis["road_placements"][:5]:
            print(f"  Edge {p['edge_id']} by player {p['owner']}")

    elif args.command == "build-mapping":
        if not args.replay:
            print("Error: --replay required for build-mapping command")
            sys.exit(1)

        with open(args.replay) as f:
            raw_data = json.load(f)

        mapping = build_mapping_from_replay(raw_data)

        output_path = args.output or "coordinate_mapping.json"
        mapping.save(output_path)
        print(f"Mapping saved to {output_path} (confidence: {mapping.confidence})")


if __name__ == "__main__":
    main()
