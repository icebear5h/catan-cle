#!/usr/bin/env python3
"""
Colonist to Engine Mapping

Maps Colonist.io game state coordinates and enums to our Catanatron engine format.

Colonist uses:
- tileCornerStates: integer corner IDs (0-53 for standard board)
- tileEdgeStates: integer edge IDs (0-71 for standard board)
- Tile positions: integer tile IDs
- Resources: 1=wood, 2=brick, 3=sheep, 4=wheat, 5=ore

Our engine uses:
- NodeId: integers assigned during map construction
- Edges: tuples of (NodeId, NodeId)
- Tile coordinates: (x, y) or (q, r, s) cube coordinates
- Resources: Resource enum (WOOD, BRICK, SHEEP, WHEAT, ORE)

The mapping is determined by analyzing the initial game state and matching
tile/node/edge structures.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Colonist resource enum -> our resource string
COLONIST_RESOURCE_MAP = {
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
    0: "DESERT",  # or None
}

# Colonist tile type enum
COLONIST_TILE_TYPE = {
    0: "DESERT",
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
    6: "WATER",
    7: "PORT",
}


@dataclass
class ColonistBoardState:
    """Parsed Colonist board state."""
    # Tiles: id -> {resource, number}
    tiles: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Corners (nodes): id -> {owner, buildingType}
    corners: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Edges: id -> {owner, type}
    edges: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Players: color -> state
    players: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Game state
    current_player: int = 0
    turn_number: int = 0
    phase: str = "initial_placement"
    dice_roll: Optional[Tuple[int, int]] = None


@dataclass
class CoordinateMapping:
    """
    Mapping between Colonist IDs and our engine IDs.

    Built by analyzing both board structures.
    """
    # Colonist corner ID -> our node ID
    corner_to_node: Dict[int, int] = field(default_factory=dict)

    # Colonist edge ID -> our edge tuple (node1, node2)
    edge_to_edge: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Colonist tile ID -> our tile coordinate
    tile_to_coord: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Colonist player color -> our player Color
    player_color_map: Dict[int, str] = field(default_factory=dict)


def parse_initial_state(raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract initial state from Colonist replay data."""
    data = raw_data.get("data", raw_data)
    return data.get("initialState")


def parse_board_from_events(raw_data: Dict[str, Any]) -> ColonistBoardState:
    """
    Parse board state by replaying events from the beginning.

    Returns the initial board state (before any player actions).
    """
    data = raw_data.get("data", raw_data)
    events = data.get("eventHistory", {}).get("events", [])

    board = ColonistBoardState()

    # Look for initial state in the data
    initial_state = data.get("initialState", {})
    if initial_state:
        # Parse tiles from initial state
        map_state = initial_state.get("mapState", {})
        tile_states = map_state.get("tileStates", {})
        for tile_id, tile_data in tile_states.items():
            board.tiles[int(tile_id)] = {
                "resource": COLONIST_TILE_TYPE.get(tile_data.get("type"), "UNKNOWN"),
                "number": tile_data.get("number"),
            }

    # Track cumulative state through events
    for event in events:
        state_change = event.get("stateChange", {})

        # Update corners (settlements/cities)
        map_state = state_change.get("mapState", {})
        for corner_id, corner_data in map_state.get("tileCornerStates", {}).items():
            if corner_data:  # Not null
                board.corners[int(corner_id)] = corner_data

        # Update edges (roads)
        for edge_id, edge_data in map_state.get("tileEdgeStates", {}).items():
            if edge_data:
                board.edges[int(edge_id)] = edge_data

        # Update player states
        for player_id, player_data in state_change.get("playerStates", {}).items():
            if player_id not in board.players:
                board.players[int(player_id)] = {}
            board.players[int(player_id)].update(player_data)

        # Update game state
        current_state = state_change.get("currentState", {})
        if "currentTurnPlayerColor" in current_state:
            board.current_player = current_state["currentTurnPlayerColor"]
        if "completedTurns" in current_state:
            board.turn_number = current_state["completedTurns"]

        # Update dice
        dice_state = state_change.get("diceState", {})
        if dice_state.get("diceThrown"):
            board.dice_roll = (dice_state.get("dice1", 0), dice_state.get("dice2", 0))

    return board


def build_standard_mapping() -> CoordinateMapping:
    """
    Build coordinate mapping for standard 4-player Colonist board.

    Colonist uses a specific numbering scheme for corners and edges
    that we need to map to our engine's auto-incrementing IDs.

    The standard board has:
    - 19 land tiles (arranged in hex pattern)
    - 54 corners (nodes where settlements can be placed)
    - 72 edges (where roads can be placed)

    TODO: This needs to be derived from actual board analysis.
    For now, we'll use identity mapping and adjust as needed.
    """
    mapping = CoordinateMapping()

    # Standard 4P board has 54 nodes
    # For initial implementation, assume Colonist uses same numbering
    # This will need refinement based on actual board analysis
    for i in range(54):
        mapping.corner_to_node[i] = i

    # Standard board has 72 edges
    # Edges in our engine are (node1, node2) tuples
    # Colonist uses single integer IDs
    # TODO: Build actual edge mapping
    for i in range(72):
        # Placeholder - needs actual edge->node pair mapping
        mapping.edge_to_edge[i] = (i, i + 1)

    # Player colors: Colonist uses 1, 2, 3, 5 for 4P games
    # (color 4 is spectator or unused)
    mapping.player_color_map = {
        1: "RED",
        2: "BLUE",
        3: "ORANGE",
        5: "WHITE",
    }

    return mapping


def generate_observation_text(
    board: ColonistBoardState,
    player_color: int,
    mapping: CoordinateMapping,
) -> str:
    """
    Generate observation text in our engine's format.

    This mimics the output of CatanObservationFormatter but works
    directly from Colonist data without needing to initialize our engine.
    """
    lines = []

    # Header
    lines.append(f"=== GAME STATE (Turn {board.turn_number}) ===")
    lines.append("")

    # Phase
    if board.turn_number < 8:
        phase = "initial_placement"
    else:
        phase = "main_game"
    lines.append(f"Phase: {phase}")

    # Player info
    player_state = board.players.get(player_color, {})
    vp_state = player_state.get("victoryPointsState", {})
    total_vp = sum(vp_state.values()) if isinstance(vp_state, dict) else 0
    lines.append(f"Your VP: {total_vp}/10")

    if board.dice_roll:
        lines.append(f"Last dice roll: {board.dice_roll[0] + board.dice_roll[1]}")

    lines.append("")

    # Your buildings
    lines.append("YOUR BUILDINGS:")
    my_settlements = []
    my_cities = []
    for corner_id, corner_data in board.corners.items():
        if corner_data.get("owner") == player_color:
            if corner_data.get("buildingType") == 1:
                my_settlements.append(corner_id)
            elif corner_data.get("buildingType") == 2:
                my_cities.append(corner_id)

    if my_settlements:
        lines.append(f"  Settlements ({len(my_settlements)}): nodes {my_settlements}")
    if my_cities:
        lines.append(f"  Cities ({len(my_cities)}): nodes {my_cities}")

    my_roads = []
    for edge_id, edge_data in board.edges.items():
        if edge_data.get("owner") == player_color:
            my_roads.append(edge_id)
    if my_roads:
        lines.append(f"  Roads ({len(my_roads)}): {len(my_roads)} connections")

    if not my_settlements and not my_cities:
        lines.append("  No buildings yet")

    lines.append("")

    # Resources
    lines.append("YOUR RESOURCES:")
    resource_cards = player_state.get("resourceCards", {})
    cards = resource_cards.get("cards", [])
    if cards:
        resource_counts = {}
        for card in cards:
            r_name = COLONIST_RESOURCE_MAP.get(card, "?")
            resource_counts[r_name] = resource_counts.get(r_name, 0) + 1
        for resource, count in resource_counts.items():
            lines.append(f"  {resource}: {count}")
        lines.append(f"  Total: {len(cards)} cards")

        # Can afford
        affordable = []
        if (resource_counts.get("WOOD", 0) >= 1 and
            resource_counts.get("BRICK", 0) >= 1 and
            resource_counts.get("SHEEP", 0) >= 1 and
            resource_counts.get("WHEAT", 0) >= 1):
            affordable.append("settlement")
        if (resource_counts.get("WHEAT", 0) >= 2 and
            resource_counts.get("ORE", 0) >= 3):
            affordable.append("city")
        if (resource_counts.get("WOOD", 0) >= 1 and
            resource_counts.get("BRICK", 0) >= 1):
            affordable.append("road")
        if affordable:
            lines.append(f"  Can afford: {', '.join(affordable)}")
    else:
        lines.append("  No resources")

    lines.append("")

    # Opponents
    lines.append("OPPONENTS:")
    for opp_color, opp_state in board.players.items():
        if opp_color == player_color:
            continue
        opp_vp = sum(opp_state.get("victoryPointsState", {}).values()) if isinstance(opp_state.get("victoryPointsState"), dict) else 0
        opp_cards = len(opp_state.get("resourceCards", {}).get("cards", []))

        # Count opponent buildings
        opp_settlements = sum(1 for c in board.corners.values() if c.get("owner") == opp_color and c.get("buildingType") == 1)
        opp_cities = sum(1 for c in board.corners.values() if c.get("owner") == opp_color and c.get("buildingType") == 2)
        opp_roads = sum(1 for e in board.edges.values() if e.get("owner") == opp_color)

        color_name = mapping.player_color_map.get(opp_color, f"Player{opp_color}")
        threat = "WINNING!" if opp_vp >= 8 else "threatening" if opp_vp >= 6 else "building"
        lines.append(f"  {color_name}: {opp_vp} VP ({opp_settlements} settlements, {opp_cities} cities, {opp_roads} roads, {opp_cards} resources) - {threat}")

    return "\n".join(lines)


def analyze_colonist_board(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a Colonist replay to understand the board structure.

    Returns analysis useful for building coordinate mappings.
    """
    data = raw_data.get("data", raw_data)
    events = data.get("eventHistory", {}).get("events", [])

    analysis = {
        "corner_ids_used": set(),
        "edge_ids_used": set(),
        "player_colors": set(),
        "max_corner_id": 0,
        "max_edge_id": 0,
        "settlement_placements": [],
        "road_placements": [],
    }

    for event in events:
        state_change = event.get("stateChange", {})
        map_state = state_change.get("mapState", {})

        for corner_id, corner_data in map_state.get("tileCornerStates", {}).items():
            corner_int = int(corner_id)
            analysis["corner_ids_used"].add(corner_int)
            analysis["max_corner_id"] = max(analysis["max_corner_id"], corner_int)
            if corner_data and "owner" in corner_data:
                analysis["settlement_placements"].append({
                    "corner_id": corner_int,
                    "owner": corner_data["owner"],
                    "type": corner_data.get("buildingType"),
                })
                analysis["player_colors"].add(corner_data["owner"])

        for edge_id, edge_data in map_state.get("tileEdgeStates", {}).items():
            edge_int = int(edge_id)
            analysis["edge_ids_used"].add(edge_int)
            analysis["max_edge_id"] = max(analysis["max_edge_id"], edge_int)
            if edge_data and "owner" in edge_data:
                analysis["road_placements"].append({
                    "edge_id": edge_int,
                    "owner": edge_data["owner"],
                    "type": edge_data.get("type"),
                })
                analysis["player_colors"].add(edge_data["owner"])

    # Convert sets to lists for JSON serialization
    analysis["corner_ids_used"] = sorted(analysis["corner_ids_used"])
    analysis["edge_ids_used"] = sorted(analysis["edge_ids_used"])
    analysis["player_colors"] = sorted(analysis["player_colors"])

    return analysis


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python colonist_to_engine.py <replay_json_file>")
        print("\nAnalyzes a Colonist replay file to understand board structure")
        sys.exit(1)

    replay_file = sys.argv[1]
    with open(replay_file) as f:
        raw_data = json.load(f)

    # Analyze board structure
    print("Analyzing board structure...")
    analysis = analyze_colonist_board(raw_data)
    print(f"\nCorner IDs used: {len(analysis['corner_ids_used'])} (max: {analysis['max_corner_id']})")
    print(f"Edge IDs used: {len(analysis['edge_ids_used'])} (max: {analysis['max_edge_id']})")
    print(f"Player colors: {analysis['player_colors']}")
    print(f"\nFirst 5 settlement placements:")
    for p in analysis["settlement_placements"][:5]:
        print(f"  Corner {p['corner_id']} by player {p['owner']}")
    print(f"\nFirst 5 road placements:")
    for p in analysis["road_placements"][:5]:
        print(f"  Edge {p['edge_id']} by player {p['owner']}")

    # Parse full board state
    print("\n\nParsing full board state...")
    board = parse_board_from_events(raw_data)
    mapping = build_standard_mapping()

    # Generate sample observation
    if board.players:
        sample_player = list(board.players.keys())[0]
        print(f"\n\nSample observation for player {sample_player}:")
        print("=" * 60)
        obs_text = generate_observation_text(board, sample_player, mapping)
        print(obs_text)
