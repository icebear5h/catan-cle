#!/usr/bin/env python3
"""
Replay to Training Data Pipeline

Converts raw Colonist.io replay events into observation-action training pairs
by replaying through our Catan engine.

Pipeline:
1. Load raw replay from data lake
2. Initialize engine with board state
3. Step through events, at each decision point:
   - Generate text observation (using our formatter)
   - Map Colonist action to our action space
   - Record (observation, action, outcome) tuple
4. Export as JSONL for training

Data Lake Structure:
    data/
    ├── raw_replays/          # Raw JSON from Colonist API
    │   ├── 192418134.json
    │   └── ...
    ├── processed/            # Intermediate processed data
    │   └── ...
    └── training/             # Final training data
        ├── observations.jsonl
        └── policy_data.jsonl
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from enum import IntEnum

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Colonist resource enum -> our resource names
COLONIST_RESOURCE = {
    1: "WOOD",
    2: "BRICK",
    3: "SHEEP",
    4: "WHEAT",
    5: "ORE",
}

# Colonist building types
class ColonistBuildingType(IntEnum):
    SETTLEMENT = 1
    CITY = 2

# Colonist edge types
class ColonistEdgeType(IntEnum):
    ROAD = 1
    SHIP = 2

# Colonist action states (what action is expected)
class ColonistActionState(IntEnum):
    ROLL_DICE = 0
    PLACE_INITIAL_SETTLEMENT = 1
    PLACE_INITIAL_ROAD = 3
    PLAY_TURN = 4  # Can build, trade, etc.
    DISCARD = 5
    MOVE_ROBBER = 6
    STEAL = 7


@dataclass
class TrainingExample:
    """A single training example."""
    game_id: str
    event_index: int
    turn_number: int
    phase: str
    acting_player: int

    # The observation (what the agent sees)
    observation: str

    # The action taken (in our format)
    action_type: str
    action_params: Dict[str, Any]

    # Outcome
    is_winner: bool

    # Metadata
    time_taken_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ColonistToEngineMapper:
    """
    Maps Colonist.io game state and actions to our engine format.

    Colonist uses different coordinate systems and enums than our engine.
    This class handles the translation.
    """

    def __init__(self):
        # Will be populated when we analyze the board layout
        self.corner_id_to_node_id: Dict[int, int] = {}
        self.edge_id_to_edge_id: Dict[int, Tuple[int, int]] = {}
        self.tile_id_to_tile_id: Dict[int, int] = {}

    def map_resource(self, colonist_resource: int) -> str:
        """Map Colonist resource enum to our resource name."""
        return COLONIST_RESOURCE.get(colonist_resource, "UNKNOWN")

    def map_corner_to_node(self, corner_id: int) -> int:
        """Map Colonist corner ID to our node ID."""
        # TODO: Build mapping from initial state analysis
        # For now, assume 1:1 (will need adjustment)
        return corner_id

    def map_edge_to_edge(self, edge_id: int) -> Tuple[int, int]:
        """Map Colonist edge ID to our edge tuple."""
        # TODO: Build mapping from initial state analysis
        return (edge_id, edge_id + 1)  # Placeholder

    def map_action(self, event: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Map a Colonist event to our action format.

        Returns (action_type, action_params)
        """
        state_change = event.get("stateChange", {})

        # Check for settlement/city placement
        map_state = state_change.get("mapState", {})
        corner_states = map_state.get("tileCornerStates", {})
        if corner_states:
            for corner_id, data in corner_states.items():
                if "owner" in data and "buildingType" in data:
                    building_type = data["buildingType"]
                    if building_type == ColonistBuildingType.SETTLEMENT:
                        return "BUILD_SETTLEMENT", {
                            "node_id": self.map_corner_to_node(int(corner_id)),
                            "player": data["owner"],
                        }
                    elif building_type == ColonistBuildingType.CITY:
                        return "BUILD_CITY", {
                            "node_id": self.map_corner_to_node(int(corner_id)),
                            "player": data["owner"],
                        }

        # Check for road placement
        edge_states = map_state.get("tileEdgeStates", {})
        if edge_states:
            for edge_id, data in edge_states.items():
                if "owner" in data and "type" in data:
                    return "BUILD_ROAD", {
                        "edge": self.map_edge_to_edge(int(edge_id)),
                        "player": data["owner"],
                    }

        # Check for dice roll
        dice_state = state_change.get("diceState", {})
        if dice_state.get("diceThrown") and "dice1" in dice_state:
            return "ROLL", {
                "dice1": dice_state["dice1"],
                "dice2": dice_state["dice2"],
            }

        # Check for trade offer
        trade_state = state_change.get("tradeState", {})
        active_offers = trade_state.get("activeOffers", {})
        for offer_id, offer in active_offers.items():
            if offer and "creator" in offer:
                return "OFFER_TRADE", {
                    "player": offer["creator"],
                    "offered": [self.map_resource(r) for r in offer.get("offeredResources", [])],
                    "wanted": [self.map_resource(r) for r in offer.get("wantedResources", [])],
                }

        # Check for end turn
        current_state = state_change.get("currentState", {})
        if "completedTurns" in current_state:
            prev_player = current_state.get("currentTurnPlayerColor")
            return "END_TURN", {"player": prev_player}

        # Check for robber movement
        # TODO: Parse robber placement

        # Check for dev card purchase/play
        # TODO: Parse dev card actions

        return "UNKNOWN", {"raw_event": event}


class ReplayProcessor:
    """
    Processes raw Colonist replays through our engine to generate training data.
    """

    def __init__(self, data_lake_path: str = "./data"):
        self.data_lake = Path(data_lake_path)
        self.raw_replays = self.data_lake / "raw_replays"
        self.processed = self.data_lake / "processed"
        self.training = self.data_lake / "training"

        # Create directories
        for dir_path in [self.raw_replays, self.processed, self.training]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.mapper = ColonistToEngineMapper()

    def save_raw_replay(self, game_id: str, raw_data: Dict[str, Any]):
        """Save raw replay to data lake."""
        output_file = self.raw_replays / f"{game_id}.json"
        with open(output_file, "w") as f:
            json.dump(raw_data, f)
        logger.info(f"Saved raw replay: {output_file}")

    def load_raw_replay(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Load raw replay from data lake."""
        input_file = self.raw_replays / f"{game_id}.json"
        if not input_file.exists():
            return None
        with open(input_file) as f:
            return json.load(f)

    def process_replay(self, game_id: str, winner_player: Optional[int] = None) -> List[TrainingExample]:
        """
        Process a replay into training examples.

        Args:
            game_id: The game to process
            winner_player: Which player won (for outcome labeling)

        Returns:
            List of TrainingExample objects
        """
        raw_data = self.load_raw_replay(game_id)
        if not raw_data:
            logger.error(f"No raw replay found for {game_id}")
            return []

        data = raw_data.get("data", raw_data)
        events = data.get("eventHistory", {}).get("events", [])

        examples = []
        turn_number = 0
        phase = "initial_placement"

        # Track cumulative state
        buildings = {}  # corner_id -> {owner, type}
        roads = {}  # edge_id -> {owner, type}
        resources = {}  # player -> {resource -> count}

        for i, event in enumerate(events):
            state_change = event.get("stateChange", {})
            delta_s = event.get("input", {}).get("deltaS", 0)

            # Map the action
            action_type, action_params = self.mapper.map_action(event)

            # Skip non-actions (pure state updates)
            if action_type == "UNKNOWN":
                continue

            # Determine acting player
            acting_player = action_params.get("player")
            if acting_player is None:
                current_state = state_change.get("currentState", {})
                acting_player = current_state.get("currentTurnPlayerColor", 0)

            # Update phase and turn tracking
            current_state = state_change.get("currentState", {})
            if "completedTurns" in current_state:
                turn_number = current_state["completedTurns"]
                if turn_number >= 8:  # After initial placements
                    phase = "main_game"

            # Generate observation (placeholder - will use actual engine)
            observation = self._generate_observation(
                turn_number=turn_number,
                phase=phase,
                acting_player=acting_player,
                buildings=buildings,
                roads=roads,
                resources=resources,
            )

            # Create training example
            example = TrainingExample(
                game_id=game_id,
                event_index=i,
                turn_number=turn_number,
                phase=phase,
                acting_player=acting_player,
                observation=observation,
                action_type=action_type,
                action_params=action_params,
                is_winner=(acting_player == winner_player) if winner_player is not None else False,
                time_taken_seconds=delta_s,
            )
            examples.append(example)

            # Update cumulative state
            map_state = state_change.get("mapState", {})
            for corner_id, data in map_state.get("tileCornerStates", {}).items():
                if "owner" in data:
                    buildings[corner_id] = data
            for edge_id, data in map_state.get("tileEdgeStates", {}).items():
                if "owner" in data:
                    roads[edge_id] = data

            for player_id, p_state in state_change.get("playerStates", {}).items():
                if "resourceCards" in p_state:
                    resources[player_id] = p_state["resourceCards"]

        return examples

    def _generate_observation(
        self,
        turn_number: int,
        phase: str,
        acting_player: int,
        buildings: Dict,
        roads: Dict,
        resources: Dict,
    ) -> str:
        """
        Generate text observation.

        TODO: Use actual engine observation formatter for consistency.
        """
        # Placeholder - will integrate with cle/env/gym_env observation formatter
        obs_parts = [
            f"Turn: {turn_number}",
            f"Phase: {phase}",
            f"Your color: Player {acting_player}",
            f"Buildings placed: {len(buildings)}",
            f"Roads placed: {len(roads)}",
        ]

        # Add resource info if available
        player_resources = resources.get(str(acting_player), {})
        if player_resources:
            cards = player_resources.get("cards", [])
            if cards:
                resource_counts = {}
                for card in cards:
                    r_name = COLONIST_RESOURCE.get(card, "?")
                    resource_counts[r_name] = resource_counts.get(r_name, 0) + 1
                obs_parts.append(f"Your resources: {resource_counts}")

        return "\n".join(obs_parts)

    def export_training_data(self, examples: List[TrainingExample], output_name: str = "policy_data"):
        """Export training examples to JSONL."""
        output_file = self.training / f"{output_name}.jsonl"

        with open(output_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        logger.info(f"Exported {len(examples)} examples to {output_file}")

    def process_all_replays(self, max_games: Optional[int] = None) -> List[TrainingExample]:
        """Process all replays in the data lake."""
        all_examples = []

        replay_files = list(self.raw_replays.glob("*.json"))
        if max_games:
            replay_files = replay_files[:max_games]

        for i, replay_file in enumerate(replay_files):
            game_id = replay_file.stem
            logger.info(f"[{i+1}/{len(replay_files)}] Processing {game_id}...")

            try:
                examples = self.process_replay(game_id)
                all_examples.extend(examples)
                logger.info(f"  Generated {len(examples)} examples")
            except Exception as e:
                logger.error(f"  Failed: {e}")

        return all_examples


def analyze_replay_structure(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a replay to understand its structure.
    Useful for debugging and mapping development.
    """
    data = raw_data.get("data", raw_data)
    events = data.get("eventHistory", {}).get("events", [])

    analysis = {
        "total_events": len(events),
        "action_types": {},
        "state_change_keys": set(),
        "map_state_keys": set(),
        "player_count": 0,
        "turn_count": 0,
    }

    for event in events:
        state_change = event.get("stateChange", {})
        analysis["state_change_keys"].update(state_change.keys())

        map_state = state_change.get("mapState", {})
        analysis["map_state_keys"].update(map_state.keys())

        # Track action types
        if "tileCornerStates" in map_state:
            for data in map_state["tileCornerStates"].values():
                if data.get("buildingType") == 1:
                    analysis["action_types"]["settlement"] = analysis["action_types"].get("settlement", 0) + 1
                elif data.get("buildingType") == 2:
                    analysis["action_types"]["city"] = analysis["action_types"].get("city", 0) + 1

        if "tileEdgeStates" in map_state:
            analysis["action_types"]["road"] = analysis["action_types"].get("road", 0) + len(map_state["tileEdgeStates"])

        dice_state = state_change.get("diceState", {})
        if dice_state.get("diceThrown"):
            analysis["action_types"]["roll"] = analysis["action_types"].get("roll", 0) + 1

        trade_state = state_change.get("tradeState", {})
        if trade_state.get("activeOffers"):
            analysis["action_types"]["trade_offer"] = analysis["action_types"].get("trade_offer", 0) + 1

        current_state = state_change.get("currentState", {})
        if "completedTurns" in current_state:
            analysis["turn_count"] = max(analysis["turn_count"], current_state["completedTurns"])

    # Convert sets to lists for JSON serialization
    analysis["state_change_keys"] = list(analysis["state_change_keys"])
    analysis["map_state_keys"] = list(analysis["map_state_keys"])

    return analysis


if __name__ == "__main__":
    import sys

    # Quick test
    processor = ReplayProcessor()

    if len(sys.argv) > 1:
        game_id = sys.argv[1]
        examples = processor.process_replay(game_id)
        print(f"\nProcessed {len(examples)} training examples from game {game_id}")

        if examples:
            print("\nSample examples:")
            for ex in examples[:5]:
                print(f"  [{ex.event_index}] {ex.action_type} by player {ex.acting_player}")
                print(f"      Params: {ex.action_params}")
    else:
        print("Usage: python replay_to_training.py <game_id>")
        print("\nFirst, save raw replays to data/raw_replays/")
        print("Then run this to process them into training data")
