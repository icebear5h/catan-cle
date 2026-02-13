"""
Schema for Colonist.io replay data.

Represents the game state and actions extracted from replays
in a format suitable for training our Catan agents.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json


class GameAction(str, Enum):
    """Actions that can be taken in the game."""
    # Dice
    ROLL_DICE = "ROLL_DICE"

    # Building
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    BUILD_CITY = "BUILD_CITY"
    BUILD_ROAD = "BUILD_ROAD"
    BUILD_SHIP = "BUILD_SHIP"

    # Development cards
    BUY_DEV_CARD = "BUY_DEV_CARD"
    PLAY_KNIGHT = "PLAY_KNIGHT"
    PLAY_ROAD_BUILDING = "PLAY_ROAD_BUILDING"
    PLAY_YEAR_OF_PLENTY = "PLAY_YEAR_OF_PLENTY"
    PLAY_MONOPOLY = "PLAY_MONOPOLY"

    # Trading
    TRADE_WITH_BANK = "TRADE_WITH_BANK"
    TRADE_WITH_PLAYER = "TRADE_WITH_PLAYER"
    PROPOSE_TRADE = "PROPOSE_TRADE"
    ACCEPT_TRADE = "ACCEPT_TRADE"
    REJECT_TRADE = "REJECT_TRADE"

    # Robber
    MOVE_ROBBER = "MOVE_ROBBER"
    STEAL_CARD = "STEAL_CARD"
    DISCARD_CARDS = "DISCARD_CARDS"

    # Turn management
    END_TURN = "END_TURN"
    PASS = "PASS"

    # Initial placement
    PLACE_INITIAL_SETTLEMENT = "PLACE_INITIAL_SETTLEMENT"
    PLACE_INITIAL_ROAD = "PLACE_INITIAL_ROAD"

    # Cities & Knights specific
    BUILD_CITY_WALL = "BUILD_CITY_WALL"
    PLACE_KNIGHT = "PLACE_KNIGHT"
    UPGRADE_KNIGHT = "UPGRADE_KNIGHT"
    ACTIVATE_KNIGHT = "ACTIVATE_KNIGHT"
    MOVE_KNIGHT = "MOVE_KNIGHT"
    UPGRADE_CITY = "UPGRADE_CITY"

    # Unknown/other
    UNKNOWN = "UNKNOWN"


class PlayerColor(str, Enum):
    """Player colors in Colonist."""
    RED = "RED"
    BLUE = "BLUE"
    ORANGE = "ORANGE"
    WHITE = "WHITE"
    GREEN = "GREEN"
    BROWN = "BROWN"


class Resource(str, Enum):
    """Resource types."""
    WOOD = "WOOD"
    BRICK = "BRICK"
    SHEEP = "SHEEP"
    WHEAT = "WHEAT"
    ORE = "ORE"


@dataclass
class TileState:
    """State of a single hex tile."""
    index: int
    resource: Optional[str]  # Resource type or None for desert
    number: Optional[int]  # Dice number (2-12) or None for desert
    has_robber: bool = False


@dataclass
class NodeState:
    """State of a node (intersection)."""
    index: int
    building: Optional[str] = None  # "SETTLEMENT", "CITY", or None
    owner: Optional[int] = None  # Player index (0-3)


@dataclass
class EdgeState:
    """State of an edge (road position)."""
    index: int
    has_road: bool = False
    owner: Optional[int] = None  # Player index


@dataclass
class PlayerState:
    """State of a single player."""
    index: int
    color: str
    username: str
    victory_points: int
    resources: Dict[str, int]  # Resource counts
    dev_cards: List[str]
    settlements: List[int]  # Node indices
    cities: List[int]  # Node indices
    roads: List[int]  # Edge indices
    knights_played: int = 0
    longest_road: bool = False
    largest_army: bool = False
    has_rolled: bool = False

    # Cities & Knights specific
    city_improvements: Optional[Dict[str, int]] = None
    knights: Optional[List[Dict[str, Any]]] = None


@dataclass
class GameStateSnapshot:
    """Complete game state at a point in time."""

    # Game metadata
    game_id: str
    step_number: int
    timestamp: Optional[datetime] = None

    # Board state
    tiles: List[TileState] = field(default_factory=list)
    nodes: List[NodeState] = field(default_factory=list)
    edges: List[EdgeState] = field(default_factory=list)

    # Player states
    players: List[PlayerState] = field(default_factory=list)

    # Current game context
    current_player: int = 0
    phase: str = "main"  # "initial_placement", "main", "robber", "discard", etc.
    dice_roll: Optional[int] = None
    turn_number: int = 0

    # Available actions
    playable_actions: List[str] = field(default_factory=list)

    # Bank state
    bank_resources: Dict[str, int] = field(default_factory=dict)
    dev_cards_remaining: int = 25

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.timestamp:
            data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass
class ActionRecord:
    """Record of an action taken."""
    action_type: str
    player_index: int
    details: Dict[str, Any] = field(default_factory=dict)

    # For building actions
    node_index: Optional[int] = None
    edge_index: Optional[int] = None
    tile_index: Optional[int] = None

    # For trading actions
    give_resources: Optional[Dict[str, int]] = None
    receive_resources: Optional[Dict[str, int]] = None
    trade_partner: Optional[int] = None

    # For development cards
    card_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayStep:
    """A single step in a replay: state before + action taken."""
    step_number: int
    state_before: GameStateSnapshot
    action: ActionRecord
    state_after: Optional[GameStateSnapshot] = None

    def to_training_example(self) -> Dict[str, Any]:
        """Convert to format suitable for LLM training."""
        return {
            "observation": self.state_before.to_dict(),
            "action": self.action.to_dict(),
            "player": self.action.player_index,
            "available_actions": self.state_before.playable_actions,
        }


@dataclass
class ReplayData:
    """Complete replay data for a game."""
    game_id: str
    mode: str  # "Classic4P", "CitiesAndKnights4P", etc.
    map_type: str
    scraped_at: datetime

    # Player info
    players: List[Dict[str, Any]]  # Username, rating, color, final_score
    winner_index: int

    # Steps
    steps: List[ReplayStep] = field(default_factory=list)

    # Metadata
    total_turns: int = 0
    duration_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "game_id": self.game_id,
            "mode": self.mode,
            "map_type": self.map_type,
            "scraped_at": self.scraped_at.isoformat(),
            "players": self.players,
            "winner_index": self.winner_index,
            "total_turns": self.total_turns,
            "duration_seconds": self.duration_seconds,
            "steps": [
                {
                    "step_number": step.step_number,
                    "state_before": step.state_before.to_dict(),
                    "action": step.action.to_dict(),
                }
                for step in self.steps
            ],
        }

    def save(self, filepath: str):
        """Save replay to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ReplayData":
        """Load replay from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Reconstruct objects
        steps = []
        for step_data in data.get("steps", []):
            step = ReplayStep(
                step_number=step_data["step_number"],
                state_before=GameStateSnapshot(**step_data["state_before"]),
                action=ActionRecord(**step_data["action"]),
            )
            steps.append(step)

        return cls(
            game_id=data["game_id"],
            mode=data["mode"],
            map_type=data.get("map_type", "Classic4P"),
            scraped_at=datetime.fromisoformat(data["scraped_at"]),
            players=data["players"],
            winner_index=data["winner_index"],
            steps=steps,
            total_turns=data.get("total_turns", 0),
            duration_seconds=data.get("duration_seconds"),
        )

    def get_training_examples(self, winner_only: bool = True) -> List[Dict[str, Any]]:
        """
        Extract training examples from the replay.

        Args:
            winner_only: Only include decisions by the winning player
                        (stronger training signal)

        Returns:
            List of training examples
        """
        examples = []

        for step in self.steps:
            if winner_only and step.action.player_index != self.winner_index:
                continue

            examples.append(step.to_training_example())

        return examples


class ReplayDataStore:
    """Storage for scraped replays."""

    def __init__(self, base_dir: str = "./data/replays"):
        self.base_dir = base_dir
        import os
        os.makedirs(base_dir, exist_ok=True)

    def save_replay(self, replay: ReplayData):
        """Save a replay to the store."""
        filepath = f"{self.base_dir}/{replay.game_id}.json"
        replay.save(filepath)

    def load_replay(self, game_id: str) -> Optional[ReplayData]:
        """Load a replay from the store."""
        filepath = f"{self.base_dir}/{game_id}.json"
        try:
            return ReplayData.load(filepath)
        except FileNotFoundError:
            return None

    def list_replays(self) -> List[str]:
        """List all saved replay game IDs."""
        import os
        return [
            f.replace(".json", "")
            for f in os.listdir(self.base_dir)
            if f.endswith(".json")
        ]

    def get_all_training_examples(self, winner_only: bool = True) -> List[Dict[str, Any]]:
        """Load all replays and extract training examples."""
        examples = []
        for game_id in self.list_replays():
            replay = self.load_replay(game_id)
            if replay:
                examples.extend(replay.get_training_examples(winner_only=winner_only))
        return examples
