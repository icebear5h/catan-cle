"""
Data schema for scraped Catan knowledge.

Stores expert reasoning, game states, and outcomes for training.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class GameState:
    """Snapshot of the game board at a decision point."""

    # Board configuration
    tiles: List[Dict[str, Any]]  # [{resource: str, number: int, position: tuple}, ...]
    ports: List[Dict[str, Any]]  # [{type: str, position: tuple}, ...]

    # Player state (from perspective of decision-maker)
    my_settlements: List[tuple]
    my_cities: List[tuple]
    my_roads: List[tuple]
    my_resources: Dict[str, int]  # Visible in own commentary
    my_dev_cards: List[str]
    my_vp: int

    # Opponent states (visible info only)
    opponent_states: List[Dict[str, Any]]

    # Game context
    turn_number: int
    phase: str  # "initial_placement", "main_game", "endgame"
    dice_roll: Optional[int]

    def to_observation_format(self) -> str:
        """Convert to text format matching your observation formatter."""
        # TODO: Format to match CatanObservationFormatter output
        return json.dumps(asdict(self), indent=2)


@dataclass
class ExpertDecision:
    """A single decision made by an expert with their reasoning."""

    # Source metadata
    source_type: str  # "youtube", "blog", "twitch", "reddit"
    source_url: str
    source_id: str  # Video ID, article slug, etc.
    timestamp: Optional[str]  # For videos: "12:34", for blogs: None
    expert_name: str
    expert_rating: Optional[int]  # ELO if known

    # Game state
    game_state: GameState

    # Expert's reasoning
    raw_reasoning: str  # Exact quote from video/blog
    action_taken: str  # Description of action

    # Structured reasoning (extracted by Claude)
    strategic_principle: Optional[str]  # What principle are they applying?
    tactical_consideration: Optional[str]  # Immediate tactical reason
    alternatives_considered: Optional[List[str]]  # Other options discussed

    # Outcome
    immediate_result: Optional[str]  # What happened next turn
    game_outcome: Optional[str]  # "win", "loss", "unknown"

    # Quality signals
    upvotes: Optional[int]  # For Reddit/forum posts
    view_count: Optional[int]  # For videos

    # Training metadata
    scraped_at: datetime
    processed: bool = False

    def to_training_example(self) -> Dict[str, Any]:
        """Convert to format for fine-tuning."""
        return {
            "observation": self.game_state.to_observation_format(),
            "reasoning": self.raw_reasoning,
            "strategic_principle": self.strategic_principle,
            "action": self.action_taken,
            "outcome": self.game_outcome,
            "quality_score": self._compute_quality_score()
        }

    def _compute_quality_score(self) -> float:
        """Compute quality score for weighting during training."""
        score = 1.0

        # Boost for expert rating
        if self.expert_rating and self.expert_rating > 1800:
            score *= 1.5

        # Boost for engagement
        if self.upvotes and self.upvotes > 100:
            score *= 1.3
        if self.view_count and self.view_count > 10000:
            score *= 1.2

        # Boost for winning games
        if self.game_outcome == "win":
            score *= 1.2

        # Boost for detailed reasoning
        if len(self.raw_reasoning) > 200:
            score *= 1.1

        return score


@dataclass
class StrategyCorpusEntry:
    """General strategic knowledge (not tied to specific game state)."""

    source_type: str
    source_url: str

    # Strategic content
    principle: str  # E.g., "6 and 8 are the best numbers"
    explanation: str  # Why this principle matters
    category: str  # "initial_placement", "trading", "development", etc.

    # Quality signals
    upvotes: Optional[int]
    expert_endorsement: bool  # Did a known expert write/say this?

    scraped_at: datetime


class DataStore:
    """Simple JSON-based storage for scraped data."""

    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        self.decisions_file = f"{base_dir}/expert_decisions.jsonl"
        self.corpus_file = f"{base_dir}/strategy_corpus.jsonl"

    def save_decision(self, decision: ExpertDecision):
        """Append decision to JSONL file."""
        import os
        os.makedirs(self.base_dir, exist_ok=True)

        with open(self.decisions_file, 'a') as f:
            # Convert dataclass to dict, handling nested objects
            data = asdict(decision)
            data['scraped_at'] = decision.scraped_at.isoformat()
            f.write(json.dumps(data) + '\n')

    def save_corpus_entry(self, entry: StrategyCorpusEntry):
        """Append corpus entry to JSONL file."""
        import os
        os.makedirs(self.base_dir, exist_ok=True)

        with open(self.corpus_file, 'a') as f:
            data = asdict(entry)
            data['scraped_at'] = entry.scraped_at.isoformat()
            f.write(json.dumps(data) + '\n')

    def load_decisions(self) -> List[ExpertDecision]:
        """Load all decisions from storage."""
        decisions = []
        try:
            with open(self.decisions_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # TODO: Reconstruct ExpertDecision from dict
                    decisions.append(data)
        except FileNotFoundError:
            pass
        return decisions

    def get_training_dataset(self, min_quality_score: float = 0.8) -> List[Dict]:
        """Get filtered dataset for training."""
        decisions = self.load_decisions()
        # TODO: Filter by quality, convert to training format
        return decisions


if __name__ == "__main__":
    # Example usage
    store = DataStore()

    # Example decision
    decision = ExpertDecision(
        source_type="youtube",
        source_url="https://youtube.com/watch?v=example",
        source_id="example",
        timestamp="12:34",
        expert_name="The Catan Guy",
        expert_rating=1950,
        game_state=GameState(
            tiles=[],
            ports=[],
            my_settlements=[(0, 0)],
            my_cities=[],
            my_roads=[],
            my_resources={"wheat": 2, "ore": 1},
            my_dev_cards=[],
            my_vp=2,
            opponent_states=[],
            turn_number=5,
            phase="main_game",
            dice_roll=8
        ),
        raw_reasoning="I'm building a city here because I need the 2:1 production on ore, and I'm already getting wheat from my port.",
        action_taken="BUILD_CITY at (0,0)",
        strategic_principle="Maximize resource production efficiency",
        tactical_consideration="Cities give 2x production, needed for ore bottleneck",
        alternatives_considered=["Build settlement", "Buy dev card"],
        immediate_result="Got 2 ore next turn",
        game_outcome="win",
        upvotes=None,
        view_count=15000,
        scraped_at=datetime.now()
    )

    print("Example training entry:")
    print(json.dumps(decision.to_training_example(), indent=2))
