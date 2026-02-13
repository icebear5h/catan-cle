#!/usr/bin/env python3
"""
Generate Training Data from Colonist Replays

This takes the practical approach: generate observations directly from Colonist
data format, without trying to map to our engine's coordinate system.

The key insight is that node/edge IDs are arbitrary - what matters is:
1. The RELATIONSHIPS (which tiles are adjacent, what resources they produce)
2. The DECISION CONTEXT (what you can see, what actions are available)
3. The ACTION TAKEN (what the expert player chose)

The agent learns PATTERNS like:
- "Place settlement on high-pip ore+wheat spot when you need cities"
- "Trade sheep for ore when you have 4+ sheep"

These patterns transfer regardless of coordinate system.

Output format (JSONL):
{
    "observation": "=== GAME STATE ===\n...",
    "action": "BUILD_SETTLEMENT node=51",
    "action_type": "BUILD_SETTLEMENT",
    "action_value": 51,
    "player": 2,
    "turn": 0,
    "phase": "initial_placement",
    "is_winner": true,
    "game_id": "192418134"
}
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Resource enum mapping
RESOURCE_MAP = {1: "WOOD", 2: "BRICK", 3: "SHEEP", 4: "WHEAT", 5: "ORE", 0: "DESERT"}
RESOURCE_MAP_REV = {v: k for k, v in RESOURCE_MAP.items()}


@dataclass
class TrainingExample:
    """A single training example for behavioral cloning."""
    game_id: str
    event_index: int
    turn: int
    phase: str
    player: int

    # The observation text (what the agent sees)
    observation: str

    # The action (what the expert did)
    action: str
    action_type: str
    action_value: Any

    # Outcome
    is_winner: bool

    # Timing
    time_taken_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ColonistObservationFormatter:
    """
    Formats Colonist game state into text observations.

    This generates observations in the same STYLE as our engine formatter,
    but using Colonist's data directly.
    """

    def __init__(self):
        pass

    def format(
        self,
        board_state: Dict[str, Any],
        player_color: int,
        turn: int,
        phase: str,
        last_dice: Optional[Tuple[int, int]] = None,
    ) -> str:
        """Generate observation text from Colonist board state."""
        lines = []

        # Header
        lines.append(f"=== GAME STATE (Turn {turn}) ===")
        lines.append("")

        # Phase and status
        lines.append(f"Phase: {phase}")
        lines.append(f"You are: Player {player_color}")

        # Victory points
        vp = self._get_player_vp(board_state, player_color)
        lines.append(f"Your VP: {vp}/10")

        if last_dice:
            lines.append(f"Last dice roll: {last_dice[0]} + {last_dice[1]} = {sum(last_dice)}")

        lines.append("")

        # Your buildings
        lines.append("YOUR BUILDINGS:")
        my_settlements = self._get_player_buildings(board_state, player_color, 1)
        my_cities = self._get_player_buildings(board_state, player_color, 2)
        my_roads = self._get_player_roads(board_state, player_color)

        if my_settlements:
            lines.append(f"  Settlements ({len(my_settlements)}): nodes {sorted(my_settlements)}")
        if my_cities:
            lines.append(f"  Cities ({len(my_cities)}): nodes {sorted(my_cities)}")
        if my_roads:
            lines.append(f"  Roads ({len(my_roads)}): edges {sorted(my_roads)}")
        if not my_settlements and not my_cities:
            lines.append("  No buildings yet")

        lines.append("")

        # Your resources
        lines.append("YOUR RESOURCES:")
        resources = self._get_player_resources(board_state, player_color)
        if resources:
            for resource, count in sorted(resources.items()):
                lines.append(f"  {resource}: {count}")
            lines.append(f"  Total: {sum(resources.values())} cards")

            # What can you afford
            affordable = self._get_affordable(resources)
            if affordable:
                lines.append(f"  Can afford: {', '.join(affordable)}")
        else:
            lines.append("  No resources")

        lines.append("")

        # Opponents
        lines.append("OPPONENTS:")
        for opp_color in board_state.get("players", {}).keys():
            opp_color = int(opp_color)
            if opp_color == player_color:
                continue

            opp_vp = self._get_player_vp(board_state, opp_color)
            opp_settlements = len(self._get_player_buildings(board_state, opp_color, 1))
            opp_cities = len(self._get_player_buildings(board_state, opp_color, 2))
            opp_roads = len(self._get_player_roads(board_state, opp_color))
            opp_cards = self._get_player_card_count(board_state, opp_color)

            threat = "WINNING!" if opp_vp >= 8 else "threatening" if opp_vp >= 6 else "building"
            lines.append(
                f"  Player {opp_color}: {opp_vp} VP "
                f"({opp_settlements}S, {opp_cities}C, {opp_roads}R, {opp_cards} cards) - {threat}"
            )

        return "\n".join(lines)

    def _get_player_vp(self, board_state: Dict, player: int) -> int:
        """Get player's visible victory points."""
        p_state = board_state.get("players", {}).get(player, {})
        vp_state = p_state.get("victoryPointsState", {})
        if isinstance(vp_state, dict):
            return sum(vp_state.values())
        return 0

    def _get_player_buildings(self, board_state: Dict, player: int, building_type: int) -> List[int]:
        """Get list of node IDs where player has buildings of given type."""
        buildings = []
        for corner_id, corner_data in board_state.get("corners", {}).items():
            if corner_data.get("owner") == player and corner_data.get("buildingType") == building_type:
                buildings.append(int(corner_id))
        return buildings

    def _get_player_roads(self, board_state: Dict, player: int) -> List[int]:
        """Get list of edge IDs where player has roads."""
        roads = []
        for edge_id, edge_data in board_state.get("edges", {}).items():
            if edge_data.get("owner") == player:
                roads.append(int(edge_id))
        return roads

    def _get_player_resources(self, board_state: Dict, player: int) -> Dict[str, int]:
        """Get player's resource counts."""
        p_state = board_state.get("players", {}).get(player, {})
        cards = p_state.get("resourceCards", {}).get("cards", [])
        counts = defaultdict(int)
        for card in cards:
            resource = RESOURCE_MAP.get(card, f"?{card}")
            counts[resource] += 1
        return dict(counts)

    def _get_player_card_count(self, board_state: Dict, player: int) -> int:
        """Get total number of cards player has."""
        p_state = board_state.get("players", {}).get(player, {})
        cards = p_state.get("resourceCards", {}).get("cards", [])
        return len(cards)

    def _get_affordable(self, resources: Dict[str, int]) -> List[str]:
        """Determine what can be afforded."""
        affordable = []

        # Settlement: wood, brick, sheep, wheat
        if (resources.get("WOOD", 0) >= 1 and resources.get("BRICK", 0) >= 1 and
            resources.get("SHEEP", 0) >= 1 and resources.get("WHEAT", 0) >= 1):
            affordable.append("settlement")

        # City: 2 wheat, 3 ore
        if resources.get("WHEAT", 0) >= 2 and resources.get("ORE", 0) >= 3:
            affordable.append("city")

        # Road: wood, brick
        if resources.get("WOOD", 0) >= 1 and resources.get("BRICK", 0) >= 1:
            affordable.append("road")

        # Dev card: sheep, wheat, ore
        if (resources.get("SHEEP", 0) >= 1 and resources.get("WHEAT", 0) >= 1 and
            resources.get("ORE", 0) >= 1):
            affordable.append("dev card")

        return affordable


def process_replay(
    raw_data: Dict[str, Any],
    game_id: str,
    winner_player: Optional[int] = None,
) -> List[TrainingExample]:
    """
    Process a Colonist replay into training examples.

    Args:
        raw_data: Raw Colonist API response
        game_id: Game identifier
        winner_player: Which player won (for labeling)

    Returns:
        List of TrainingExample objects
    """
    data = raw_data.get("data", raw_data)
    events = data.get("eventHistory", {}).get("events", [])

    formatter = ColonistObservationFormatter()
    examples = []

    # Build cumulative board state
    board_state = {
        "corners": {},  # corner_id -> {owner, buildingType}
        "edges": {},    # edge_id -> {owner, type}
        "players": {},  # player_id -> {resourceCards, victoryPointsState, ...}
        "turn": 0,
        "phase": "initial_placement",
        "last_dice": None,
    }

    for event_idx, event in enumerate(events):
        state_change = event.get("stateChange", {})
        delta_s = event.get("input", {}).get("deltaS", 0)

        # Determine action from state change
        action_type = None
        action_value = None
        acting_player = None

        map_state = state_change.get("mapState", {})

        # Settlement/city placement
        for corner_id, corner_data in map_state.get("tileCornerStates", {}).items():
            if corner_data and "owner" in corner_data:
                acting_player = corner_data["owner"]
                corner_int = int(corner_id)

                if corner_data.get("buildingType") == 1:
                    action_type = "BUILD_SETTLEMENT"
                    action_value = corner_int
                elif corner_data.get("buildingType") == 2:
                    action_type = "BUILD_CITY"
                    action_value = corner_int

                # Update board state
                board_state["corners"][corner_int] = corner_data

        # Road placement
        for edge_id, edge_data in map_state.get("tileEdgeStates", {}).items():
            if edge_data and "owner" in edge_data:
                acting_player = edge_data["owner"]
                edge_int = int(edge_id)
                action_type = "BUILD_ROAD"
                action_value = edge_int
                board_state["edges"][edge_int] = edge_data

        # Dice roll
        dice_state = state_change.get("diceState", {})
        if dice_state.get("diceThrown"):
            board_state["last_dice"] = (dice_state.get("dice1", 0), dice_state.get("dice2", 0))
            current_state = state_change.get("currentState", {})
            acting_player = current_state.get("currentTurnPlayerColor")
            action_type = "ROLL"
            action_value = board_state["last_dice"]

        # Bank trade (4:1 or port)
        trade_state = state_change.get("tradeState", {})
        bank_trade = trade_state.get("bankTrade")
        if bank_trade and not action_type:
            acting_player = bank_trade.get("player")
            offered = bank_trade.get("offeredResources", [])
            wanted = bank_trade.get("wantedResources", [])
            action_type = "BANK_TRADE"
            action_value = {
                "offered": [RESOURCE_MAP.get(r, f"?{r}") for r in offered],
                "wanted": [RESOURCE_MAP.get(r, f"?{r}") for r in wanted],
            }

        # Player trade offer
        active_offers = trade_state.get("activeOffers", {})
        for offer_id, offer in active_offers.items():
            if offer and "creator" in offer and not action_type:
                acting_player = offer["creator"]
                action_type = "OFFER_TRADE"
                action_value = {
                    "offered": [RESOURCE_MAP.get(r, f"?{r}") for r in offer.get("offeredResources", [])],
                    "wanted": [RESOURCE_MAP.get(r, f"?{r}") for r in offer.get("wantedResources", [])],
                }
                break

        # Robber movement
        robber_state = map_state.get("robberState", {})
        if robber_state.get("tileId") is not None and not action_type:
            current_state = state_change.get("currentState", {})
            acting_player = current_state.get("currentTurnPlayerColor")
            action_type = "MOVE_ROBBER"
            action_value = robber_state.get("tileId")

        # Dev card purchase
        dev_card_state = state_change.get("devCardState", {})
        if dev_card_state.get("purchased") and not action_type:
            current_state = state_change.get("currentState", {})
            acting_player = current_state.get("currentTurnPlayerColor")
            action_type = "BUY_DEV_CARD"
            action_value = dev_card_state.get("purchased")

        # Dev card play (knight, road building, etc)
        if dev_card_state.get("played") and not action_type:
            current_state = state_change.get("currentState", {})
            acting_player = current_state.get("currentTurnPlayerColor")
            card_type = dev_card_state.get("played", {}).get("type", "unknown")
            action_type = "PLAY_DEV_CARD"
            action_value = card_type

        # Update player states
        for player_id, p_state in state_change.get("playerStates", {}).items():
            player_int = int(player_id)
            if player_int not in board_state["players"]:
                board_state["players"][player_int] = {}
            # Deep update
            for key, value in p_state.items():
                if isinstance(value, dict) and key in board_state["players"][player_int]:
                    board_state["players"][player_int][key].update(value)
                else:
                    board_state["players"][player_int][key] = value

        # Update turn/phase
        current_state = state_change.get("currentState", {})
        if "completedTurns" in current_state:
            board_state["turn"] = current_state["completedTurns"]
            if board_state["turn"] >= 8:
                board_state["phase"] = "main_game"

        # Skip non-action events
        if not action_type or acting_player is None:
            continue

        # Generate observation (what the player saw BEFORE acting)
        observation = formatter.format(
            board_state=board_state,
            player_color=acting_player,
            turn=board_state["turn"],
            phase=board_state["phase"],
            last_dice=board_state["last_dice"],
        )

        # Format action string
        if action_type in ("BUILD_SETTLEMENT", "BUILD_CITY"):
            action_str = f"{action_type} node={action_value}"
        elif action_type == "BUILD_ROAD":
            action_str = f"{action_type} edge={action_value}"
        elif action_type == "ROLL":
            action_str = f"ROLL {action_value[0]}+{action_value[1]}={sum(action_value)}"
        elif action_type == "MOVE_ROBBER":
            action_str = f"MOVE_ROBBER tile={action_value}"
        elif action_type == "BANK_TRADE":
            offered = ",".join(action_value["offered"])
            wanted = ",".join(action_value["wanted"])
            action_str = f"BANK_TRADE give={offered} get={wanted}"
        elif action_type == "OFFER_TRADE":
            offered = ",".join(action_value["offered"])
            wanted = ",".join(action_value["wanted"])
            action_str = f"OFFER_TRADE give={offered} want={wanted}"
        elif action_type == "BUY_DEV_CARD":
            action_str = "BUY_DEV_CARD"
        elif action_type == "PLAY_DEV_CARD":
            action_str = f"PLAY_DEV_CARD type={action_value}"
        else:
            action_str = f"{action_type} {action_value}"

        # Create training example
        example = TrainingExample(
            game_id=game_id,
            event_index=event_idx,
            turn=board_state["turn"],
            phase=board_state["phase"],
            player=acting_player,
            observation=observation,
            action=action_str,
            action_type=action_type,
            action_value=action_value,
            is_winner=(acting_player == winner_player) if winner_player else False,
            time_taken_seconds=delta_s,
        )
        examples.append(example)

    return examples


def process_all_replays(
    input_dir: str,
    output_file: Optional[str] = None,
    max_games: Optional[int] = None,
    use_supabase: bool = False,
) -> Dict[str, int]:
    """
    Process all replays in a directory into training data.

    Args:
        input_dir: Directory containing raw replay JSON files
        output_file: Output JSONL file path (optional if using Supabase)
        max_games: Maximum games to process
        use_supabase: Save to Supabase instead of/in addition to file

    Returns:
        Stats dict
    """
    input_path = Path(input_dir)
    replay_files = list(input_path.glob("*.json"))

    if max_games:
        replay_files = replay_files[:max_games]

    stats = {"games": 0, "examples": 0, "errors": 0, "db_saved": 0}

    # Setup Supabase if requested
    db = None
    if use_supabase:
        try:
            from . import db as db_module
            if db_module.is_configured():
                db = db_module
                logger.info("Supabase configured, will save to database")
            else:
                logger.warning("Supabase not configured, falling back to file only")
        except ImportError:
            logger.warning("Could not import db module")

    # Setup file output
    file_handle = None
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        file_handle = open(output_file, "w")

    try:
        for replay_file in replay_files:
            game_id = replay_file.stem
            logger.info(f"Processing {game_id}...")

            try:
                with open(replay_file) as rf:
                    raw_data = json.load(rf)

                # Get winner from metadata if available
                metadata = raw_data.get("_metadata", {})
                winner = metadata.get("player_color") if metadata.get("result") == "win" else None

                examples = process_replay(raw_data, game_id, winner)

                # Write to file
                if file_handle:
                    for example in examples:
                        file_handle.write(json.dumps(example.to_dict()) + "\n")

                # Save to Supabase
                if db:
                    example_dicts = [ex.to_dict() for ex in examples]
                    saved = db.save_training_examples(example_dicts)
                    stats["db_saved"] += saved

                stats["games"] += 1
                stats["examples"] += len(examples)
                logger.info(f"  Generated {len(examples)} examples")

            except Exception as e:
                logger.error(f"  Error: {e}")
                stats["errors"] += 1

    finally:
        if file_handle:
            file_handle.close()

    logger.info(f"\nProcessed {stats['games']} games, {stats['examples']} examples, {stats['errors']} errors")
    if db:
        logger.info(f"Saved {stats['db_saved']} examples to Supabase")
    return stats


def process_from_supabase(max_games: Optional[int] = None) -> Dict[str, int]:
    """
    Process replays directly from Supabase (no local files needed).

    Args:
        max_games: Maximum games to process

    Returns:
        Stats dict
    """
    from . import db

    if not db.is_configured():
        raise ValueError("Supabase not configured")

    stats = {"games": 0, "examples": 0, "errors": 0}

    replays = db.list_replays(limit=max_games or 10000)
    logger.info(f"Found {len(replays)} replays in database")

    for replay_meta in replays:
        game_id = replay_meta["game_id"]
        logger.info(f"Processing {game_id}...")

        try:
            replay = db.get_replay(game_id)
            if not replay:
                continue

            raw_data = replay["data"]
            metadata = replay.get("metadata", {})
            winner = metadata.get("player_color") if metadata.get("result") == "win" else None

            examples = process_replay(raw_data, game_id, winner)

            # Save examples
            example_dicts = [ex.to_dict() for ex in examples]
            db.save_training_examples(example_dicts)

            stats["games"] += 1
            stats["examples"] += len(examples)
            logger.info(f"  Generated {len(examples)} examples")

        except Exception as e:
            logger.error(f"  Error: {e}")
            stats["errors"] += 1

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data from Colonist replays")
    parser.add_argument("--input-dir", type=str, default="data/raw_replays",
                       help="Directory containing raw replay JSON files")
    parser.add_argument("--output", type=str, default="data/training/policy_data.jsonl",
                       help="Output JSONL file")
    parser.add_argument("--max-games", type=int, help="Max games to process")
    parser.add_argument("--single", type=str, help="Process single replay file")
    parser.add_argument("--supabase", action="store_true",
                       help="Save to Supabase database (requires SUPABASE_URL and SUPABASE_KEY)")
    parser.add_argument("--from-db", action="store_true",
                       help="Process replays from Supabase instead of local files")
    parser.add_argument("--no-file", action="store_true",
                       help="Don't write to local file (use with --supabase)")

    args = parser.parse_args()

    if args.single:
        # Process single file for testing
        with open(args.single) as f:
            raw_data = json.load(f)

        examples = process_replay(raw_data, Path(args.single).stem)

        print(f"\nGenerated {len(examples)} examples\n")
        for ex in examples[:5]:
            print("=" * 60)
            print(f"Event {ex.event_index}: {ex.action_type} by Player {ex.player}")
            print(f"Turn: {ex.turn}, Phase: {ex.phase}")
            print(f"Action: {ex.action}")
            print(f"Time: {ex.time_taken_seconds}s")
            print("\nObservation:")
            print(ex.observation)

    elif args.from_db:
        # Process from Supabase
        stats = process_from_supabase(args.max_games)
        print(f"\nStats: {stats}")

    else:
        # Process from local files
        output_file = None if args.no_file else args.output
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        stats = process_all_replays(
            args.input_dir,
            output_file,
            args.max_games,
            use_supabase=args.supabase
        )
        print(f"\nStats: {stats}")
