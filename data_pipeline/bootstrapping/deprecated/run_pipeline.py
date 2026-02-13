#!/usr/bin/env python3
"""
End-to-end Training Data Pipeline

Usage:
    # Step 1: Scrape raw replays to data lake
    COLONIST_JWT="<token>" python run_pipeline.py scrape --max-games 100

    # Step 2: Process replays into training data
    python run_pipeline.py process --max-games 100

    # Step 3: Export final training data
    python run_pipeline.py export
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent / "scrapers"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Data lake paths
DATA_LAKE = Path(__file__).parent / "data"
RAW_REPLAYS = DATA_LAKE / "raw_replays"
PROCESSED = DATA_LAKE / "processed"
TRAINING = DATA_LAKE / "training"


def ensure_dirs():
    """Create data lake directories."""
    for d in [RAW_REPLAYS, PROCESSED, TRAINING]:
        d.mkdir(parents=True, exist_ok=True)


async def scrape_replays(max_games: int, jwt_token: str):
    """
    Step 1: Scrape raw replays from Colonist API.
    """
    from replay_api_scraper import ReplayAPIClient

    ensure_dirs()

    # Load game index
    index_file = Path(__file__).parent / "scrapers" / "4p_games_top100.json"
    if not index_file.exists():
        logger.error(f"Game index not found: {index_file}")
        logger.error("Run the game indexer first: python scrape_top_players.py --mode test-api")
        return

    with open(index_file) as f:
        games = json.load(f)

    logger.info(f"Loaded {len(games)} games from index")

    if max_games:
        games = games[:max_games]

    # Scrape each game
    stats = {"success": 0, "failed": 0, "skipped": 0}

    async with ReplayAPIClient(jwt_token=jwt_token) as client:
        for i, game in enumerate(games):
            game_id = game["game_id"]
            player_color = game.get("player_color", 0)

            output_file = RAW_REPLAYS / f"{game_id}.json"
            if output_file.exists():
                stats["skipped"] += 1
                continue

            logger.info(f"[{i+1}/{len(games)}] Scraping {game_id}...")

            try:
                raw_data = await client.get_replay_data(game_id, player_color)
                if raw_data:
                    # Add metadata
                    raw_data["_metadata"] = {
                        "game_id": game_id,
                        "player_color": player_color,
                        "username": game.get("username"),
                        "result": game.get("result"),
                    }

                    with open(output_file, "w") as f:
                        json.dump(raw_data, f)

                    stats["success"] += 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                logger.error(f"  Failed: {e}")
                stats["failed"] += 1

            await asyncio.sleep(0.3)  # Rate limit

    logger.info(f"\nScraping complete: {stats}")


def process_replays(max_games: Optional[int] = None):
    """
    Step 2: Process raw replays into observation-action pairs.
    """
    from colonist_to_engine import (
        parse_board_from_events,
        build_standard_mapping,
        generate_observation_text,
        analyze_colonist_board,
        COLONIST_RESOURCE_MAP,
    )

    ensure_dirs()

    replay_files = list(RAW_REPLAYS.glob("*.json"))
    if max_games:
        replay_files = replay_files[:max_games]

    logger.info(f"Processing {len(replay_files)} replays...")

    all_examples = []

    for i, replay_file in enumerate(replay_files):
        game_id = replay_file.stem
        logger.info(f"[{i+1}/{len(replay_files)}] Processing {game_id}...")

        try:
            with open(replay_file) as f:
                raw_data = json.load(f)

            metadata = raw_data.get("_metadata", {})
            player_color = metadata.get("player_color", 2)
            is_winner = metadata.get("result") == "win"

            # Parse events
            data = raw_data.get("data", raw_data)
            events = data.get("eventHistory", {}).get("events", [])

            mapping = build_standard_mapping()

            # Build cumulative state through events
            board_state = {
                "corners": {},
                "edges": {},
                "players": {},
                "turn_number": 0,
                "dice_roll": None,
            }

            examples_for_game = []

            for event_idx, event in enumerate(events):
                state_change = event.get("stateChange", {})
                delta_s = event.get("input", {}).get("deltaS", 0)

                # Determine if this is an actionable event
                action_type = None
                action_params = {}

                map_state = state_change.get("mapState", {})

                # Settlement/city placement
                for corner_id, corner_data in map_state.get("tileCornerStates", {}).items():
                    if corner_data and "owner" in corner_data:
                        board_state["corners"][int(corner_id)] = corner_data
                        acting_player = corner_data["owner"]
                        if corner_data.get("buildingType") == 1:
                            action_type = "BUILD_SETTLEMENT"
                            action_params = {"node_id": int(corner_id)}
                        elif corner_data.get("buildingType") == 2:
                            action_type = "BUILD_CITY"
                            action_params = {"node_id": int(corner_id)}

                # Road placement
                for edge_id, edge_data in map_state.get("tileEdgeStates", {}).items():
                    if edge_data and "owner" in edge_data:
                        board_state["edges"][int(edge_id)] = edge_data
                        acting_player = edge_data["owner"]
                        action_type = "BUILD_ROAD"
                        action_params = {"edge_id": int(edge_id)}

                # Dice roll
                dice_state = state_change.get("diceState", {})
                if dice_state.get("diceThrown") and "dice1" in dice_state:
                    board_state["dice_roll"] = (dice_state["dice1"], dice_state["dice2"])
                    action_type = "ROLL"
                    action_params = {"dice": board_state["dice_roll"]}
                    current_state = state_change.get("currentState", {})
                    acting_player = current_state.get("currentTurnPlayerColor", player_color)

                # Update player states
                for p_id, p_state in state_change.get("playerStates", {}).items():
                    if p_id not in board_state["players"]:
                        board_state["players"][int(p_id)] = {}
                    board_state["players"][int(p_id)].update(p_state)

                # Track turn
                current_state = state_change.get("currentState", {})
                if "completedTurns" in current_state:
                    board_state["turn_number"] = current_state["completedTurns"]

                # Skip non-actions
                if not action_type:
                    continue

                # Generate observation (from state BEFORE this action)
                # For simplicity, we'll use the current state - in production,
                # we'd track the previous state
                observation = _generate_simple_observation(
                    board_state,
                    acting_player,
                    mapping,
                )

                example = {
                    "game_id": game_id,
                    "event_index": event_idx,
                    "turn_number": board_state["turn_number"],
                    "acting_player": acting_player,
                    "observation": observation,
                    "action_type": action_type,
                    "action_params": action_params,
                    "time_taken_seconds": delta_s,
                    "is_winner": is_winner and acting_player == player_color,
                }
                examples_for_game.append(example)

            all_examples.extend(examples_for_game)
            logger.info(f"  Generated {len(examples_for_game)} examples")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save processed examples
    output_file = PROCESSED / "all_examples.json"
    with open(output_file, "w") as f:
        json.dump(all_examples, f, indent=2)

    logger.info(f"\nProcessed {len(all_examples)} total examples -> {output_file}")
    return all_examples


def _generate_simple_observation(board_state: dict, player_color: int, mapping) -> str:
    """Generate simple observation text."""
    lines = []
    turn = board_state.get("turn_number", 0)

    lines.append(f"=== GAME STATE (Turn {turn}) ===")
    lines.append("")

    # Phase
    if turn < 8:
        lines.append("Phase: initial_placement")
    else:
        lines.append("Phase: main_game")

    # VP
    player_data = board_state.get("players", {}).get(player_color, {})
    vp_state = player_data.get("victoryPointsState", {})
    total_vp = sum(vp_state.values()) if isinstance(vp_state, dict) else 0
    lines.append(f"Your VP: {total_vp}/10")

    # Dice
    if board_state.get("dice_roll"):
        d1, d2 = board_state["dice_roll"]
        lines.append(f"Last dice roll: {d1 + d2}")

    lines.append("")

    # Buildings
    lines.append("YOUR BUILDINGS:")
    my_settlements = [cid for cid, cd in board_state.get("corners", {}).items()
                     if cd.get("owner") == player_color and cd.get("buildingType") == 1]
    my_cities = [cid for cid, cd in board_state.get("corners", {}).items()
                if cd.get("owner") == player_color and cd.get("buildingType") == 2]
    my_roads = [eid for eid, ed in board_state.get("edges", {}).items()
               if ed.get("owner") == player_color]

    if my_settlements:
        lines.append(f"  Settlements ({len(my_settlements)}): nodes {my_settlements}")
    if my_cities:
        lines.append(f"  Cities ({len(my_cities)}): nodes {my_cities}")
    if my_roads:
        lines.append(f"  Roads ({len(my_roads)})")
    if not my_settlements and not my_cities:
        lines.append("  No buildings yet")

    lines.append("")

    # Resources
    lines.append("YOUR RESOURCES:")
    resource_cards = player_data.get("resourceCards", {})
    cards = resource_cards.get("cards", [])
    if cards:
        from colonist_to_engine import COLONIST_RESOURCE_MAP
        resource_counts = {}
        for card in cards:
            r_name = COLONIST_RESOURCE_MAP.get(card, "?")
            resource_counts[r_name] = resource_counts.get(r_name, 0) + 1
        for resource, count in sorted(resource_counts.items()):
            lines.append(f"  {resource}: {count}")
        lines.append(f"  Total: {len(cards)} cards")
    else:
        lines.append("  No resources")

    return "\n".join(lines)


def export_training_data():
    """
    Step 3: Export final training data in JSONL format.
    """
    ensure_dirs()

    processed_file = PROCESSED / "all_examples.json"
    if not processed_file.exists():
        logger.error("No processed data found. Run 'process' first.")
        return

    with open(processed_file) as f:
        examples = json.load(f)

    logger.info(f"Exporting {len(examples)} examples...")

    # Export as JSONL (one example per line)
    output_file = TRAINING / "policy_data.jsonl"
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    logger.info(f"Exported to {output_file}")

    # Stats
    action_types = {}
    winner_examples = 0
    for ex in examples:
        action_types[ex["action_type"]] = action_types.get(ex["action_type"], 0) + 1
        if ex.get("is_winner"):
            winner_examples += 1

    logger.info(f"\nStats:")
    logger.info(f"  Total examples: {len(examples)}")
    logger.info(f"  Winner examples: {winner_examples}")
    logger.info(f"  Action types: {action_types}")


def main():
    parser = argparse.ArgumentParser(description="Training Data Pipeline")
    parser.add_argument("command", choices=["scrape", "process", "export", "all"],
                       help="Pipeline command")
    parser.add_argument("--max-games", type=int, default=None,
                       help="Max games to process")

    args = parser.parse_args()

    jwt_token = os.environ.get("COLONIST_JWT")

    if args.command == "scrape":
        if not jwt_token:
            logger.error("Set COLONIST_JWT environment variable")
            sys.exit(1)
        asyncio.run(scrape_replays(args.max_games, jwt_token))

    elif args.command == "process":
        process_replays(args.max_games)

    elif args.command == "export":
        export_training_data()

    elif args.command == "all":
        if not jwt_token:
            logger.error("Set COLONIST_JWT environment variable")
            sys.exit(1)
        logger.info("Running full pipeline...")
        asyncio.run(scrape_replays(args.max_games, jwt_token))
        process_replays(args.max_games)
        export_training_data()


if __name__ == "__main__":
    main()
