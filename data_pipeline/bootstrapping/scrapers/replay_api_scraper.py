#!/usr/bin/env python3
"""
Colonist.io Replay API Scraper

Uses the direct replay data API endpoint instead of WebSocket interception.
Much simpler and more reliable than browser-based scraping.

API Endpoint:
    GET /api/replay/data-from-game-id?gameId={id}&playerColor={color}

Returns full event history with structured state changes.
"""

import argparse
import asyncio
import httpx
import json
import logging
import os
import sys
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_FILE = (
    PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "indexes" / "4p_games_top100.json"
)
DEFAULT_STAGING_DIR = PROJECT_ROOT / "artifacts" / "staging" / "colonist" / "replays"


# Resource enum mapping (from Colonist)
RESOURCE_ENUM = {
    1: "wood",
    2: "brick",
    3: "sheep",
    4: "wheat",
    5: "ore",
}

# Building type mapping
BUILDING_TYPE = {
    1: "settlement",
    2: "city",
}

# Edge type mapping
EDGE_TYPE = {
    1: "road",
    2: "ship",
}

# Action state mapping
ACTION_STATE = {
    0: "roll_dice",
    1: "place_settlement",
    2: "place_city",
    3: "place_road",
    4: "play_turn",
    5: "discard",
    6: "move_robber",
    7: "steal",
}

# Game log type mapping (partial)
LOG_TYPE = {
    4: "place_piece",
    5: "build_piece",
    10: "roll_dice",
    44: "end_turn",
    47: "receive_resources",
    118: "trade_offer",
}


@dataclass
class ReplayEvent:
    """A single event in the replay."""
    event_index: int
    delta_seconds: float
    state_change: Dict[str, Any]

    # Parsed fields
    action_type: Optional[str] = None
    acting_player: Optional[int] = None
    dice_roll: Optional[tuple] = None
    resources_gained: Optional[Dict[int, List[int]]] = None  # player -> resources
    building_placed: Optional[Dict[str, Any]] = None
    road_placed: Optional[Dict[str, Any]] = None
    trade_offer: Optional[Dict[str, Any]] = None


@dataclass
class ParsedReplay:
    """Fully parsed replay data."""
    game_id: str
    player_color: int
    total_events: int
    events: List[ReplayEvent]
    initial_state: Optional[Dict[str, Any]] = None
    final_state: Optional[Dict[str, Any]] = None
    winner: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "player_color": self.player_color,
            "total_events": self.total_events,
            "events": [asdict(e) for e in self.events],
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "winner": self.winner,
        }


class ReplayAPIClient:
    """Client for Colonist replay API."""

    BASE_URL = "https://colonist.io"

    def __init__(self, jwt_token: Optional[str] = None):
        self.jwt_token = jwt_token
        cookies = {}
        if jwt_token:
            cookies["jwt_colonist.io"] = jwt_token

        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            cookies=cookies,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            },
            timeout=60.0,
        )

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def get_replay_data(self, game_id: str, player_color: int = 0) -> Optional[Dict[str, Any]]:
        """
        Fetch replay data from the API.

        Args:
            game_id: The game ID
            player_color: Player perspective (0-3)

        Returns:
            Raw API response or None if failed
        """
        url = "/api/replay/data-from-game-id"
        params = {"gameId": game_id, "playerColor": player_color}

        try:
            response = await self.client.get(url, params=params)

            if response.status_code == 401:
                logger.error("Authentication required - provide valid JWT token")
                return None
            elif response.status_code == 403:
                logger.error(
                    "Access denied from the direct replay API. Colonist may require browser "
                    "session/Cloudflare state; try replay_playwright_scraper.py."
                )
                return None
            elif response.status_code == 404:
                logger.error(f"Replay not found: {game_id}")
                return None

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching replay {game_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for replay {game_id}: {e}")
            return None


def parse_replay_event(index: int, event: Dict[str, Any]) -> ReplayEvent:
    """Parse a single replay event into structured format."""
    delta_s = event.get("input", {}).get("deltaS", 0)
    state_change = event.get("stateChange", {})

    parsed = ReplayEvent(
        event_index=index,
        delta_seconds=delta_s,
        state_change=state_change,
    )

    # Parse current state changes
    current_state = state_change.get("currentState", {})
    if "currentTurnPlayerColor" in current_state:
        parsed.acting_player = current_state["currentTurnPlayerColor"]

    # Parse dice roll
    dice_state = state_change.get("diceState", {})
    if dice_state.get("diceThrown") and "dice1" in dice_state and "dice2" in dice_state:
        parsed.dice_roll = (dice_state["dice1"], dice_state["dice2"])
        parsed.action_type = "roll_dice"

    # Parse building placements
    map_state = state_change.get("mapState", {})
    corner_states = map_state.get("tileCornerStates", {})
    if corner_states:
        for corner_id, corner_data in corner_states.items():
            if "owner" in corner_data and "buildingType" in corner_data:
                parsed.building_placed = {
                    "corner_id": int(corner_id),
                    "owner": corner_data["owner"],
                    "type": BUILDING_TYPE.get(corner_data["buildingType"], "unknown"),
                }
                parsed.action_type = "place_settlement" if corner_data["buildingType"] == 1 else "place_city"
                parsed.acting_player = corner_data["owner"]

    # Parse road placements
    edge_states = map_state.get("tileEdgeStates", {})
    if edge_states:
        for edge_id, edge_data in edge_states.items():
            if "owner" in edge_data and "type" in edge_data:
                parsed.road_placed = {
                    "edge_id": int(edge_id),
                    "owner": edge_data["owner"],
                    "type": EDGE_TYPE.get(edge_data["type"], "unknown"),
                }
                parsed.action_type = "place_road"
                parsed.acting_player = edge_data["owner"]

    # Parse resource distribution
    player_states = state_change.get("playerStates", {})
    if player_states:
        resources_gained = {}
        for player_id, p_state in player_states.items():
            resource_cards = p_state.get("resourceCards", {})
            if "cards" in resource_cards:
                resources_gained[int(player_id)] = resource_cards["cards"]
        if resources_gained:
            parsed.resources_gained = resources_gained

    # Parse trade offers
    trade_state = state_change.get("tradeState", {})
    active_offers = trade_state.get("activeOffers", {})
    if active_offers:
        for offer_id, offer_data in active_offers.items():
            if offer_data and "creator" in offer_data:
                parsed.trade_offer = {
                    "offer_id": offer_id,
                    "creator": offer_data["creator"],
                    "offered": [RESOURCE_ENUM.get(r, r) for r in offer_data.get("offeredResources", [])],
                    "wanted": [RESOURCE_ENUM.get(r, r) for r in offer_data.get("wantedResources", [])],
                    "responses": offer_data.get("playerResponses", {}),
                }
                parsed.action_type = "trade_offer"
                parsed.acting_player = offer_data["creator"]

    return parsed


def parse_replay(game_id: str, player_color: int, raw_data: Dict[str, Any]) -> ParsedReplay:
    """Parse raw API response into structured replay."""
    data = raw_data.get("data", raw_data)
    event_history = data.get("eventHistory", {})
    raw_events = event_history.get("events", [])

    events = []
    for i, raw_event in enumerate(raw_events):
        parsed_event = parse_replay_event(i, raw_event)
        events.append(parsed_event)

    # Try to determine winner from final VP states
    winner = None
    if events:
        last_event = events[-1]
        player_states = last_event.state_change.get("playerStates", {})
        for player_id, p_state in player_states.items():
            vp_state = p_state.get("victoryPointsState", {})
            total_vp = sum(vp_state.values()) if isinstance(vp_state, dict) else 0
            if total_vp >= 10:
                winner = int(player_id)
                break

    return ParsedReplay(
        game_id=game_id,
        player_color=player_color,
        total_events=len(events),
        events=events,
        initial_state=data.get("initialState"),
        final_state=data.get("finalState"),
        winner=winner,
    )


async def scrape_replays_from_index(
    index_file: str,
    output_dir: str,
    jwt_token: str,
    max_games: Optional[int] = None,
    max_attempts: Optional[int] = None,
    skip_existing: bool = True,
) -> Dict[str, int]:
    """
    Scrape replays from the game index file.

    Args:
        index_file: Path to 4p_games_top100.json
        output_dir: Directory to save raw replay payloads
        jwt_token: Authentication token
        max_games: Maximum successful new games to scrape (None for all)
        max_attempts: Maximum non-skipped games to try before stopping
        skip_existing: Skip already-scraped games

    Returns:
        Stats dict with success/failure counts
    """
    # Load game index
    with open(index_file) as f:
        games = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed": 0, "skipped": 0}
    attempts = 0

    async with ReplayAPIClient(jwt_token=jwt_token) as client:
        for i, game in enumerate(games):
            game_id = game["game_id"]
            player_color = game.get("player_color", 0)

            # Check if already scraped
            output_file = output_path / f"{game_id}.json"
            if skip_existing and output_file.exists():
                stats["skipped"] += 1
                continue

            if max_games is not None and stats["success"] >= max_games:
                break
            if max_attempts is not None and attempts >= max_attempts:
                break
            attempts += 1

            logger.info(f"[{i+1}/{len(games)}] Scraping {game_id} (player {game['username']})...")

            try:
                raw_data = await client.get_replay_data(game_id, player_color)
                if raw_data:
                    replay = parse_replay(game_id, player_color, raw_data)

                    with open(output_file, "w") as f:
                        json.dump(raw_data, f)

                    stats["success"] += 1
                    logger.info(f"  Saved: {replay.total_events} events")
                else:
                    stats["failed"] += 1
                    logger.warning("  Failed: No data returned")

            except Exception as e:
                stats["failed"] += 1
                logger.error(f"  Failed: {e}")

            # Rate limiting
            await asyncio.sleep(0.5)

    return stats


def validate_jwt(token: str) -> bool:
    """Validate JWT token hasn't expired."""
    try:
        parts = token.split('.')
        payload = parts[1] + '=' * (4 - len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        exp = datetime.fromtimestamp(data.get('exp', 0))

        if datetime.now() > exp:
            logger.error(f"JWT token expired on {exp}")
            return False

        logger.info(f"JWT valid for user: {data.get('username')} (expires: {exp})")
        return True
    except Exception as e:
        logger.warning(f"Could not validate JWT: {e}")
        return True  # Proceed anyway


async def main():
    """CLI entry point."""
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Scrape Colonist.io replays via API")
    parser.add_argument("--game-id", type=str, help="Single game ID to scrape")
    parser.add_argument(
        "--index-file", type=str, default=str(DEFAULT_INDEX_FILE), help="Game index file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_STAGING_DIR), help="Output directory"
    )
    parser.add_argument("--max-games", type=int, help="Max successful new games to scrape")
    parser.add_argument("--max-attempts", type=int, help="Max non-skipped games to try")
    parser.add_argument("--player-color", type=int, default=0, help="Player color perspective")

    args = parser.parse_args()

    jwt_token = os.environ.get("COLONIST_JWT")
    if not jwt_token:
        print("ERROR: Set COLONIST_JWT environment variable")
        print("This direct API scraper is a debug fallback and may still 403 with only JWT.")
        print("For replay downloads, prefer replay_playwright_scraper.py.")
        print("Get JWT from DevTools > Application > Cookies > jwt_colonist.io")
        sys.exit(1)

    if not validate_jwt(jwt_token):
        sys.exit(1)

    if args.game_id:
        async with ReplayAPIClient(jwt_token=jwt_token) as client:
            raw_data = await client.get_replay_data(args.game_id, args.player_color)
        if not raw_data:
            print("Failed to scrape replay")
            sys.exit(1)

        replay = parse_replay(args.game_id, args.player_color, raw_data)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{args.game_id}.json"
        output_file.write_text(json.dumps(raw_data))
        print(f"\nGame {replay.game_id}: {replay.total_events} events")
        print(f"Saved raw payload to {output_file}")
    else:
        # Batch mode from index
        stats = await scrape_replays_from_index(
            index_file=args.index_file,
            output_dir=args.output_dir,
            jwt_token=jwt_token,
            max_games=args.max_games,
            max_attempts=args.max_attempts,
        )
        print("\nScraping complete:")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")


if __name__ == "__main__":
    asyncio.run(main())
