"""
Colonist.io API client for scraping top player replays.

Endpoints discovered:
- GET /api/leaderboards-tabs/ - List leaderboard categories
- GET /api/leaderboards/{type} - Get rankings for a category
- GET /api/profile/{username} - Get player profile
- GET /api/profile/{username}/games - Get player's game history
"""

import httpx
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LeaderboardEntry:
    """A player on the leaderboard."""
    rank: int
    username: str
    rating: int
    games_played: int
    win_rate: Optional[float] = None


@dataclass
class GameHistoryEntry:
    """A game from a player's history."""
    game_id: str
    date: datetime
    mode: str  # "Classic4P", "CitiesAndKnights4P", etc.
    player_color: int
    result: str  # "win", "loss", "draw"
    players: List[Dict[str, Any]]
    replay_url: str


class ColonistAPI:
    """Client for Colonist.io API."""

    BASE_URL = "https://colonist.io"

    # Common headers to mimic browser
    HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "Referer": "https://colonist.io/leaderboards/Classic4P",
    }

    def __init__(self, cookies: Optional[Dict[str, str]] = None, jwt_token: Optional[str] = None):
        """
        Initialize the API client.

        Args:
            cookies: Optional session cookies for authenticated requests.
            jwt_token: JWT token from jwt_colonist.io cookie (for authenticated endpoints)
        """
        self.cookies = cookies or {}

        # Add JWT cookie if provided
        if jwt_token:
            self.cookies["jwt_colonist.io"] = jwt_token

        self.client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers=self.HEADERS,
            cookies=self.cookies,
            timeout=30.0,
            follow_redirects=True,
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def get_leaderboard_tabs(self) -> List[str]:
        """
        Get available leaderboard categories.

        Returns:
            List of leaderboard type names (e.g., ["Classic4P", "1v1", "CitiesAndKnights4P"])
        """
        response = await self.client.get("/api/leaderboards-tabs/")
        response.raise_for_status()
        data = response.json()

        # Extract tab names from response
        # Structure may vary - adjust based on actual response
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "tabs" in data:
            return data["tabs"]
        else:
            logger.warning(f"Unexpected leaderboard-tabs response: {data}")
            return list(data.keys()) if isinstance(data, dict) else []

    async def get_leaderboard(
        self,
        leaderboard_type: str = "Classic4P",
        start: int = 1,
        end: int = 100,
        search: str = "",
    ) -> List[LeaderboardEntry]:
        """
        Get rankings for a specific leaderboard.

        Args:
            leaderboard_type: Type of leaderboard (e.g., "Classic4P", "1v1")
            start: Starting rank (1-indexed)
            end: Ending rank
            search: Optional username search filter

        Returns:
            List of LeaderboardEntry objects
        """
        endpoint = f"/api/leaderboards/{leaderboard_type}/"
        params = {"start": start, "end": end, "search": search}

        try:
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_leaderboard_response(data)
        except Exception as e:
            logger.error(f"Failed to fetch leaderboard: {e}")
            raise Exception(f"Could not fetch leaderboard for {leaderboard_type}: {e}")

    def _parse_leaderboard_response(self, data: Any) -> List[LeaderboardEntry]:
        """Parse leaderboard API response into LeaderboardEntry objects."""
        entries = []

        # Debug: print first item to see structure
        if data and len(data) > 0:
            logger.debug(f"Sample leaderboard item: {data[0] if isinstance(data, list) else data}")

        # Handle different response structures
        items = data if isinstance(data, list) else data.get("players", data.get("rankings", []))

        for i, item in enumerate(items):
            try:
                entry = LeaderboardEntry(
                    rank=item.get("rank", i + 1),
                    username=item.get("username", ""),
                    rating=item.get("skillRating", 0),
                    games_played=item.get("totalGamesPlayed", 0),
                    win_rate=item.get("winRate"),
                )
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to parse leaderboard entry: {item}, error: {e}")

        return entries

    async def get_player_profile(self, username: str) -> Dict[str, Any]:
        """
        Get a player's profile.

        Args:
            username: Player's username

        Returns:
            Profile data dictionary
        """
        response = await self.client.get(f"/api/profile/{username}")
        response.raise_for_status()
        return response.json()

    async def get_player_games(
        self,
        username: str,
        limit: int = 50,
    ) -> List[GameHistoryEntry]:
        """
        Get a player's game history.

        Args:
            username: Player's username
            limit: Number of games to fetch

        Returns:
            List of GameHistoryEntry objects
        """
        endpoint = f"/api/profile/{username}/history"

        try:
            response = await self.client.get(endpoint)
            response.raise_for_status()
            data = response.json()
            games = self._parse_games_response(data, username)
            return games[:limit]
        except Exception as e:
            logger.error(f"Failed to fetch games for {username}: {e}")
            raise Exception(f"Could not fetch games for {username}: {e}")

    def _parse_games_response(self, data: Any, username: str) -> List[GameHistoryEntry]:
        """Parse games API response into GameHistoryEntry objects."""
        entries = []

        # History endpoint returns {"profileUserId": ..., "gameDatas": [...]}
        items = data.get("gameDatas", []) if isinstance(data, dict) else data

        # Game type mapping (from setting.gameType)
        GAME_TYPES = {
            0: "Classic4P",
            1: "Classic3P",
            2: "1v1",
            3: "CitiesAndKnights4P",
            4: "CitiesAndKnights3P",
            5: "Seafarers4P",
            6: "Seafarers",  # 1v1 seafarers or general
            7: "Custom",
            8: "Tournament",
        }

        for item in items:
            try:
                game_id = str(item.get("id", ""))
                if not game_id:
                    continue

                # Skip games without replay
                if not item.get("hasReplay", True):
                    continue

                # Get player data
                players = item.get("players", [])
                player_color = 0
                result = "unknown"

                for player in players:
                    if player.get("username", "").lower() == username.lower():
                        player_color = player.get("playerColor", 0)
                        # rank=1 means winner
                        result = "win" if player.get("rank") == 1 else "loss"
                        break

                # Parse date from startTime (Unix timestamp in ms, stored as string)
                start_time = item.get("startTime", "0")
                try:
                    game_date = datetime.fromtimestamp(int(start_time) / 1000)
                except (ValueError, TypeError):
                    game_date = datetime.now()

                # Get game mode from setting
                setting = item.get("setting", {})
                game_type = setting.get("gameType", 0)
                mode = GAME_TYPES.get(game_type, f"Unknown_{game_type}")

                entry = GameHistoryEntry(
                    game_id=game_id,
                    date=game_date,
                    mode=mode,
                    player_color=player_color,
                    result=result,
                    players=players,
                    replay_url=f"https://colonist.io/replay?gameId={game_id}&playerColor={player_color}",
                )
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to parse game entry: {item}, error: {e}")

        return entries

    async def get_top_player_replays(
        self,
        leaderboard_type: str = "Classic4P",
        top_n_players: int = 10,
        games_per_player: int = 20,
        wins_only: bool = True,
    ) -> List[GameHistoryEntry]:
        """
        Get replay URLs for top players' games.

        Args:
            leaderboard_type: Which leaderboard to pull from
            top_n_players: Number of top players to fetch
            games_per_player: Number of games per player
            wins_only: Only include won games (better training signal)

        Returns:
            List of GameHistoryEntry objects with replay URLs
        """
        all_games = []

        # Get top players
        logger.info(f"Fetching top {top_n_players} players from {leaderboard_type} leaderboard...")
        leaderboard = await self.get_leaderboard(leaderboard_type, start=1, end=top_n_players)

        for entry in leaderboard:
            logger.info(f"Fetching games for {entry.username} (rank #{entry.rank}, rating: {entry.rating})...")

            try:
                games = await self.get_player_games(entry.username, limit=games_per_player)

                # Filter for wins if requested
                if wins_only:
                    games = [g for g in games if g.result == "win"]

                all_games.extend(games)

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to fetch games for {entry.username}: {e}")
                continue

        logger.info(f"Collected {len(all_games)} replay URLs from {len(leaderboard)} players")
        return all_games


async def discover_api_endpoints(jwt_token: Optional[str] = None):
    """
    Helper to discover API endpoints by testing common patterns.
    Run this to find the correct endpoints for your Colonist instance.

    Args:
        jwt_token: Your JWT token from jwt_colonist.io cookie
    """
    async with ColonistAPI(jwt_token=jwt_token) as api:
        # Test leaderboard tabs
        print("Testing /api/leaderboards-tabs/...")
        try:
            tabs = await api.get_leaderboard_tabs()
            print(f"  Found tabs: {tabs}")
        except Exception as e:
            print(f"  Failed: {e}")

        # Test leaderboard
        print("\nTesting leaderboard endpoints...")
        top_player = None
        for leaderboard_type in ["Classic4P", "CitiesAndKnights4P"]:
            try:
                entries = await api.get_leaderboard(leaderboard_type, start=1, end=5)
                print(f"  {leaderboard_type}: {len(entries)} entries")
                if entries:
                    print(f"    Top player: {entries[0].username} (rating: {entries[0].rating})")
                    if not top_player:
                        top_player = entries[0].username
            except Exception as e:
                print(f"  {leaderboard_type}: Failed - {e}")

        # Test game history
        if top_player:
            print(f"\nTesting game history for {top_player}...")
            try:
                games = await api.get_player_games(top_player, limit=5)
                print(f"  Found {len(games)} games")
                for g in games[:3]:
                    print(f"    Game {g.game_id}: {g.mode} - {g.result}")
                    print(f"      Replay: {g.replay_url}")
            except Exception as e:
                print(f"  Failed: {e}")


if __name__ == "__main__":
    import sys
    import os

    # Get JWT from environment or command line
    jwt_token = os.environ.get("COLONIST_JWT")
    if len(sys.argv) > 1:
        jwt_token = sys.argv[1]

    if jwt_token:
        print(f"Using JWT token: {jwt_token[:20]}...")
    else:
        print("No JWT token provided. Set COLONIST_JWT env var or pass as argument.")
        print("Usage: python colonist_api.py <jwt_token>")
        print("       COLONIST_JWT=<token> python colonist_api.py")

    # Run endpoint discovery
    asyncio.run(discover_api_endpoints(jwt_token))
