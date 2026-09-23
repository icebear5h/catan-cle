"""Async HTTP client for the Colonist.io leaderboard, profile, and history APIs."""

import asyncio

import httpx

from data_pipeline.bootstrapping.scrapers.colonist_api._models import (
    GameHistoryEntry,
    LeaderboardEntry,
    logger,
)
from data_pipeline.bootstrapping.scrapers.colonist_api._parsing import (
    parse_games_response,
    parse_leaderboard_response,
)
from data_pipeline.json_coerce import (
    as_dict,
    as_str,
    as_str_list,
)
from data_pipeline.json_types import JsonDict, JsonValue


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

    def __init__(
        self,
        cookies: dict[str, str] | None = None,
        jwt_token: str | None = None,
    ) -> None:
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

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self) -> "ColonistAPI":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def get_leaderboard_tabs(self) -> list[str]:
        """
        Get available leaderboard categories.

        Returns:
            List of leaderboard type names (e.g., ["Classic4P", "1v1", "CitiesAndKnights4P"])
        """
        response = await self.client.get("/api/leaderboards-tabs/")
        response.raise_for_status()
        data: JsonValue = response.json()

        # Extract tab names from response
        # Structure may vary - adjust based on actual response
        if isinstance(data, list):
            return as_str_list(data)
        elif isinstance(data, dict) and "tabs" in data:
            return as_str_list(data["tabs"])
        else:
            logger.warning(f"Unexpected leaderboard-tabs response: {data}")
            return list(data.keys()) if isinstance(data, dict) else []

    async def get_leaderboard(
        self,
        leaderboard_type: str = "Classic4P",
        start: int = 1,
        end: int = 100,
        search: str = "",
    ) -> list[LeaderboardEntry]:
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
        params: dict[str, str | int] = {"start": start, "end": end, "search": search}

        try:
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            data: JsonValue = response.json()
            return self._parse_leaderboard_response(data)
        except Exception as e:
            logger.error(f"Failed to fetch leaderboard: {e}")
            raise Exception(f"Could not fetch leaderboard for {leaderboard_type}: {e}")

    def _parse_leaderboard_response(self, data: JsonValue) -> list[LeaderboardEntry]:
        """Parse leaderboard API response into LeaderboardEntry objects."""
        return parse_leaderboard_response(data)

    async def get_player_profile(self, username: str) -> JsonDict:
        """
        Get a player's profile.

        Args:
            username: Player's username

        Returns:
            Profile data dictionary
        """
        response = await self.client.get(f"/api/profile/{username}")
        response.raise_for_status()
        return as_dict(response.json())

    async def get_user_state(self) -> JsonDict:
        """Get authenticated user state from the current JWT cookie."""
        response = await self.client.get("/api/user-state")
        response.raise_for_status()
        return as_dict(response.json())

    async def get_current_username(self) -> str:
        """Get the authenticated Colonist username from /api/user-state."""
        data = await self.get_user_state()
        user_state = data.get("userState", {})
        username = user_state.get("username") if isinstance(user_state, dict) else None
        if not username:
            raise Exception("Could not resolve authenticated username from /api/user-state")
        return as_str(username)

    async def get_player_games(
        self,
        username: str,
        limit: int | None = 50,
    ) -> list[GameHistoryEntry]:
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
            data: JsonValue = response.json()
            games = self._parse_games_response(data, username)
            return games[:limit] if limit is not None else games
        except Exception as e:
            logger.error(f"Failed to fetch games for {username}: {e}")
            raise Exception(f"Could not fetch games for {username}: {e}")

    def _parse_games_response(self, data: JsonValue, username: str) -> list[GameHistoryEntry]:
        """Parse games API response into GameHistoryEntry objects."""
        return parse_games_response(data, username)

    async def get_top_player_replays(
        self,
        leaderboard_type: str = "Classic4P",
        top_n_players: int = 10,
        games_per_player: int = 20,
        wins_only: bool = True,
    ) -> list[GameHistoryEntry]:
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
        all_games: list[GameHistoryEntry] = []

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


__all__ = ["ColonistAPI"]
