"""Direct HTTP client for Colonist's replay data endpoint."""

import json

import httpx

from data_pipeline.bootstrapping.scrapers.replay_api_scraper._config import logger
from data_pipeline.json_coerce import as_dict
from data_pipeline.json_types import JsonDict


class ReplayAPIClient:
    """Client for Colonist replay API."""

    BASE_URL = "https://colonist.io"

    def __init__(self, jwt_token: str | None = None) -> None:
        self.jwt_token = jwt_token
        cookies: dict[str, str] = {}
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

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "ReplayAPIClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def get_replay_data(self, game_id: str, player_color: int = 0) -> JsonDict | None:
        """
        Fetch replay data from the API.

        Args:
            game_id: The game ID
            player_color: Player perspective (0-3)

        Returns:
            Raw API response or None if failed
        """
        url = "/api/replay/data-from-game-id"
        params: dict[str, str | int] = {"gameId": game_id, "playerColor": player_color}

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
            return as_dict(response.json())

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching replay {game_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for replay {game_id}: {e}")
            return None


__all__ = ["ReplayAPIClient"]
