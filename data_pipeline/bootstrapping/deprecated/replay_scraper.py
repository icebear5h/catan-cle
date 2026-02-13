"""
Colonist.io Replay Scraper

Uses Playwright to:
1. Open replay URLs in a browser
2. Intercept WebSocket messages via Chrome DevTools Protocol
3. Decode MessagePack game state messages
4. Step through the replay
5. Save structured game data for training
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import msgpack
from playwright.async_api import async_playwright, Page, CDPSession

from colonist_schema import (
    ReplayData,
    ReplayStep,
    GameStateSnapshot,
    ActionRecord,
    PlayerState,
    TileState,
    NodeState,
    EdgeState,
    ReplayDataStore,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WebSocketMessage:
    """A captured WebSocket message."""
    timestamp: datetime
    direction: str  # "sent" or "received"
    data: Any  # Decoded MessagePack data
    raw_bytes: Optional[bytes] = None


class ColonistMessageDecoder:
    """
    Decodes Colonist WebSocket messages.

    Message format (from reverse engineering):
    - Binary messages are MessagePack encoded
    - Structure: [type, subType, recipient, payload]
    """

    # Game update types (from SocketGameListen)
    class GameUpdate:
        FirstGameState = 0
        GameStateUpdated = 1
        BuildGame = 2
        SpectatorData = 3
        # ... add more as discovered

    def decode_message(self, raw_data: bytes) -> Optional[Dict[str, Any]]:
        """Decode a binary WebSocket message."""
        try:
            decoded = msgpack.unpackb(raw_data, raw=False)
            return self._parse_message(decoded)
        except Exception as e:
            logger.debug(f"Failed to decode message: {e}")
            return None

    def _parse_message(self, decoded: Any) -> Dict[str, Any]:
        """Parse decoded MessagePack data into structured format."""
        if isinstance(decoded, dict):
            return decoded

        if isinstance(decoded, (list, tuple)) and len(decoded) >= 2:
            # Try to parse as [type, subType, recipient, payload] or similar
            return {
                "raw": decoded,
                "type": decoded[0] if len(decoded) > 0 else None,
                "subType": decoded[1] if len(decoded) > 1 else None,
                "payload": decoded[-1] if len(decoded) > 2 else decoded[1],
            }

        return {"raw": decoded}

    def is_game_state_update(self, message: Dict[str, Any]) -> bool:
        """Check if this message contains a game state update."""
        # Look for game state indicators
        if "gameState" in str(message).lower():
            return True
        if "type" in message and message["type"] in [0, 1, 2]:  # Common game update types
            return True
        return False

    def extract_game_state(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract game state from a message if present."""
        # Handle nested structures
        payload = message.get("payload", message.get("data", message))

        if isinstance(payload, dict):
            # Look for game state fields
            if any(k in payload for k in ["tiles", "players", "gameState", "board"]):
                return payload

        return None


class ReplayScraper:
    """
    Scrapes game replays from Colonist.io.

    Uses Playwright to control a browser and intercept WebSocket traffic.
    """

    WEBSOCKET_URL = "wss://socket.svr.colonist.io"

    def __init__(
        self,
        headless: bool = False,
        slow_mo: int = 100,
        step_delay: float = 0.3,
        jwt_token: Optional[str] = None,
    ):
        """
        Initialize the scraper.

        Args:
            headless: Run browser in headless mode
            slow_mo: Slow down operations by this many ms (for debugging)
            step_delay: Delay between replay steps in seconds
            jwt_token: JWT token from jwt_colonist.io cookie for authentication
        """
        self.headless = headless
        self.slow_mo = slow_mo
        self.step_delay = step_delay
        self.jwt_token = jwt_token
        self.decoder = ColonistMessageDecoder()
        self.captured_messages: List[WebSocketMessage] = []
        self.game_states: List[Dict[str, Any]] = []

    async def scrape_replay(
        self,
        game_id: str,
        player_color: int = 0,
        max_steps: Optional[int] = None,
    ) -> Optional[ReplayData]:
        """
        Scrape a single replay.

        Args:
            game_id: The game ID to replay
            player_color: Which player's perspective (0-3)
            max_steps: Maximum steps to capture (None for all)

        Returns:
            ReplayData object or None if failed
        """
        replay_url = f"https://colonist.io/replay?gameId={game_id}&playerColor={player_color}"
        logger.info(f"Scraping replay: {replay_url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
            )

            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            )

            # Add JWT cookie for authentication
            if self.jwt_token:
                await context.add_cookies([{
                    "name": "jwt_colonist.io",
                    "value": self.jwt_token,
                    "domain": "colonist.io",
                    "path": "/",
                }])

            page = await context.new_page()

            # Set up WebSocket interception via CDP
            cdp = await context.new_cdp_session(page)
            await self._setup_websocket_interception(cdp)

            try:
                # Navigate to replay
                await page.goto(replay_url, wait_until="networkidle")
                await asyncio.sleep(2)  # Wait for WebSocket connection

                # Wait for game canvas to load
                await page.wait_for_selector("#game-canvas", timeout=30000)
                logger.info("Game canvas loaded")

                # Find and click through replay steps
                step_count = 0
                while True:
                    if max_steps and step_count >= max_steps:
                        break

                    # Try to click next step button
                    stepped = await self._click_next_step(page)
                    if not stepped:
                        logger.info("No more steps available")
                        break

                    step_count += 1
                    await asyncio.sleep(self.step_delay)

                    if step_count % 10 == 0:
                        logger.info(f"Captured {step_count} steps...")

                logger.info(f"Captured {step_count} total steps, {len(self.game_states)} game states")

                # Build ReplayData from captured messages
                replay = self._build_replay_data(game_id, player_color)
                return replay

            except Exception as e:
                logger.error(f"Failed to scrape replay: {e}")
                raise
            finally:
                await browser.close()

    async def _setup_websocket_interception(self, cdp: CDPSession):
        """Set up CDP to intercept WebSocket messages."""
        await cdp.send("Network.enable")

        # Listen for WebSocket events
        cdp.on("Network.webSocketFrameReceived", self._on_ws_frame_received)
        cdp.on("Network.webSocketFrameSent", self._on_ws_frame_sent)

        logger.info("WebSocket interception enabled")

    def _on_ws_frame_received(self, event: Dict[str, Any]):
        """Handle received WebSocket frame."""
        try:
            frame = event.get("response", {})
            payload = frame.get("payloadData", "")

            # Decode base64 binary data
            if payload:
                import base64
                try:
                    raw_bytes = base64.b64decode(payload)
                    decoded = self.decoder.decode_message(raw_bytes)
                    if decoded:
                        msg = WebSocketMessage(
                            timestamp=datetime.now(),
                            direction="received",
                            data=decoded,
                            raw_bytes=raw_bytes,
                        )
                        self.captured_messages.append(msg)

                        # Extract game state if present
                        game_state = self.decoder.extract_game_state(decoded)
                        if game_state:
                            self.game_states.append(game_state)

                except Exception as e:
                    logger.debug(f"Failed to decode WS frame: {e}")
        except Exception as e:
            logger.debug(f"Error processing WS frame: {e}")

    def _on_ws_frame_sent(self, event: Dict[str, Any]):
        """Handle sent WebSocket frame."""
        # We can capture outgoing messages too if needed
        pass

    async def _click_next_step(self, page: Page) -> bool:
        """
        Click the next step button in the replay UI.

        Returns True if step was taken, False if no more steps.
        """
        # Common selectors for next step button
        selectors = [
            "button:has-text('Next')",
            "button:has-text('>')",
            ".replay-next",
            ".next-step",
            "[data-testid='next-step']",
            "button.next",
            # Fallback: look for forward arrow icons
            "button svg[data-icon='forward']",
            "button svg[data-icon='step-forward']",
        ]

        for selector in selectors:
            try:
                button = page.locator(selector).first
                if await button.is_visible():
                    await button.click()
                    return True
            except Exception:
                continue

        # Try keyboard shortcut
        try:
            await page.keyboard.press("ArrowRight")
            return True
        except Exception:
            pass

        return False

    def _build_replay_data(self, game_id: str, player_color: int) -> ReplayData:
        """Build ReplayData from captured messages."""
        steps = []

        for i, state in enumerate(self.game_states):
            # Create game state snapshot
            snapshot = self._parse_game_state(state, game_id, i)

            # Infer action from state diff (if we have previous state)
            action = ActionRecord(
                action_type="UNKNOWN",
                player_index=state.get("currentPlayer", 0),
                details=state.get("lastAction", {}),
            )

            step = ReplayStep(
                step_number=i,
                state_before=snapshot,
                action=action,
            )
            steps.append(step)

        # Extract player info and winner
        players = []
        winner_index = 0

        if self.game_states:
            last_state = self.game_states[-1]
            player_data = last_state.get("players", [])
            for i, p in enumerate(player_data):
                players.append({
                    "index": i,
                    "username": p.get("username", f"Player{i}"),
                    "color": p.get("color", i),
                    "final_score": p.get("victoryPoints", 0),
                })
                if p.get("isWinner") or p.get("victoryPoints", 0) >= 10:
                    winner_index = i

        return ReplayData(
            game_id=game_id,
            mode=self._infer_game_mode(),
            map_type="Classic4P",  # TODO: Extract from game data
            scraped_at=datetime.now(),
            players=players,
            winner_index=winner_index,
            steps=steps,
            total_turns=len(steps),
        )

    def _parse_game_state(
        self,
        raw_state: Dict[str, Any],
        game_id: str,
        step_number: int,
    ) -> GameStateSnapshot:
        """Parse raw game state into structured format."""
        # Extract tiles
        tiles = []
        for tile_data in raw_state.get("tiles", []):
            tiles.append(TileState(
                index=tile_data.get("index", 0),
                resource=tile_data.get("resource"),
                number=tile_data.get("number"),
                has_robber=tile_data.get("hasRobber", False),
            ))

        # Extract nodes
        nodes = []
        for node_data in raw_state.get("nodes", raw_state.get("corners", [])):
            nodes.append(NodeState(
                index=node_data.get("index", 0),
                building=node_data.get("building"),
                owner=node_data.get("owner"),
            ))

        # Extract edges
        edges = []
        for edge_data in raw_state.get("edges", raw_state.get("roads", [])):
            edges.append(EdgeState(
                index=edge_data.get("index", 0),
                has_road=edge_data.get("hasRoad", False),
                owner=edge_data.get("owner"),
            ))

        # Extract players
        players = []
        for p_data in raw_state.get("players", []):
            players.append(PlayerState(
                index=p_data.get("index", 0),
                color=p_data.get("color", ""),
                username=p_data.get("username", ""),
                victory_points=p_data.get("victoryPoints", 0),
                resources=p_data.get("resources", {}),
                dev_cards=p_data.get("devCards", []),
                settlements=p_data.get("settlements", []),
                cities=p_data.get("cities", []),
                roads=p_data.get("roads", []),
            ))

        return GameStateSnapshot(
            game_id=game_id,
            step_number=step_number,
            tiles=tiles,
            nodes=nodes,
            edges=edges,
            players=players,
            current_player=raw_state.get("currentPlayer", 0),
            phase=raw_state.get("phase", "main"),
            dice_roll=raw_state.get("diceRoll"),
            turn_number=raw_state.get("turnNumber", 0),
            playable_actions=raw_state.get("playableActions", []),
        )

    def _infer_game_mode(self) -> str:
        """Infer game mode from captured data."""
        # Look for C&K indicators
        for state in self.game_states:
            if "cityImprovements" in str(state) or "knights" in str(state):
                return "CitiesAndKnights4P"
            if "ships" in str(state) or "harbors" in str(state):
                return "Seafarers4P"
        return "Classic4P"


async def scrape_multiple_replays(
    game_ids: List[str],
    output_dir: str = "./data/replays",
    headless: bool = True,
    max_concurrent: int = 1,
) -> List[ReplayData]:
    """
    Scrape multiple replays.

    Args:
        game_ids: List of game IDs to scrape
        output_dir: Directory to save replays
        headless: Run browsers in headless mode
        max_concurrent: Max concurrent scrapers

    Returns:
        List of successfully scraped ReplayData objects
    """
    store = ReplayDataStore(output_dir)
    scraped = []

    for game_id in game_ids:
        # Check if already scraped
        if game_id in store.list_replays():
            logger.info(f"Skipping {game_id} - already scraped")
            continue

        try:
            scraper = ReplayScraper(headless=headless)
            replay = await scraper.scrape_replay(game_id)

            if replay:
                store.save_replay(replay)
                scraped.append(replay)
                logger.info(f"Saved replay {game_id}")

        except Exception as e:
            logger.error(f"Failed to scrape {game_id}: {e}")

        # Rate limiting
        await asyncio.sleep(2)

    return scraped


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python replay_scraper.py <game_id> [--headless]")
        print("Example: python replay_scraper.py 194340297")
        print("\nSet COLONIST_JWT env var for authentication:")
        print("  COLONIST_JWT=<token> python replay_scraper.py <game_id>")
        sys.exit(1)

    game_id = sys.argv[1]
    headless = "--headless" in sys.argv
    jwt_token = os.environ.get("COLONIST_JWT")

    if jwt_token:
        # Decode and check expiration
        import base64
        import json as json_mod
        from datetime import datetime as dt
        try:
            parts = jwt_token.split('.')
            payload = parts[1] + '=' * (4 - len(parts[1]) % 4)
            data = json_mod.loads(base64.urlsafe_b64decode(payload))
            exp = dt.fromtimestamp(data.get('exp', 0))
            if dt.now() > exp:
                print(f"ERROR: JWT token expired on {exp}")
                print("Get a fresh token from browser DevTools > Application > Cookies > jwt_colonist.io")
                sys.exit(1)
            print(f"Using JWT for user: {data.get('username')} (expires: {exp})")
        except Exception as e:
            print(f"Warning: Could not validate JWT: {e}")
    else:
        print("Warning: No JWT token provided. Replay viewing requires authentication.")
        print("Get your token from browser DevTools > Application > Cookies > jwt_colonist.io")

    async def main():
        scraper = ReplayScraper(headless=headless, jwt_token=jwt_token)
        replay = await scraper.scrape_replay(game_id)

        if replay:
            store = ReplayDataStore()
            store.save_replay(replay)
            print(f"\nSaved replay with {len(replay.steps)} steps")
            print(f"Players: {[p['username'] for p in replay.players]}")
            print(f"Winner: Player {replay.winner_index}")
        else:
            print("Failed to scrape replay")

    asyncio.run(main())
