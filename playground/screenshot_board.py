"""
Frontend board screenshotter: GameEngine -> Playwright -> PNG bytes.

Captures the React+Colonist-asset board by:
1. Starting Flask backend + Vite dev server (if not already running)
2. Injecting game state via /api/inject-state
3. Screenshotting the rendered HexBoard with Playwright

Usage (async — works in Jupyter):
    from playground.screenshot_board import FrontendScreenshotter

    screenshotter = FrontendScreenshotter()
    await screenshotter.start()

    png_bytes = await screenshotter.screenshot(game)

    await screenshotter.stop()

Or as an async context manager:
    async with FrontendScreenshotter() as s:
        png_bytes = await s.screenshot(game)
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import io

import httpx
from PIL import Image
from playwright.async_api import async_playwright

from cle.game_engine.game import GameEngine
from cle.game_engine.json import GameEncoder
from playground.game_viewer.serialize import serialize_game_for_inject

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GAME_VIEWER_DIR = Path(__file__).resolve().parent / "game_viewer"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

FLASK_PORT = 5001
VITE_PORT = 5173
FLASK_URL = f"http://localhost:{FLASK_PORT}"
VITE_URL = f"http://localhost:{VITE_PORT}"


def _port_in_use(port: int) -> bool:
    """Check if a port is already listening."""
    try:
        with httpx.Client(timeout=2.0) as client:
            client.get(f"http://localhost:{port}")
        return True
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False


class FrontendScreenshotter:
    """Captures board screenshots from the React frontend via Playwright."""

    def __init__(
        self,
        flask_port: int = FLASK_PORT,
        vite_port: int = VITE_PORT,
        headless: bool = True,
        board_width: int = 800,
        board_height: int = 800,
        crop_pct: float = 0.15,
        vertical_offset_pct: float = 0.03,
        horizontal_offset_pct: float = 0.0,
    ):
        self.flask_port = flask_port
        self.vite_port = vite_port
        self.headless = headless
        self.board_width = board_width
        self.board_height = board_height
        self.crop_pct = crop_pct
        self.vertical_offset_pct = vertical_offset_pct
        self.horizontal_offset_pct = horizontal_offset_pct

        self._flask_proc: Optional[subprocess.Popen] = None
        self._vite_proc: Optional[subprocess.Popen] = None
        self._playwright = None
        self._browser = None
        self._page = None

    async def start(self):
        """Start servers and browser. Skips servers already running."""
        self._start_flask()
        self._start_vite()
        await self._start_browser()

    async def stop(self):
        """Shut down browser and any servers we started."""
        if self._page:
            await self._page.close()
            self._page = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        if self._vite_proc:
            self._vite_proc.terminate()
            self._vite_proc.wait(timeout=5)
            self._vite_proc = None
        if self._flask_proc:
            self._flask_proc.terminate()
            self._flask_proc.wait(timeout=5)
            self._flask_proc = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()

    async def screenshot(
        self,
        game: GameEngine,
        settle_ms: int = 600,
    ) -> bytes:
        """Render a GameEngine state in the frontend and return PNG bytes.

        Args:
            game: Engine GameEngine object at any point in play.
            settle_ms: Milliseconds to wait after state injection for render.

        Returns:
            PNG image bytes of the board.
        """
        if not self._page:
            raise RuntimeError("Screenshotter not started. Call .start() first.")

        # Serialize and inject
        payload = serialize_game_for_inject(game)
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{FLASK_URL}/api/inject-state",
                json=payload,
            )
            resp.raise_for_status()

        # Wait for frontend to receive SocketIO event and re-render
        await self._page.wait_for_timeout(settle_ms)

        # Screenshot the board container
        board = self._page.locator(".board-container")
        await board.wait_for(state="visible", timeout=5000)

        raw_png = await board.screenshot(type="png")
        img = Image.open(io.BytesIO(raw_png))

        # Crop to center on the hex grid, trimming ocean border
        if self.crop_pct > 0:
            w, h = img.size
            cx = self.crop_pct * w
            cy = self.crop_pct * h
            vx = self.horizontal_offset_pct * w
            vy = self.vertical_offset_pct * h
            box = (int(cx + vx), int(cy + vy), int(w - cx + vx), int(h - cy + vy))
            img = img.crop(box)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    async def screenshot_to_file(self, game: GameEngine, path: str, **kwargs) -> str:
        """Screenshot and save to file. Returns path."""
        png = await self.screenshot(game, **kwargs)
        Path(path).write_bytes(png)
        return path

    # --- Private helpers ---

    def _start_flask(self):
        if _port_in_use(self.flask_port):
            print(f"[screenshot] Flask already running on :{self.flask_port}")
            return

        print(f"[screenshot] Starting Flask on :{self.flask_port}...")
        self._flask_proc = subprocess.Popen(
            ["python", "-m", "playground.game_viewer.app"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_port(self.flask_port, "Flask", timeout=15)

    def _start_vite(self):
        if _port_in_use(self.vite_port):
            print(f"[screenshot] Vite already running on :{self.vite_port}")
            return

        print(f"[screenshot] Starting Vite dev server on :{self.vite_port}...")
        self._vite_proc = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_port(self.vite_port, "Vite", timeout=30)

    async def _start_browser(self):
        print(f"[screenshot] Launching Playwright ({'headless' if self.headless else 'headed'})...")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)

        self._page = await self._browser.new_page(
            viewport={"width": self.board_width + 400, "height": self.board_height + 100}
        )
        await self._page.goto(VITE_URL)

        # Wait for the app to mount and socket to connect
        await self._page.wait_for_selector(".app", timeout=10000)
        game_viewer_button = self._page.get_by_role("button", name="GameEngine Viewer")
        if await game_viewer_button.count():
            await game_viewer_button.first.click()
            await self._page.wait_for_selector(".board-container", timeout=10000)
        # Give SocketIO a moment to establish connection
        await self._page.wait_for_timeout(1000)
        print("[screenshot] Browser ready.")

    def _wait_for_port(self, port: int, name: str, timeout: int = 15):
        """Poll until a port starts responding."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _port_in_use(port):
                print(f"[screenshot] {name} ready on :{port}")
                return
            time.sleep(0.5)
        raise TimeoutError(f"{name} did not start on :{port} within {timeout}s")
