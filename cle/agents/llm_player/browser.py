"""Playwright board capture for the VLM player's vision channel."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from cle.game_engine.game import GameEngine

if TYPE_CHECKING:
    from cle.agents.llm_player import LLMPlayer

__all__ = ["_SharedBrowser"]

_SCREENSHOT_DIR = Path(__file__).resolve().parents[3] / "playground" / "screenshots"

# Crop settings tuned for the frontend board (from VLM playground)
_CROP_PCT = 0.17
_VERTICAL_OFFSET_PCT = -0.017
_HORIZONTAL_OFFSET_PCT = 0.022


class _SharedBrowser:
    """Lazy singleton Playwright browser for frontend screenshots."""

    _instance: _SharedBrowser | None = None

    def __init__(self) -> None:
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._ready = False

    @classmethod
    def get(cls) -> '_SharedBrowser':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_started(self, vite_url: str = "http://localhost:5173") -> None:
        """Start browser if not already running."""
        if self._ready:
            return

        print("[screenshot] Starting Playwright (headless, sync)...")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page(
            viewport={"width": 1200, "height": 900}
        )
        self._page.goto(vite_url)
        self._page.wait_for_selector(".app", timeout=10000)
        # Let SocketIO connect and receive initial state
        self._page.wait_for_timeout(1500)
        self._ready = True
        print("[screenshot] Browser ready.")

    def screenshot_board(self, settle_ms: int = 400) -> bytes:
        """Screenshot the .board-container and crop to the hex grid."""
        if not self._ready:
            raise RuntimeError("Browser not started")
        assert self._page is not None

        self._page.wait_for_timeout(settle_ms)

        board = self._page.locator(".board-container")
        board.wait_for(state="visible", timeout=5000)
        raw_png = board.screenshot(type="png")

        # Crop to center on hex grid (same settings as VLM playground)
        img: Image.Image = Image.open(io.BytesIO(raw_png))
        w, h = img.size
        cx = _CROP_PCT * w
        cy = _CROP_PCT * h
        vx = _HORIZONTAL_OFFSET_PCT * w
        vy = _VERTICAL_OFFSET_PCT * h
        box = (int(cx + vx), int(cy + vy), int(w - cx + vx), int(h - cy + vy))
        img = img.crop(box)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def stop(self) -> None:
        if self._page:
            self._page.close()
            self._page = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        self._ready = False


def _capture_board(self: LLMPlayer, game: GameEngine) -> bytes | None:
    """Capture the frontend board via Playwright and save to disk."""
    try:
        browser = _SharedBrowser.get()
        browser.ensure_started()
        image_bytes = browser.screenshot_board()
    except Exception as e:
        print(f"[{self.color}] Screenshot failed: {e}")
        return None

    self.last_screenshot = image_bytes

    # Save to disk for verification
    _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = _SCREENSHOT_DIR / "latest.png"
    latest_path.write_bytes(image_bytes)
    turn_path = _SCREENSHOT_DIR / f"turn_{game.state.num_turns}_{self.color.value}.png"
    turn_path.write_bytes(image_bytes)
    print(f"Screenshot saved: {latest_path} ({len(image_bytes)} bytes)")

    return image_bytes
