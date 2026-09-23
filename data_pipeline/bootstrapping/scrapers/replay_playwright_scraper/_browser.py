"""Persistent Playwright browser session that captures replay API responses."""

import asyncio
from pathlib import Path

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    Response,
    async_playwright,
)
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._config import (
    DEFAULT_TIMEOUT_MS,
    ReplayRateLimitedError,
    logger,
)
from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper._payload import (
    is_matching_replay_response,
    is_valid_replay_payload,
    replay_page_url,
)
from data_pipeline.json_types import JsonDict, JsonValue


class PlaywrightReplayScraper:
    """Persistent-browser replay scraper."""

    def __init__(
        self,
        profile_dir: Path,
        headless: bool = False,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        cdp_url: str | None = None,
    ) -> None:
        self.profile_dir = profile_dir
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.cdp_url = cdp_url
        self._playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def __aenter__(self) -> "PlaywrightReplayScraper":
        playwright = await async_playwright().start()
        self._playwright = playwright
        if self.cdp_url:
            logger.info("Connecting to existing Chrome over CDP: %s", self.cdp_url)
            self.browser = await playwright.chromium.connect_over_cdp(self.cdp_url)
            if not self.browser.contexts:
                raise RuntimeError("Connected Chrome did not expose a browser context")
            self.context = self.browser.contexts[0]
            self.page = await self.context.new_page()
            return self

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            viewport={"width": 1440, "height": 900},
            accept_downloads=False,
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self.context and not self.cdp_url:
            await self.context.close()
        if self._playwright:
            await self._playwright.stop()

    async def capture_replay_data(self, game_id: str, player_color: int) -> JsonDict | None:
        """Open a replay page and return the first valid 200 replay data response."""
        if not self.page:
            raise RuntimeError("PlaywrightReplayScraper must be used as an async context manager")

        response_task = asyncio.create_task(
            self._wait_for_replay_response(self.page, game_id, player_color)
        )
        url = replay_page_url(game_id, player_color)
        logger.info("Opening replay %s (playerColor=%s)", game_id, player_color)

        try:
            await self.page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            return await response_task
        except ReplayRateLimitedError:
            raise
        except (asyncio.TimeoutError, PlaywrightTimeoutError):
            logger.error(
                "Timed out waiting for replay data for %s. If the browser is showing login "
                "or a challenge, complete it and rerun the same command.",
                game_id,
            )
            return None
        except Exception as exc:
            logger.error("Failed to capture replay %s: %s", game_id, exc)
            return None
        finally:
            if not response_task.done():
                response_task.cancel()

    async def _wait_for_replay_response(
        self,
        page: Page,
        game_id: str,
        player_color: int,
    ) -> JsonDict:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[JsonDict] = loop.create_future()

        async def read_successful_response(response: Response) -> None:
            try:
                data: JsonValue = await response.json()
                if not is_valid_replay_payload(data) or not isinstance(data, dict):
                    logger.warning(
                        "Ignoring 200 replay response for %s because it did not contain "
                        "eventHistory.events",
                        game_id,
                    )
                    return
                if not future.done():
                    future.set_result(data)
            except Exception as exc:
                logger.warning("Could not read replay response for %s: %s", game_id, exc)

        async def record_rate_limit(response: Response) -> None:
            # Colonist sends no Retry-After; keep every other header and the body so the
            # window can be worked out from the log instead of by probing again.
            headers = {k: v for k, v in response.headers.items() if k.lower() != "set-cookie"}
            try:
                body = (await response.text())[:500]
            except Exception as exc:
                body = f"<unreadable: {exc}>"
            logger.error("429 for %s: headers=%s body=%r", game_id, headers, body)
            if not future.done():
                future.set_exception(ReplayRateLimitedError(game_id, headers.get("retry-after")))

        def on_response(response: Response) -> None:
            if not is_matching_replay_response(response.url, game_id, player_color):
                return

            status = response.status
            logger.info("Replay API response for %s: %s", game_id, status)
            if status == 200:
                asyncio.create_task(read_successful_response(response))
            elif status in {401, 403}:
                logger.info("Waiting for a later successful replay response after %s", status)
            elif status == 429 and not future.done():
                asyncio.create_task(record_rate_limit(response))
            elif status == 404 and not future.done():
                future.set_exception(RuntimeError(f"Replay not found: {game_id}"))

        page.on("response", on_response)
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_ms / 1000)
        finally:
            page.remove_listener("response", on_response)


__all__ = ["PlaywrightReplayScraper"]
