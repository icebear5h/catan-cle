"""Run the Playwright replay scraper through its established module name."""

import asyncio

from data_pipeline.bootstrapping.scrapers.replay_playwright_scraper import main

raise SystemExit(asyncio.run(main()))
