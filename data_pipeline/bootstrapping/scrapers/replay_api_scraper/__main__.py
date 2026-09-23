"""Run the direct replay API scraper through its established module name."""

import asyncio

from data_pipeline.bootstrapping.scrapers.replay_api_scraper import main

asyncio.run(main())
