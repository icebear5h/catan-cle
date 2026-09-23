"""Scrape one or more YouTube transcripts from the command line."""

import sys

from data_pipeline.ingestion.youtube_scraper import YouTubeScraper

if len(sys.argv) < 2:
    print("Usage: python -m data_pipeline.ingestion.youtube_scraper <youtube_url> [youtube_url2 ...]")
    print("  Scrapes transcript and saves to artifacts/generated/pretraining/legacy_corpus/transcript_{id}.md")
    sys.exit(1)

scraper = YouTubeScraper()

for url in sys.argv[1:]:
    try:
        result = scraper.scrape_video(url)
        if result:
            scraper.save_transcript(result)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
