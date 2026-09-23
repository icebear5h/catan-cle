"""
YouTube Transcript Scraper

Extracts transcripts and metadata from Catan strategy videos.
"""

from data_pipeline.ingestion.youtube_scraper._curated import (
    CURATED_CHANNELS,
    CURATED_VIDEOS,
)
from data_pipeline.ingestion.youtube_scraper._scraper import YouTubeScraper
from data_pipeline.ingestion.youtube_scraper._types import TranscriptSegment

__all__ = [
    "CURATED_CHANNELS",
    "CURATED_VIDEOS",
    "TranscriptSegment",
    "YouTubeScraper",
]
