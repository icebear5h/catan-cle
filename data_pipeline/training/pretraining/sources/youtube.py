"""
YouTube transcript fetcher for Catan strategy content.

Wraps the existing YouTubeScraper with curated channel lists and batch
processing to build a pretraining corpus from top Catan creators.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

from data_pipeline.ingestion.youtube_scraper import YouTubeScraper

# Curated Catan strategy channels (channel_id, display_name)
# These are the best sources of Catan strategic reasoning on YouTube.
CURATED_CHANNELS = [
    ("UCxB_gPJz-oAtYzVVIjHPNMg", "DyLighted"),
    ("UCEdvzk_UOOdmpIMOsz_RNOA", "jorbs"),
    ("UCYqYMBp6jYqEco2kCOk3uQg", "The Game Haus"),
    ("UCX6b17PVsYBQ0ip5gyeme-Q", "Catan"),  # Official Catan channel
    ("UCi3mNYrJq5nMJD7MOo1Iiew", "No Rolls Barred"),
    ("UCWL2DysMGBGPfcB0cK86zHQ", "Actualol"),
]

# Search queries to find additional Catan content
SEARCH_QUERIES = [
    "catan strategy guide",
    "catan tips advanced",
    "settlers of catan tutorial",
    "catan placement strategy",
    "catan trading strategy",
    "catan road building strategy",
    "catan development cards strategy",
    "catan resource management",
    "catan tournament gameplay",
    "catan opening placement",
]

# Cache directory for raw transcripts (avoid re-fetching)
CACHE_DIR = Path(__file__).parent.parent / "cache" / "youtube"


def _clean_transcript_text(text: str) -> str:
    """Clean up YouTube auto-generated transcript artifacts."""
    # Remove [Music], [Applause], etc.
    text = re.sub(r"\[.*?\]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_catan_relevant(text: str) -> bool:
    """Quick relevance check - does the text mention Catan concepts?"""
    keywords = [
        "catan", "settler", "settlement", "city", "road", "resource",
        "brick", "lumber", "wool", "grain", "ore", "wheat", "sheep",
        "wood", "robber", "knight", "development card", "longest road",
        "largest army", "victory point", "harbor", "port", "trade",
        "hex", "dice", "roll", "pip", "placement", "board game",
    ]
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower) >= 3


class YouTubeCorpusBuilder:
    """Builds a text corpus from YouTube Catan strategy content."""

    def __init__(self, api_key: Optional[str] = None, cache: bool = True):
        self.scraper = YouTubeScraper(api_key=api_key)
        self.cache = cache
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, video_id: str) -> Path:
        return CACHE_DIR / f"{video_id}.json"

    def _load_cached(self, video_id: str) -> Optional[Dict]:
        path = self._cache_path(video_id)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, video_id: str, data: Dict):
        if not self.cache:
            return
        with open(self._cache_path(video_id), "w") as f:
            json.dump(data, f)

    def get_channel_video_ids(self, channel_id: str, max_results: int = 50) -> List[str]:
        """Get video IDs from a channel using YouTube Data API."""
        if not self.scraper.youtube:
            print(f"  [WARN] No API key, skipping channel {channel_id}")
            return []

        try:
            # Get uploads playlist
            channel_resp = self.scraper.youtube.channels().list(
                part="contentDetails",
                id=channel_id,
            ).execute()

            if not channel_resp.get("items"):
                print(f"  [WARN] Channel {channel_id} not found")
                return []

            uploads_id = channel_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

            # Paginate through uploads
            video_ids = []
            next_page = None
            while len(video_ids) < max_results:
                pl_resp = self.scraper.youtube.playlistItems().list(
                    part="contentDetails",
                    playlistId=uploads_id,
                    maxResults=min(50, max_results - len(video_ids)),
                    pageToken=next_page,
                ).execute()

                for item in pl_resp["items"]:
                    video_ids.append(item["contentDetails"]["videoId"])

                next_page = pl_resp.get("nextPageToken")
                if not next_page:
                    break

            return video_ids

        except Exception as e:
            print(f"  [ERROR] Failed to get channel videos: {e}")
            return []

    def fetch_video_transcript(self, video_id: str) -> Optional[str]:
        """Fetch and clean a single video's transcript. Returns full text or None."""
        # Check cache first
        cached = self._load_cached(video_id)
        if cached:
            return cached.get("text")

        transcript_segments = self.scraper.get_transcript(video_id)
        if not transcript_segments:
            return None

        full_text = " ".join(seg["text"] for seg in transcript_segments)
        full_text = _clean_transcript_text(full_text)

        if not full_text or len(full_text) < 100:
            return None

        # Cache raw result
        self._save_cache(video_id, {
            "video_id": video_id,
            "text": full_text,
            "char_count": len(full_text),
        })

        return full_text

    def fetch_from_channels(
        self,
        channels: Optional[List[tuple]] = None,
        max_per_channel: int = 50,
        rate_limit: float = 0.5,
    ) -> List[Dict[str, str]]:
        """Fetch transcripts from curated channels.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        channels = channels or CURATED_CHANNELS
        results = []

        for channel_id, channel_name in channels:
            print(f"[YouTube] Fetching from channel: {channel_name}")
            video_ids = self.get_channel_video_ids(channel_id, max_results=max_per_channel)
            print(f"  Found {len(video_ids)} videos")

            for vid_id in video_ids:
                text = self.fetch_video_transcript(vid_id)
                if text and _is_catan_relevant(text):
                    results.append({
                        "source": f"youtube/{channel_name}/{vid_id}",
                        "text": text,
                    })
                    print(f"  + {vid_id}: {len(text)} chars (relevant)")
                else:
                    if text:
                        print(f"  - {vid_id}: skipped (not Catan-relevant)")
                    else:
                        print(f"  - {vid_id}: no transcript")

                time.sleep(rate_limit)

        print(f"[YouTube] Collected {len(results)} relevant transcripts from channels")
        return results

    def fetch_from_search(
        self,
        queries: Optional[List[str]] = None,
        max_per_query: int = 20,
        rate_limit: float = 0.5,
    ) -> List[Dict[str, str]]:
        """Search YouTube for Catan content and fetch transcripts.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        queries = queries or SEARCH_QUERIES
        seen_ids = set()
        results = []

        for query in queries:
            print(f"[YouTube] Searching: '{query}'")
            urls = self.scraper.search_catan_videos(query, max_results=max_per_query)

            for url in urls:
                vid_id = self.scraper.extract_video_id(url)
                if vid_id in seen_ids:
                    continue
                seen_ids.add(vid_id)

                text = self.fetch_video_transcript(vid_id)
                if text and _is_catan_relevant(text):
                    results.append({
                        "source": f"youtube/search/{vid_id}",
                        "text": text,
                    })
                time.sleep(rate_limit)

        print(f"[YouTube] Collected {len(results)} relevant transcripts from search")
        return results

    def build(
        self,
        include_channels: bool = True,
        include_search: bool = True,
        max_per_channel: int = 50,
        max_per_query: int = 20,
    ) -> List[Dict[str, str]]:
        """Run the full YouTube corpus pipeline.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        results = []

        if include_channels:
            results.extend(self.fetch_from_channels(max_per_channel=max_per_channel))

        if include_search:
            results.extend(self.fetch_from_search(max_per_query=max_per_query))

        # Deduplicate by text content (some videos may appear in both)
        seen_texts = set()
        deduped = []
        for item in results:
            text_hash = hash(item["text"][:500])
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                deduped.append(item)

        print(f"[YouTube] Total unique documents: {len(deduped)}")
        return deduped
