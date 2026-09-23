"""Scrape Catan forum threads from BoardGameGeek using their XML API."""

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from data_pipeline.training.pretraining.sources.forums._config import (
    BGG_API_BASE,
    BGG_GAME_ID,
    CACHE_DIR,
)
from data_pipeline.training.pretraining.sources.forums._reddit import Document


class BGGScraper:
    """Scrape Catan forum threads from BoardGameGeek using their XML API."""

    def __init__(self, cache: bool = True) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CatanLearning/1.0 (research)",
        })
        self.cache = cache
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        safe_key = re.sub(r"[^\w\-]", "_", key)
        return CACHE_DIR / f"bgg_{safe_key}.json"

    def _load_cached(self, key: str) -> list[Document] | None:
        path = self._cache_path(key)
        if path.exists():
            with open(path) as f:
                cached: list[Document] = json.load(f)
                return cached
        return None

    def _save_cache(self, key: str, data: list[Document]) -> None:
        if not self.cache:
            return
        with open(self._cache_path(key), "w") as f:
            json.dump(data, f)

    def _resolve_forum_id(self, game_id: int, forum_type: str) -> tuple[bool, str | None]:
        """Return whether a forum element was found, and its id attribute.

        The flag and the id are separate because the original scraper stopped
        only when the game exposed no forum at all; a forum element that merely
        lacks an `id` attribute still went on to be paged with an empty id.
        """
        forumlist_params: dict[str, str | int] = {"id": game_id, "type": "thing"}
        try:
            resp = self.session.get(
                f"{BGG_API_BASE}/forumlist",
                params=forumlist_params,
                timeout=30,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception as e:
            print(f"  [BGG] Error fetching forum list: {e}")
            return False, None

        # Find the strategy/general forum
        forum_id: str | None = None
        for forum_el in root.findall(".//forum"):
            title = forum_el.get("title", "").lower()
            if forum_type.lower() in title or "strategy" in title or "general" in title:
                forum_id = forum_el.get("id")
                print(f"  [BGG] Found forum: {forum_el.get('title')} (id={forum_id})")
                break

        if not forum_id:
            # Fallback: use first forum
            first = root.find(".//forum")
            if first is None:
                print("  [BGG] No forums found")
                return False, None
            forum_id = first.get("id")
            print(f"  [BGG] Using fallback forum: {first.get('title')} (id={forum_id})")
        return True, forum_id

    def _collect_thread_ids(self, forum_id: str | None, max_threads: int) -> list[str | None]:
        """Page through a forum and collect its thread ids."""
        thread_ids: list[str | None] = []
        page = 1
        while len(thread_ids) < max_threads:
            try:
                resp = self.session.get(
                    f"{BGG_API_BASE}/forum",
                    params={"id": forum_id, "page": page},
                    timeout=30,
                )
                resp.raise_for_status()
                forum_root = ET.fromstring(resp.text)
            except Exception as e:
                print(f"  [BGG] Error fetching forum page {page}: {e}")
                break

            threads = forum_root.findall(".//thread")
            if not threads:
                break

            for t in threads:
                thread_ids.append(t.get("id"))

            page += 1
            time.sleep(1)

        return thread_ids[:max_threads]

    def _fetch_thread(self, tid: str | None) -> Document | None:
        """Fetch one thread and flatten it into a corpus document."""
        try:
            resp = self.session.get(
                f"{BGG_API_BASE}/thread",
                params={"id": tid},
                timeout=30,
            )
            resp.raise_for_status()
            thread_root = ET.fromstring(resp.text)
        except Exception as e:
            print(f"  [BGG] Error fetching thread {tid}: {e}")
            time.sleep(2)
            return None

        subject = thread_root.get("subject", "")
        articles = thread_root.findall(".//article")

        if not articles:
            time.sleep(1)
            return None

        # Combine all posts in the thread
        thread_text_parts = [subject]
        for article in articles:
            body_el = article.find("body")
            if body_el is not None and body_el.text:
                # Clean HTML-ish BGG formatting
                body = body_el.text
                body = re.sub(r"\[/?[a-zA-Z]+[^\]]*\]", "", body)  # BBCode tags
                body = re.sub(r"\s+", " ", body).strip()
                if len(body) > 20:
                    thread_text_parts.append(body)

        full_text = "\n\n".join(thread_text_parts)
        time.sleep(1)  # BGG rate limits
        if len(full_text) > 100:
            return {"source": f"bgg/thread/{tid}", "text": full_text}
        return None

    def fetch_forum_threads(
        self,
        game_id: int = BGG_GAME_ID,
        forum_type: str = "strategy",
        max_threads: int = 200,
    ) -> list[Document]:
        """Fetch threads from BGG forums for a game.

        The BGG XML API exposes forums via /forumlist and /forum endpoints.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        cache_key = f"threads_{game_id}_{forum_type}_{max_threads}"
        cached = self._load_cached(cache_key)
        if cached:
            print(f"  [BGG] Using cached threads ({len(cached)} threads)")
            return cached

        found, forum_id = self._resolve_forum_id(game_id, forum_type)
        if not found:
            return []

        thread_ids = self._collect_thread_ids(forum_id, max_threads)
        print(f"  [BGG] Found {len(thread_ids)} thread IDs")

        results: list[Document] = []
        for tid in thread_ids:
            document = self._fetch_thread(tid)
            if document is not None:
                results.append(document)

        self._save_cache(cache_key, results)
        print(f"  [BGG] Fetched {len(results)} threads with content")
        return results

    def build(self, max_threads: int = 200) -> list[Document]:
        """Run the full BGG scraping pipeline."""
        return self.fetch_forum_threads(max_threads=max_threads)


__all__ = ["BGGScraper"]
