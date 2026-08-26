"""
Reddit and BoardGameGeek forum scraper for Catan discussion content.

Pulls top posts and comments from r/catan and the BGG Catan forum
to build pretraining corpus with community strategy discussion.
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional

import requests

# Cache directory
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache" / "pretraining" / "forums"

# Reddit config
REDDIT_SUBREDDITS = ["catan", "Catan", "boardgames"]
REDDIT_SEARCH_QUERIES = [
    "strategy", "tips", "placement", "trading", "longest road",
    "development cards", "robber", "resource", "city", "settlement",
]

# BGG Catan forum IDs
# Main Catan game ID on BGG is 13 (the original Settlers of Catan)
BGG_GAME_ID = 13
BGG_API_BASE = "https://boardgamegeek.com/xmlapi2"


class RedditScraper:
    """Scrape Catan-related posts from Reddit using the public JSON API."""

    def __init__(self, cache: bool = True):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CatanLearning/1.0 (research; contact: github.com/catan-learning)",
        })
        self.cache = cache
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        safe_key = re.sub(r"[^\w\-]", "_", key)
        return CACHE_DIR / f"reddit_{safe_key}.json"

    def _load_cached(self, key: str) -> Optional[List]:
        path = self._cache_path(key)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: List):
        if not self.cache:
            return
        with open(self._cache_path(key), "w") as f:
            json.dump(data, f)

    def fetch_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 100,
    ) -> List[Dict[str, str]]:
        """Fetch top posts from a subreddit.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        cache_key = f"{subreddit}_{sort}_{time_filter}_{limit}"
        cached = self._load_cached(cache_key)
        if cached:
            print(f"  [Reddit] Using cached {subreddit}/{sort} ({len(cached)} posts)")
            return cached

        results = []
        after = None

        while len(results) < limit:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params = {
                "t": time_filter,
                "limit": min(100, limit - len(results)),
            }
            if after:
                params["after"] = after

            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    print("  [Reddit] Rate limited, waiting 60s...")
                    time.sleep(60)
                    continue
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [Reddit] Error fetching r/{subreddit}: {e}")
                break

            posts = data.get("data", {}).get("children", [])
            if not posts:
                break

            for post in posts:
                pd = post["data"]
                # Skip non-text posts
                if pd.get("is_video") or not pd.get("selftext"):
                    continue

                title = pd.get("title", "")
                body = pd.get("selftext", "")
                score = pd.get("score", 0)

                # Only keep posts with substance
                if len(body) < 50 or score < 5:
                    continue

                # Combine title and body
                text = f"{title}\n\n{body}"
                # Clean markdown artifacts
                text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # [text](url) -> text
                text = re.sub(r"#{1,6}\s*", "", text)  # Remove heading markers
                text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)  # Bold/italic
                text = re.sub(r"\s+", " ", text).strip()

                results.append({
                    "source": f"reddit/r/{subreddit}/{pd.get('id', 'unknown')}",
                    "text": text,
                })

            after = data.get("data", {}).get("after")
            if not after:
                break

            time.sleep(2)  # Be polite to Reddit

        self._save_cache(cache_key, results)
        print(f"  [Reddit] Fetched {len(results)} posts from r/{subreddit}")
        return results

    def fetch_post_comments(self, subreddit: str, post_id: str) -> str:
        """Fetch top comments from a specific post. Returns concatenated text."""
        try:
            url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [Reddit] Error fetching comments: {e}")
            return ""

        comments_data = data[1]["data"]["children"] if len(data) > 1 else []
        comment_texts = []

        def _extract_comments(children, depth=0):
            if depth > 3:  # Don't go too deep
                return
            for child in children:
                if child["kind"] != "t1":
                    continue
                cd = child["data"]
                body = cd.get("body", "")
                score = cd.get("score", 0)
                if body and score >= 3 and len(body) > 20:
                    comment_texts.append(body)
                replies = cd.get("replies")
                if isinstance(replies, dict):
                    _extract_comments(
                        replies.get("data", {}).get("children", []),
                        depth + 1,
                    )

        _extract_comments(comments_data)
        return "\n\n".join(comment_texts)

    def build(self, max_per_subreddit: int = 200) -> List[Dict[str, str]]:
        """Fetch posts from all configured subreddits.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        results = []
        for subreddit in REDDIT_SUBREDDITS:
            posts = self.fetch_subreddit_posts(
                subreddit, sort="top", time_filter="all", limit=max_per_subreddit,
            )
            results.extend(posts)

        print(f"[Reddit] Total posts collected: {len(results)}")
        return results


class BGGScraper:
    """Scrape Catan forum threads from BoardGameGeek using their XML API."""

    def __init__(self, cache: bool = True):
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

    def _load_cached(self, key: str) -> Optional[List]:
        path = self._cache_path(key)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: List):
        if not self.cache:
            return
        with open(self._cache_path(key), "w") as f:
            json.dump(data, f)

    def fetch_forum_threads(
        self,
        game_id: int = BGG_GAME_ID,
        forum_type: str = "strategy",
        max_threads: int = 200,
    ) -> List[Dict[str, str]]:
        """Fetch threads from BGG forums for a game.

        The BGG XML API exposes forums via /forumlist and /forum endpoints.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        cache_key = f"threads_{game_id}_{forum_type}_{max_threads}"
        cached = self._load_cached(cache_key)
        if cached:
            print(f"  [BGG] Using cached threads ({len(cached)} threads)")
            return cached

        # Step 1: Get forum list for the game
        try:
            resp = self.session.get(
                f"{BGG_API_BASE}/forumlist",
                params={"id": game_id, "type": "thing"},
                timeout=30,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
        except Exception as e:
            print(f"  [BGG] Error fetching forum list: {e}")
            return []

        # Find the strategy/general forum
        forum_id = None
        for forum_el in root.findall(".//forum"):
            title = forum_el.get("title", "").lower()
            if forum_type.lower() in title or "strategy" in title or "general" in title:
                forum_id = forum_el.get("id")
                print(f"  [BGG] Found forum: {forum_el.get('title')} (id={forum_id})")
                break

        if not forum_id:
            # Fallback: use first forum
            first = root.find(".//forum")
            if first is not None:
                forum_id = first.get("id")
                print(f"  [BGG] Using fallback forum: {first.get('title')} (id={forum_id})")
            else:
                print("  [BGG] No forums found")
                return []

        # Step 2: Get thread list from forum
        thread_ids = []
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

        thread_ids = thread_ids[:max_threads]
        print(f"  [BGG] Found {len(thread_ids)} thread IDs")

        # Step 3: Fetch individual threads
        results = []
        for tid in thread_ids:
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
                continue

            subject = thread_root.get("subject", "")
            articles = thread_root.findall(".//article")

            if not articles:
                time.sleep(1)
                continue

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
            if len(full_text) > 100:
                results.append({
                    "source": f"bgg/thread/{tid}",
                    "text": full_text,
                })

            time.sleep(1)  # BGG rate limits

        self._save_cache(cache_key, results)
        print(f"  [BGG] Fetched {len(results)} threads with content")
        return results

    def build(self, max_threads: int = 200) -> List[Dict[str, str]]:
        """Run the full BGG scraping pipeline."""
        return self.fetch_forum_threads(max_threads=max_threads)


class ForumCorpusBuilder:
    """Combines Reddit and BGG forum content into a unified corpus."""

    def __init__(self, cache: bool = True):
        self.reddit = RedditScraper(cache=cache)
        self.bgg = BGGScraper(cache=cache)

    def build(
        self,
        include_reddit: bool = True,
        include_bgg: bool = True,
        max_reddit_per_sub: int = 200,
        max_bgg_threads: int = 200,
    ) -> List[Dict[str, str]]:
        """Build forum corpus from all sources.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        results = []

        if include_reddit:
            print("[Forums] Fetching Reddit posts...")
            results.extend(self.reddit.build(max_per_subreddit=max_reddit_per_sub))

        if include_bgg:
            print("[Forums] Fetching BGG threads...")
            results.extend(self.bgg.build(max_threads=max_bgg_threads))

        print(f"[Forums] Total forum documents: {len(results)}")
        return results
