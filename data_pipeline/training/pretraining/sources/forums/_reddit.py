"""Scrape Catan-related posts and comments from Reddit's public JSON API."""

import json
import re
import time
from pathlib import Path
from typing import TypedDict

import requests

from data_pipeline.training.pretraining.sources.forums._config import (
    CACHE_DIR,
    REDDIT_SUBREDDITS,
)

Document = dict[str, str]


class RedditPostFields(TypedDict, total=False):
    """The subset of a Reddit link payload that the corpus builder reads."""

    id: str
    title: str
    selftext: str
    score: int
    is_video: bool


class RedditPost(TypedDict, total=False):
    """One `t3` child of a Reddit listing."""

    kind: str
    data: RedditPostFields


class RedditListing(TypedDict, total=False):
    """The `data` envelope of a Reddit listing response."""

    children: list[RedditPost]
    after: str


class RedditListingResponse(TypedDict, total=False):
    """A Reddit listing response body."""

    data: RedditListing


class RedditCommentFields(TypedDict, total=False):
    """The subset of a Reddit comment payload that the corpus builder reads."""

    body: str
    score: int
    replies: "RedditCommentResponse | str"


class RedditComment(TypedDict, total=False):
    """One `t1` child of a Reddit comment listing."""

    kind: str
    data: RedditCommentFields


class RedditCommentListing(TypedDict, total=False):
    """The `data` envelope of a Reddit comment listing."""

    children: list[RedditComment]


class RedditCommentResponse(TypedDict, total=False):
    """A Reddit comment listing response body."""

    data: RedditCommentListing


class RedditScraper:
    """Scrape Catan-related posts from Reddit using the public JSON API."""

    def __init__(self, cache: bool = True) -> None:
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

    def fetch_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 100,
    ) -> list[Document]:
        """Fetch top posts from a subreddit.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        cache_key = f"{subreddit}_{sort}_{time_filter}_{limit}"
        cached = self._load_cached(cache_key)
        if cached:
            print(f"  [Reddit] Using cached {subreddit}/{sort} ({len(cached)} posts)")
            return cached

        results: list[Document] = []
        after: str | None = None

        while len(results) < limit:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params: dict[str, str | int] = {
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
                data: RedditListingResponse = resp.json()
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
            data: list[RedditCommentResponse] = resp.json()
        except Exception as e:
            print(f"  [Reddit] Error fetching comments: {e}")
            return ""

        comments_data = data[1]["data"]["children"] if len(data) > 1 else []
        comment_texts: list[str] = []

        def _extract_comments(children: list[RedditComment], depth: int = 0) -> None:
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

    def build(self, max_per_subreddit: int = 200) -> list[Document]:
        """Fetch posts from all configured subreddits.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        results: list[Document] = []
        for subreddit in REDDIT_SUBREDDITS:
            posts = self.fetch_subreddit_posts(
                subreddit, sort="top", time_filter="all", limit=max_per_subreddit,
            )
            results.extend(posts)

        print(f"[Reddit] Total posts collected: {len(results)}")
        return results


__all__ = ["Document", "RedditScraper"]
