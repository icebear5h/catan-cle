"""
Reddit and BoardGameGeek forum scraper for Catan discussion content.

Pulls top posts and comments from r/catan and the BGG Catan forum
to build pretraining corpus with community strategy discussion.
"""

from data_pipeline.training.pretraining.sources.forums._bgg import BGGScraper
from data_pipeline.training.pretraining.sources.forums._config import (
    BGG_API_BASE,
    BGG_GAME_ID,
    CACHE_DIR,
    PROJECT_ROOT,
    REDDIT_SEARCH_QUERIES,
    REDDIT_SUBREDDITS,
)
from data_pipeline.training.pretraining.sources.forums._reddit import (
    Document,
    RedditScraper,
)


class ForumCorpusBuilder:
    """Combines Reddit and BGG forum content into a unified corpus."""

    def __init__(self, cache: bool = True) -> None:
        self.reddit = RedditScraper(cache=cache)
        self.bgg = BGGScraper(cache=cache)

    def build(
        self,
        include_reddit: bool = True,
        include_bgg: bool = True,
        max_reddit_per_sub: int = 200,
        max_bgg_threads: int = 200,
    ) -> list[Document]:
        """Build forum corpus from all sources.

        Returns list of {"source": ..., "text": ...} dicts.
        """
        results: list[Document] = []

        if include_reddit:
            print("[Forums] Fetching Reddit posts...")
            results.extend(self.reddit.build(max_per_subreddit=max_reddit_per_sub))

        if include_bgg:
            print("[Forums] Fetching BGG threads...")
            results.extend(self.bgg.build(max_threads=max_bgg_threads))

        print(f"[Forums] Total forum documents: {len(results)}")
        return results


__all__ = [
    "BGG_API_BASE",
    "BGG_GAME_ID",
    "CACHE_DIR",
    "PROJECT_ROOT",
    "REDDIT_SEARCH_QUERIES",
    "REDDIT_SUBREDDITS",
    "BGGScraper",
    "Document",
    "ForumCorpusBuilder",
    "RedditScraper",
]
