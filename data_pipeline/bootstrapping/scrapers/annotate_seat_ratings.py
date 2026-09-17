#!/usr/bin/env python3
"""Annotate every seat in captured replays with its Classic4P leaderboard rating.

Replay payloads and the history endpoint carry no ratings, and the profile
endpoint does not exist; the only public source is the Classic4P leaderboard,
which lists every rated player (30k+, max 100 per page, `search` is ignored).
This fetches one daily snapshot, then joins seats by username. Ratings are the
player's rating on the snapshot date, not at game time. The manifest is
acquisition provenance only; usernames never reach model-facing data.
"""

import argparse
import asyncio
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from data_pipeline.bootstrapping.scrapers.colonist_api import ColonistAPI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "indexes"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "manifests" / "colonist" / "seat_ratings.json"
REPLAY_DIRS = (
    PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "replays",
    PROJECT_ROOT / "artifacts" / "staging" / "colonist" / "replays",
)
PAGE_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def snapshot_path(day: date) -> Path:
    return INDEX_DIR / f"classic4p_leaderboard_{day.isoformat()}.json"


async def fetch_page(api: ColonistAPI, start: int, attempts: int = 4) -> Any:
    """One leaderboard page; transient transport errors are retried with backoff, 429 is not."""
    for attempt in range(1, attempts + 1):
        try:
            response = await api.client.get(
                "/api/leaderboards/Classic4P/",
                params={"start": start, "end": start + PAGE_SIZE - 1, "search": ""},
                timeout=60.0,
            )
            if response.status_code == 429:
                raise RuntimeError(f"Leaderboard rate-limited at start={start}")
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if attempt == attempts:
                raise
            wait = 2.0 * attempt
            logger.warning("Page start=%d failed (%s); retry %d/%d in %.0fs", start, exc, attempt, attempts - 1, wait)
            await asyncio.sleep(wait)


async def fetch_leaderboard(pause_seconds: float) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    async with ColonistAPI() as api:
        start = 1
        while True:
            page = await fetch_page(api, start)
            if not isinstance(page, list) or not page:
                break
            entries.extend(page)
            if len(entries) % 5000 == 0:
                logger.info("Leaderboard: %d entries so far (rating %s)", len(entries), page[-1].get("skillRating"))
            start += PAGE_SIZE
            await asyncio.sleep(pause_seconds)
    return entries


def load_or_fetch_snapshot(pause_seconds: float, refresh: bool) -> tuple[Path, list[dict[str, Any]]]:
    path = snapshot_path(date.today())
    if path.exists() and not refresh:
        return path, json.loads(path.read_text(encoding="utf-8"))
    started = time.time()
    entries = asyncio.run(fetch_leaderboard(pause_seconds))
    path.write_text(json.dumps(entries), encoding="utf-8")
    logger.info("Saved %d leaderboard entries to %s in %.0fs", len(entries), path, time.time() - started)
    return path, entries


def annotate(entries: list[dict[str, Any]], snapshot: Path) -> dict[str, Any]:
    by_name = {entry["username"].lower(): entry for entry in entries if entry.get("username")}
    games: dict[str, Any] = {}
    human_seats = matched = 0
    for directory in REPLAY_DIRS:
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            top = payload.get("data", payload)
            if not isinstance(top, dict) or "playerUserStates" not in top:
                continue
            settings = top.get("gameSettings") or {}
            seats = []
            for user in top["playerUserStates"]:
                is_bot = bool(user.get("isBot"))
                entry = None if is_bot else by_name.get(str(user.get("username", "")).lower())
                if not is_bot:
                    human_seats += 1
                    matched += entry is not None
                seats.append({
                    "color": user.get("selectedColor"),
                    "userId": user.get("userId"),
                    "username": user.get("username"),
                    "isBot": is_bot,
                    "rank": entry.get("rank") if entry else None,
                    "skillRating": entry.get("skillRating") if entry else None,
                    "division": entry.get("division") if entry else None,
                    "totalGamesPlayed": entry.get("totalGamesPlayed") if entry else None,
                })
            games[path.stem] = {
                "source": directory.name if directory.name != "replays" else directory.parent.parent.name,
                "eloType": settings.get("eloType"),
                "gameType": settings.get("gameType"),
                "seats": seats,
            }
    return {
        "leaderboard_snapshot": snapshot.name,
        "leaderboard_size": len(entries),
        "games": games,
        "coverage": {"human_seats": human_seats, "rated_seats": matched},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--refresh", action="store_true", help="Refetch today's snapshot even if present")
    args = parser.parse_args()
    snapshot, entries = load_or_fetch_snapshot(args.pause_seconds, args.refresh)
    manifest = annotate(entries, snapshot)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    cov = manifest["coverage"]
    ratings = sorted(s["skillRating"] for g in manifest["games"].values() for s in g["seats"] if s["skillRating"])
    mid = ratings[len(ratings) // 2] if ratings else None
    logger.info(
        "Annotated %d games: %d/%d human seats rated (%.0f%%); rating min/median/max = %s/%s/%s -> %s",
        len(manifest["games"]), cov["rated_seats"], cov["human_seats"],
        100 * cov["rated_seats"] / max(cov["human_seats"], 1),
        ratings[0] if ratings else None, mid, ratings[-1] if ratings else None, MANIFEST_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
