"""Which replays have a curated transcript beside them, and where it lives."""

from pathlib import Path
from typing import TypedDict

__all__ = [
    "PROJECT_ROOT",
    "TRANSCRIPT_ALIGNMENT_VERSION",
    "TRANSCRIPT_SCHEMA",
    "CuratedPair",
    "get_curated_replay_path",
]

# One level deeper than the old module, so the repo root is four parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRANSCRIPT_SCHEMA = "paired-replay-transcript-v1"
TRANSCRIPT_ALIGNMENT_VERSION = "caption-end-availability-v2"


class CuratedPair(TypedDict):
    """The three files one curated replay/transcript pairing is assembled from."""

    replay_path: Path
    transcript_path: Path
    manifest_path: Path


_CURATED_PAIRS: dict[str, CuratedPair] = {
    "242781000": {
        "replay_path": PROJECT_ROOT
        / "data_pipeline/bootstrapping/data/replay_staging/242781000.json",
        "transcript_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/transcript.json",
        "manifest_path": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/pairing_manifest.json",
    }
}


def get_curated_replay_path(game_id: str) -> Path | None:
    """Return the staged replay path for a curated transcript pair, if any."""
    pair = _CURATED_PAIRS.get(str(game_id))
    return pair["replay_path"] if pair else None


def curated_pair(game_id: str) -> CuratedPair | None:
    """Return every path for one curated pairing, or None when it is not curated."""
    return _CURATED_PAIRS.get(str(game_id))
