"""The per-cursor reasoning job record and its contract error type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict

from evals.json_types import JsonDict, as_number


@dataclass(frozen=True)
class ReasoningJob:
    """One transcript interval and its immutable public board inspection tools."""

    job_id: str
    game_id: str
    replay_index: int
    input_hash: str
    window_start_s: float
    window_end_s: float
    utterances: Tuple[JsonDict, ...]
    board_snapshot: JsonDict
    locations: Dict[str, JsonDict]

    def _evidence_numbers(self, field: str) -> list[int | float]:
        return [as_number(item[field], f"utterance {field}") for item in self.utterances]

    def plan_record(self) -> JsonDict:
        return {
            "job_id": self.job_id,
            "game_id": self.game_id,
            "replay_index": self.replay_index,
            "input_hash": self.input_hash,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "source_start_s": min(self._evidence_numbers("start_s")),
            "source_end_s": max(self._evidence_numbers("end_s")),
            "source_start_index": min(self._evidence_numbers("source_start_index")),
            "source_end_index": max(self._evidence_numbers("source_end_index")),
            "utterance_count": len(self.utterances),
            "board_state_hash": self.board_snapshot["board_state_hash"],
        }


class NarratorReasoningError(ValueError):
    """Raised when generation inputs or artifacts violate their contract."""


class ReasoningScan(TypedDict):
    """One replay pass: its provenance hashes plus every nonempty-interval job."""

    game_id: str
    total_events: int
    replay_sha256: str
    transcript_sha256: str
    narrator: JsonDict
    jobs: List[ReasoningJob]
