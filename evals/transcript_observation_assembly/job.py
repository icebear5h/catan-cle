"""The per-packet assembly job record and its contract error type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict

from evals.json_types import JsonDict, as_number


class ObservationAssemblyError(ValueError):
    """Raised when observation assembly violates its causal artifact contract."""


@dataclass(frozen=True)
class ObservationAssemblyJob:
    """One causal packet assembled at a narrator decision or public observation."""

    job_id: str
    game_id: str
    replay_index: int
    previous_replay_index: int
    anchor_kind: str
    decision_ids: Tuple[str, ...]
    input_hash: str
    window_start_s: float
    window_end_s: float
    utterances: Tuple[JsonDict, ...]
    visible_observations: Tuple[JsonDict, ...]
    board_snapshot: JsonDict
    locations: Dict[str, JsonDict]

    def _evidence_numbers(self, field: str) -> List[int | float]:
        return [as_number(item[field], f"utterance {field}") for item in self.utterances]

    def plan_record(self) -> JsonDict:
        return {
            "job_id": self.job_id,
            "game_id": self.game_id,
            "replay_index": self.replay_index,
            "previous_replay_index": self.previous_replay_index,
            "anchor_kind": self.anchor_kind,
            "decision_ids": list(self.decision_ids),
            "input_hash": self.input_hash,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "source_start_s": min(self._evidence_numbers("start_s")),
            "source_end_s": max(self._evidence_numbers("end_s")),
            "source_start_index": min(self._evidence_numbers("source_start_index")),
            "source_end_index": max(self._evidence_numbers("source_end_index")),
            "utterance_count": len(self.utterances),
            "evidence_ids": [item["evidence_id"] for item in self.utterances],
            "visible_observation_count": len(self.visible_observations),
            "board_state_hash": self.board_snapshot["board_state_hash"],
        }


class ObservationScan(TypedDict):
    """One replay pass: provenance hashes, packet anchors, and every assembly job."""

    game_id: str
    total_events: int
    replay_sha256: str
    transcript_sha256: str
    narrator: JsonDict
    decision_count: int
    decision_plan_sha256: str
    decision_manifest_sha256: str
    anchors: List[JsonDict]
    evidence_count: int
    jobs: List[ObservationAssemblyJob]
