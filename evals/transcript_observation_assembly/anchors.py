"""Decision anchors and evidence availability over the paired transcript."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

from evals.json_types import JsonDict, JsonValue
from evals.transcript_observation_assembly.artifacts import _read_json, _read_jsonl
from evals.transcript_observation_assembly.job import ObservationAssemblyError


class DecisionAnchors(TypedDict):
    """Validated narrator decision identities keyed by their safe viewer cursor."""

    decision_ids_by_cursor: Dict[int, Tuple[str, ...]]
    decision_count: int
    decision_plan_sha256: str
    decision_manifest_sha256: str


def _int_items(value: JsonValue) -> Optional[List[int]]:
    if not isinstance(value, list):
        return None
    items = [item for item in value if isinstance(item, int)]
    return items if len(items) == len(value) else None


def _sorted_numbers(value: JsonValue) -> Optional[List[int | float]]:
    if not isinstance(value, list):
        return None
    items = [item for item in value if isinstance(item, (int, float))]
    return sorted(items) if len(items) == len(value) else None


def _canonical_availability(plan: JsonDict) -> Dict[int, int]:
    availability: Dict[int, int] = {}
    canonicalizations = plan.get("canonicalizations", [])
    if not isinstance(canonicalizations, list):
        raise ObservationAssemblyError("Decision plan canonicalizations must be a list")
    for change in canonicalizations:
        if not isinstance(change, dict):
            raise ObservationAssemblyError("Decision canonicalization must be an object")
        canonical_indices = _int_items(change.get("canonical_indices"))
        source_indices = _sorted_numbers(change.get("source_replay_indices"))
        if (
            canonical_indices is None
            or len(canonical_indices) != 2
            or canonical_indices[1] != canonical_indices[0] + 1
            or source_indices is None
            or source_indices != canonical_indices
        ):
            raise ObservationAssemblyError("Invalid same-event decision canonicalization")
        first, second = canonical_indices
        availability[first] = first
        availability[second] = second + 1
    return availability


def load_decision_anchors(
    artifact_dir: Path,
    *,
    game_id: str,
    narrator_colonist_color: JsonValue,
    total_events: int,
) -> DecisionAnchors:
    """Load only validated decision identities and their safe viewer cursors."""
    artifact_dir = Path(artifact_dir)
    plan_path = artifact_dir / "plan.json"
    manifest_path = artifact_dir / "decision_manifest.jsonl"
    plan = _read_json(plan_path)
    rows = _read_jsonl(manifest_path)
    if str(plan.get("game_id")) != str(game_id):
        raise ObservationAssemblyError("Decision artifact belongs to another game")
    target_player = plan.get("target_player_id")
    if narrator_colonist_color is not None and target_player != narrator_colonist_color:
        raise ObservationAssemblyError("Decision artifact does not target the narrator")

    reordered = _canonical_availability(plan)
    by_cursor: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row.get("classification") != "exact":
            continue
        decision_id = row.get("decision_id")
        replay_index = row.get("replay_index")
        actor = row.get("actor")
        if (
            not isinstance(decision_id, str)
            or not decision_id
            or decision_id in seen
            or not isinstance(replay_index, int)
            or isinstance(replay_index, bool)
            or replay_index < 0
            or replay_index >= total_events
            or not isinstance(actor, dict)
        ):
            raise ObservationAssemblyError("Malformed exact decision anchor")
        actor_player = actor.get("colonist_player")
        if actor_player is not None and actor_player != target_player:
            raise ObservationAssemblyError("Exact decision actor is not the narrator")
        available_cursor = reordered.get(replay_index, replay_index)
        if available_cursor > total_events:
            raise ObservationAssemblyError("Decision availability exceeds replay")
        by_cursor[available_cursor].append((replay_index, decision_id))
        seen.add(decision_id)

    normalized = {
        cursor: tuple(decision_id for _, decision_id in sorted(items))
        for cursor, items in by_cursor.items()
    }
    return {
        "decision_ids_by_cursor": normalized,
        "decision_count": len(seen),
        "decision_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "decision_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }


def _finite_wall_time(timing: Mapping[str, object]) -> Optional[float]:
    value = timing.get("wall_time_s")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _availability_cursor(end_s: float, timings: Sequence[Mapping[str, object]]) -> int:
    for replay_index, timing in enumerate(timings):
        wall_time = _finite_wall_time(timing)
        if wall_time is not None and end_s < wall_time:
            return replay_index
    return len(timings)

