"""Assemble causal narrator commentary by decision and public observation."""

from __future__ import annotations

from cle.replay.runtime.access import get_game_engine

import contextlib
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from cle.replay.activity import format_visible_replay_activity
from cle.replay.runtime.step_executor import replay_step_logic
from playground.game_viewer.app import app
from playground.game_viewer.replay.transcript import (
    paired_transcript_fingerprint,
    parse_transcript_segments,
)
from playground.game_viewer.state import server_state
from playground.openrouter_client import query_text_with_tools

from evals.transcript_reasoning import (
    build_location_inspections,
    build_public_board_snapshot,
)

RUN_SCHEMA = "narrator-observation-assembly-run-v1"
ATTEMPT_SCHEMA = "narrator-observation-assembly-attempt-v1"
RESULT_SCHEMA = "narrator-observation-assembly-result-v1"
PROMPT_VERSION = "narrator-observation-assembly-v1"
DEFAULT_MODEL = "openai/gpt-5.6-sol"
MAX_PACKET_UTTERANCES = 24
MAX_PACKET_SECONDS = 45.0
MAX_PARAGRAPHS = 8
MAX_PARAGRAPH_CHARS = 1_200
PARAGRAPH_KINDS = frozenset(
    {
        "decision_reasoning",
        "board_observation",
        "opponent_assessment",
        "reaction",
        "reflection",
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISION_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
    / "action_selection_diff/qwen3_8_27b_blue_20260817"
)

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_board",
            "description": (
                "Inspect the authoritative public Catan board at this causal "
                "decision or observation cursor. Hidden hands and the next recorded "
                "action are unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_location",
            "description": (
                "Resolve one number-based location phrase that occurs verbatim in "
                "the supplied caption evidence. Ambiguity is preserved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "Exact location phrase from the captions.",
                    }
                },
                "required": ["reference"],
                "additionalProperties": False,
            },
        },
    },
]

_SYSTEM_PROMPT = f"""You are an editorial assembly agent for human Catan commentary.
Turn the supplied noisy YouTube ASR into coherent first-person reasoning for one
strictly causal decision or public-observation packet. The packet may span many replay
actions because human thoughts do not stop at engine row boundaries.

Rules ({PROMPT_VERSION}):
- Call inspect_board before answering. Use inspect_location only for a number phrase
  that appears in the supplied evidence.
- Use only this packet's captions, already-visible public events, and public board.
  The action at this cursor, future actions, and all hidden hands are unavailable.
- Assemble related sentences across replay boundaries. Remove rolling-caption
  repetition, filler, greetings, false starts, and verbal tics.
- Preserve uncertainty, alternatives, mistakes, and changes of mind. Do not improve
  the strategy or infer a chosen move, resource hand, route, visual pointer, or result.
- `decision_reasoning` is allowed only when the captions explicitly deliberate the
  narrator's current decision opportunity. Opponent comments and general table reads
  are observations, not narrator action rationale.
- Every evidence ID must appear exactly once: cite it in one paragraph, or put it in
  `omitted_evidence_ids` when it is filler, repetition, unrelated chatter, or too
  unclear to reconstruct safely.
- Return prose, not bullets. Return at most {MAX_PARAGRAPHS} paragraphs.

Allowed paragraph kinds: {", ".join(sorted(PARAGRAPH_KINDS))}.

Return only this JSON shape:
{{
  "paragraphs": [
    {{
      "kind": "decision_reasoning",
      "text": "one coherent first-person paragraph",
      "evidence_ids": ["e0001", "e0002"],
      "uncertainties": ["brief unresolved point, if any"]
    }}
  ],
  "omitted_evidence_ids": ["e0003"]
}}"""


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
    utterances: Tuple[Dict[str, Any], ...]
    visible_observations: Tuple[Dict[str, Any], ...]
    board_snapshot: Dict[str, Any]
    locations: Dict[str, Dict[str, Any]]

    def plan_record(self) -> Dict[str, Any]:
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
            "source_start_s": min(item["start_s"] for item in self.utterances),
            "source_end_s": max(item["end_s"] for item in self.utterances),
            "source_start_index": min(
                item["source_start_index"] for item in self.utterances
            ),
            "source_end_index": max(
                item["source_end_index"] for item in self.utterances
            ),
            "utterance_count": len(self.utterances),
            "evidence_ids": [item["evidence_id"] for item in self.utterances],
            "visible_observation_count": len(self.visible_observations),
            "board_state_hash": self.board_snapshot["board_state_hash"],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ObservationAssemblyError(f"Missing artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ObservationAssemblyError(f"Artifact is not an object: {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ObservationAssemblyError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _canonical_availability(plan: Dict[str, Any]) -> Dict[int, int]:
    availability = {}
    canonicalizations = plan.get("canonicalizations", [])
    if not isinstance(canonicalizations, list):
        raise ObservationAssemblyError("Decision plan canonicalizations must be a list")
    for change in canonicalizations:
        if not isinstance(change, dict):
            raise ObservationAssemblyError("Decision canonicalization must be an object")
        canonical_indices = change.get("canonical_indices")
        source_indices = change.get("source_replay_indices")
        if (
            not isinstance(canonical_indices, list)
            or len(canonical_indices) != 2
            or not all(isinstance(item, int) for item in canonical_indices)
            or canonical_indices[1] != canonical_indices[0] + 1
            or not isinstance(source_indices, list)
            or sorted(source_indices) != canonical_indices
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
    narrator_colonist_color: Optional[int],
    total_events: int,
) -> Dict[str, Any]:
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
    seen = set()
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


def _finite_wall_time(timing: Dict[str, Any]) -> Optional[float]:
    value = timing.get("wall_time_s")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _availability_cursor(end_s: float, timings: Sequence[Dict[str, Any]]) -> int:
    for replay_index, timing in enumerate(timings):
        wall_time = _finite_wall_time(timing)
        if wall_time is not None and end_s < wall_time:
            return replay_index
    return len(timings)


def build_global_evidence(paired_transcript: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Reflow the full transcript once and assign each utterance causally."""
    timings = paired_transcript.get("action_timings", [])
    if not isinstance(timings, list):
        raise ObservationAssemblyError("Transcript action timings must be a list")
    utterances = parse_transcript_segments(paired_transcript.get("segments", []))
    evidence = []
    for index, utterance in enumerate(utterances):
        evidence.append(
            {
                "evidence_id": f"e{index:04d}",
                **utterance,
                "available_replay_index": _availability_cursor(
                    float(utterance["end_s"]), timings
                ),
            }
        )
    return tuple(evidence)


def _packet_exceeds_limit(items: Sequence[Dict[str, Any]]) -> bool:
    if len(items) >= MAX_PACKET_UTTERANCES:
        return True
    if not items:
        return False
    return float(items[-1]["end_s"]) - float(items[0]["start_s"]) >= MAX_PACKET_SECONDS


def build_packet_anchors(
    evidence: Sequence[Dict[str, Any]],
    decision_ids_by_cursor: Dict[int, Tuple[str, ...]],
    total_events: int,
) -> Tuple[Dict[str, Any], ...]:
    """Add bounded public-observation checkpoints between narrator decisions."""
    if total_events < 0:
        raise ValueError("total_events must be nonnegative")
    required = sorted({*decision_ids_by_cursor, total_events})
    if any(cursor < 0 or cursor > total_events for cursor in required):
        raise ObservationAssemblyError("Decision anchor is outside the replay")

    anchors = set(required)
    previous_required = -1
    for required_cursor in required:
        interval_items = [
            item
            for item in evidence
            if previous_required < item["available_replay_index"] <= required_cursor
        ]
        grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for item in interval_items:
            grouped[item["available_replay_index"]].append(item)
        packet_items: List[Dict[str, Any]] = []
        for cursor in sorted(grouped):
            packet_items.extend(grouped[cursor])
            if cursor < required_cursor and _packet_exceeds_limit(packet_items):
                anchors.add(cursor)
                packet_items = []
        previous_required = required_cursor

    records = []
    for cursor in sorted(anchors):
        decision_ids = decision_ids_by_cursor.get(cursor, ())
        if decision_ids:
            kind = "decision"
        elif cursor == total_events:
            kind = "complete"
        else:
            kind = "observation"
        records.append(
            {
                "replay_index": cursor,
                "anchor_kind": kind,
                "decision_ids": list(decision_ids),
            }
        )
    return tuple(records)


def build_packet_specs(
    paired_transcript: Dict[str, Any],
    decision_ids_by_cursor: Dict[int, Tuple[str, ...]],
    total_events: int,
) -> Tuple[Dict[str, Any], ...]:
    """Partition globally reflowed evidence exactly once across causal anchors."""
    evidence = build_global_evidence(paired_transcript)
    anchors = build_packet_anchors(evidence, decision_ids_by_cursor, total_events)
    specs = []
    previous_cursor = -1
    for packet_index, anchor in enumerate(anchors):
        replay_index = anchor["replay_index"]
        owned = tuple(
            item
            for item in evidence
            if previous_cursor < item["available_replay_index"] <= replay_index
        )
        specs.append(
            {
                "packet_index": packet_index,
                "replay_index": replay_index,
                "previous_replay_index": previous_cursor,
                "anchor_kind": anchor["anchor_kind"],
                "decision_ids": tuple(anchor["decision_ids"]),
                "utterances": owned,
            }
        )
        previous_cursor = replay_index

    assigned_ids = [
        item["evidence_id"] for spec in specs for item in spec["utterances"]
    ]
    expected_ids = [item["evidence_id"] for item in evidence]
    if assigned_ids != expected_ids:
        raise ObservationAssemblyError("Packet anchors did not partition evidence exactly")
    return tuple(specs)


def _window_time(
    timings: Sequence[Dict[str, Any]], replay_index: int, fallback: float
) -> float:
    if replay_index < 0:
        return 0.0
    if replay_index >= len(timings):
        return fallback
    value = _finite_wall_time(timings[replay_index])
    return fallback if value is None else value


def _visible_observations(
    state: Any, previous_replay_index: int, replay_index: int
) -> Tuple[Dict[str, Any], ...]:
    parsed_actions = state.replay_data.get("parsed_actions", [])
    start = max(0, previous_replay_index)
    stop = min(replay_index, len(parsed_actions))
    colors = get_game_engine(state).state.colors
    return tuple(
        {
            "observation_id": f"o{index}",
            "replay_index": index,
            "summary": format_visible_replay_activity(
                parsed_actions[index],
                state.replay_data,
                colors,
                observer_color=None,
            ),
        }
        for index in range(start, stop)
    )


def _build_job_from_spec(state: Any, spec: Dict[str, Any]) -> ObservationAssemblyJob:
    utterances = tuple(deepcopy(spec["utterances"]))
    if not utterances:
        raise ObservationAssemblyError("Cannot generate a model job without evidence")
    board_snapshot = build_public_board_snapshot(state)
    visible_observations = _visible_observations(
        state, spec["previous_replay_index"], spec["replay_index"]
    )
    locations = build_location_inspections(state, utterances)
    timings = state.replay_data["paired_transcript"].get("action_timings", [])
    source_end = max(float(item["end_s"]) for item in utterances)
    window_start = _window_time(
        timings,
        spec["previous_replay_index"],
        min(float(item["start_s"]) for item in utterances),
    )
    window_end = _window_time(timings, spec["replay_index"], source_end)
    input_payload = {
        "prompt_version": PROMPT_VERSION,
        "game_id": str(state.replay_data.get("game_id")),
        "replay_index": spec["replay_index"],
        "previous_replay_index": spec["previous_replay_index"],
        "anchor_kind": spec["anchor_kind"],
        "decision_ids": spec["decision_ids"],
        "window_start_s": window_start,
        "window_end_s": window_end,
        "utterances": utterances,
        "visible_observations": visible_observations,
        "board_state_hash": board_snapshot["board_state_hash"],
        "locations": locations,
    }
    game_id = str(state.replay_data.get("game_id"))
    return ObservationAssemblyJob(
        job_id=f"{game_id}:observation:{spec['replay_index']}",
        game_id=game_id,
        replay_index=spec["replay_index"],
        previous_replay_index=spec["previous_replay_index"],
        anchor_kind=spec["anchor_kind"],
        decision_ids=tuple(spec["decision_ids"]),
        input_hash=_stable_hash(input_payload),
        window_start_s=window_start,
        window_end_s=window_end,
        utterances=utterances,
        visible_observations=visible_observations,
        board_snapshot=board_snapshot,
        locations=locations,
    )


def _step_replay_without_lookahead(state: Any) -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        result = replay_step_logic(state, lambda: None, allow_lookahead=False)
    if isinstance(result, tuple):
        payload, status = result
        raise ObservationAssemblyError(
            f"Replay step failed with status {status}: {payload}\n{output.getvalue()[-2000:]}"
        )
    if result.get("error"):
        raise ObservationAssemblyError(
            f"Replay step failed: {result}\n{output.getvalue()[-2000:]}"
        )


def scan_observation_jobs(
    game_id: str,
    decision_artifact_dir: Path = DEFAULT_DECISION_ARTIFACT_DIR,
) -> Dict[str, Any]:
    """Replay once and capture every nonempty causal assembly packet."""
    app.config["TESTING"] = True
    server_state.reset()
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            response = app.test_client().post("/api/load-replay", json={"game_id": game_id})
        if response.status_code != 200:
            raise ObservationAssemblyError(
                f"Could not load replay {game_id}: {response.get_json()}\n"
                f"{output.getvalue()[-2000:]}"
            )
        paired_transcript = server_state.replay_data.get("paired_transcript")
        if not paired_transcript:
            raise ObservationAssemblyError(f"Replay {game_id} has no transcript")
        total_events = int(server_state.replay_data.get("total_events", 0))
        narrator = deepcopy(paired_transcript.get("narrator") or {})
        decision_data = load_decision_anchors(
            decision_artifact_dir,
            game_id=str(game_id),
            narrator_colonist_color=narrator.get("colonist_color"),
            total_events=total_events,
        )
        specs = build_packet_specs(
            paired_transcript,
            decision_data["decision_ids_by_cursor"],
            total_events,
        )
        specs_by_cursor = {spec["replay_index"]: spec for spec in specs}
        jobs = []
        while True:
            spec = specs_by_cursor.get(server_state.replay_index)
            if spec is not None and spec["utterances"]:
                jobs.append(_build_job_from_spec(server_state, spec))
            if server_state.replay_index >= total_events:
                break
            _step_replay_without_lookahead(server_state)

        semantic_errors = [
            issue
            for issue in server_state.replay_semantic_issues
            if issue.get("severity") == "error"
        ]
        if semantic_errors:
            raise ObservationAssemblyError(
                f"Replay reconstruction reported {len(semantic_errors)} semantic errors"
            )
        replay_path = Path(server_state.replay_data["file"])
        return {
            "game_id": str(game_id),
            "total_events": total_events,
            "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
            "transcript_sha256": paired_transcript_fingerprint(paired_transcript),
            "narrator": narrator,
            "decision_count": decision_data["decision_count"],
            "decision_plan_sha256": decision_data["decision_plan_sha256"],
            "decision_manifest_sha256": decision_data["decision_manifest_sha256"],
            "anchors": [
                {
                    "replay_index": spec["replay_index"],
                    "anchor_kind": spec["anchor_kind"],
                    "decision_ids": list(spec["decision_ids"]),
                    "utterance_count": len(spec["utterances"]),
                }
                for spec in specs
            ],
            "evidence_count": sum(len(spec["utterances"]) for spec in specs),
            "jobs": jobs,
        }
    finally:
        server_state.reset()


def build_generation_plan(scan: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    jobs = scan["jobs"]
    return {
        "schema": RUN_SCHEMA,
        "generator_version": PROMPT_VERSION,
        "created_at": utc_now(),
        "game_id": scan["game_id"],
        "model_id": model_id,
        "total_replay_events": scan["total_events"],
        "replay_sha256": scan["replay_sha256"],
        "transcript_sha256": scan["transcript_sha256"],
        "decision_plan_sha256": scan["decision_plan_sha256"],
        "decision_manifest_sha256": scan["decision_manifest_sha256"],
        "narrator": scan["narrator"],
        "decision_count": scan["decision_count"],
        "evidence_count": scan["evidence_count"],
        "anchor_count": len(scan["anchors"]),
        "anchors": scan["anchors"],
        "job_count": len(jobs),
        "jobs": [job.plan_record() for job in jobs],
    }


def _user_prompt(job: ObservationAssemblyJob) -> str:
    transcript_lines = "\n".join(
        (
            f"[{item['evidence_id']} | {item['start_s']:.1f}-"
            f"{item['end_s']:.1f}s | source "
            f"{item['source_start_index']}-{item['source_end_index']}] "
            f"{item['text']}"
        )
        for item in job.utterances
    )
    observation_lines = "\n".join(
        f"[{item['observation_id']}] {item['summary']}"
        for item in job.visible_observations
    ) or "none"
    references = sorted(item["reference"] for item in job.locations.values())
    reference_text = ", ".join(references) if references else "none"
    decision_text = ", ".join(job.decision_ids) if job.decision_ids else "none"
    return f"""<causal_commentary_packet game_id="{job.game_id}" replay_index="{job.replay_index}">
<anchor kind="{job.anchor_kind}" narrator_decision_ids="{decision_text}" />
<availability>Everything below is visible at this cursor. The action at this cursor and
all future replay state are unavailable. Attach the reconstruction only to this current
packet; do not point it back to an older decision.</availability>
<already_visible_public_events>
{observation_lines}
</already_visible_public_events>
<known_location_phrases>{reference_text}</known_location_phrases>
<captions>
{transcript_lines}
</captions>
</causal_commentary_packet>

Inspect the board, partition every evidence ID, and assemble coherent paragraphs for
this decision or observation packet."""


def _tool_handler(
    job: ObservationAssemblyJob, name: str, arguments: Dict[str, Any]
) -> Any:
    if name == "inspect_board":
        return deepcopy(job.board_snapshot)
    if name == "inspect_location":
        reference = arguments.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            return {"error": "reference must be a nonempty string"}
        normalized = " ".join(reference.lower().split())
        result = job.locations.get(normalized)
        if result is None:
            return {
                "status": "unavailable",
                "reference": reference,
                "reason": "The phrase is not a number reference in this packet.",
                "available_references": sorted(
                    item["reference"] for item in job.locations.values()
                ),
            }
        return deepcopy(result)
    return {"error": f"Unknown tool {name!r}"}


def _parse_json_object(raw_response: str) -> Dict[str, Any]:
    normalized = _JSON_FENCE.sub("", raw_response.strip())
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ObservationAssemblyError(
            f"Model response is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ObservationAssemblyError("Model response must be a JSON object")
    return parsed


def parse_assembly_response(
    job: ObservationAssemblyJob, raw_response: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate prose and an exact evidence partition for one causal packet."""
    parsed = _parse_json_object(raw_response)
    raw_paragraphs = parsed.get("paragraphs")
    omitted_ids = parsed.get("omitted_evidence_ids")
    if not isinstance(raw_paragraphs, list):
        raise ObservationAssemblyError("Model response paragraphs must be a list")
    if len(raw_paragraphs) > MAX_PARAGRAPHS:
        raise ObservationAssemblyError(
            f"Model returned more than {MAX_PARAGRAPHS} paragraphs"
        )
    if not isinstance(omitted_ids, list) or not all(
        isinstance(item, str) and item for item in omitted_ids
    ):
        raise ObservationAssemblyError("omitted_evidence_ids must be strings")
    if len(omitted_ids) != len(set(omitted_ids)):
        raise ObservationAssemblyError("Omitted evidence IDs contain duplicates")

    evidence_by_id = {item["evidence_id"]: item for item in job.utterances}
    cited_ids = []
    paragraphs = []
    for paragraph_index, raw_paragraph in enumerate(raw_paragraphs):
        if not isinstance(raw_paragraph, dict):
            raise ObservationAssemblyError("Each paragraph must be an object")
        kind = raw_paragraph.get("kind")
        if kind not in PARAGRAPH_KINDS:
            raise ObservationAssemblyError(f"Unsupported paragraph kind: {kind!r}")
        if kind == "decision_reasoning" and job.anchor_kind != "decision":
            raise ObservationAssemblyError(
                "decision_reasoning is only valid at a decision anchor"
            )
        text = raw_paragraph.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ObservationAssemblyError("Each paragraph must contain text")
        text = " ".join(text.split())
        if len(text) > MAX_PARAGRAPH_CHARS:
            raise ObservationAssemblyError(
                f"Paragraph exceeds {MAX_PARAGRAPH_CHARS} characters"
            )
        evidence_ids = raw_paragraph.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or not all(isinstance(item, str) and item for item in evidence_ids)
        ):
            raise ObservationAssemblyError("Each paragraph must cite evidence")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ObservationAssemblyError("Paragraph evidence contains duplicates")
        cited_ids.extend(evidence_ids)
        uncertainties = raw_paragraph.get("uncertainties", [])
        if not isinstance(uncertainties, list) or not all(
            isinstance(item, str) and item.strip() for item in uncertainties
        ):
            raise ObservationAssemblyError("Paragraph uncertainties must be strings")
        evidence = [evidence_by_id.get(evidence_id) for evidence_id in evidence_ids]
        if any(item is None for item in evidence):
            unknown = sorted(set(evidence_ids) - set(evidence_by_id))
            raise ObservationAssemblyError(
                f"Paragraph cites unknown evidence IDs: {', '.join(unknown)}"
            )
        resolved_evidence = [item for item in evidence if item is not None]
        identity = {
            "job_id": job.job_id,
            "paragraph_index": paragraph_index,
            "kind": kind,
            "text": text,
            "evidence_ids": evidence_ids,
        }
        paragraphs.append(
            {
                "paragraph_id": _stable_hash(identity)[:16],
                "kind": kind,
                "text": text,
                "evidence_ids": evidence_ids,
                "uncertainties": [" ".join(item.split()) for item in uncertainties],
                "start_s": min(item["start_s"] for item in resolved_evidence),
                "end_s": max(item["end_s"] for item in resolved_evidence),
                "source_start_index": min(
                    item["source_start_index"] for item in resolved_evidence
                ),
                "source_end_index": max(
                    item["source_end_index"] for item in resolved_evidence
                ),
                "subject_replay_index": job.replay_index,
                "available_replay_index": job.replay_index,
                "anchor_kind": job.anchor_kind,
                "decision_ids": list(job.decision_ids),
            }
        )

    if len(cited_ids) != len(set(cited_ids)):
        raise ObservationAssemblyError("Evidence is cited by more than one paragraph")
    cited = set(cited_ids)
    omitted = set(omitted_ids)
    expected = set(evidence_by_id)
    if cited.intersection(omitted):
        raise ObservationAssemblyError("Evidence cannot be both cited and omitted")
    if cited.union(omitted) != expected:
        missing = sorted(expected - cited - omitted)
        extra = sorted(cited.union(omitted) - expected)
        raise ObservationAssemblyError(
            f"Evidence partition mismatch: missing={missing}, extra={extra}"
        )
    return paragraphs, omitted_ids


def generate_assembly_job(
    job: ObservationAssemblyJob,
    model_id: str = DEFAULT_MODEL,
    *,
    max_tokens: int = 2_000,
    timeout: float = 180.0,
    query: Callable[..., Dict[str, Any]] = query_text_with_tools,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one forced board-inspection loop and validate exact evidence use."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(job)},
    ]
    recorded_at = utc_now()
    response: Dict[str, Any] = {}
    try:
        response = query(
            model_id,
            messages,
            _TOOL_DEFINITIONS,
            lambda name, arguments: _tool_handler(job, name, arguments),
            temperature=0.0,
            max_tokens=max_tokens,
            timeout=timeout,
            provider="openrouter",
            forced_first_tool="inspect_board",
            response_format={"type": "json_object"},
            max_tool_rounds=6,
        )
        called_tools = response.get("called_tools") or []
        if not called_tools or called_tools[0] != "inspect_board":
            raise ObservationAssemblyError("Model did not inspect the board first")
        paragraphs, omitted_ids = parse_assembly_response(
            job, response.get("content") or ""
        )
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job.job_id,
            "game_id": job.game_id,
            "replay_index": job.replay_index,
            "previous_replay_index": job.previous_replay_index,
            "anchor_kind": job.anchor_kind,
            "decision_ids": list(job.decision_ids),
            "input_hash": job.input_hash,
            "model_id": model_id,
            "generator_version": PROMPT_VERSION,
            "status": "ready" if paragraphs else "empty",
            "recorded_at": recorded_at,
            "board_state_hash": job.board_snapshot["board_state_hash"],
            "window_start_s": job.window_start_s,
            "window_end_s": job.window_end_s,
            "called_tools": called_tools,
            "evidence_count": len(job.utterances),
            "omitted_evidence_ids": omitted_ids,
            "paragraphs": paragraphs,
            "usage": response.get("usage") or {},
            "latency_ms": response.get("latency_ms"),
            "error": None,
        }
        attempt = {
            "schema": ATTEMPT_SCHEMA,
            "job_id": job.job_id,
            "input_hash": job.input_hash,
            "model_id": model_id,
            "recorded_at": recorded_at,
            "status": result["status"],
            "messages": messages,
            "tool_messages": response.get("tool_messages") or [],
            "called_tools": called_tools,
            "raw_response": response.get("content") or "",
            "usage": response.get("usage") or {},
            "latency_ms": response.get("latency_ms"),
            "error": None,
        }
        return result, attempt
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job.job_id,
            "game_id": job.game_id,
            "replay_index": job.replay_index,
            "previous_replay_index": job.previous_replay_index,
            "anchor_kind": job.anchor_kind,
            "decision_ids": list(job.decision_ids),
            "input_hash": job.input_hash,
            "model_id": model_id,
            "generator_version": PROMPT_VERSION,
            "status": "error",
            "recorded_at": recorded_at,
            "board_state_hash": job.board_snapshot["board_state_hash"],
            "window_start_s": job.window_start_s,
            "window_end_s": job.window_end_s,
            "called_tools": response.get("called_tools") or [],
            "evidence_count": len(job.utterances),
            "omitted_evidence_ids": [],
            "paragraphs": [],
            "usage": response.get("usage") or {},
            "latency_ms": response.get("latency_ms"),
            "error": error,
        }
        attempt = {
            "schema": ATTEMPT_SCHEMA,
            "job_id": job.job_id,
            "input_hash": job.input_hash,
            "model_id": model_id,
            "recorded_at": recorded_at,
            "status": "error",
            "messages": messages,
            "tool_messages": response.get("tool_messages") or [],
            "called_tools": response.get("called_tools") or [],
            "raw_response": response.get("content") or "",
            "usage": response.get("usage") or {},
            "latency_ms": response.get("latency_ms"),
            "error": error,
        }
        return result, attempt


def _plan_identity(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "created_at"}


def run_generation(
    output_dir: Path,
    scan: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    workers: int = 4,
    max_tokens: int = 2_000,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Resume generation and atomically select the latest successful results."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = Path(output_dir)
    plan_path = output_dir / "plan.json"
    results_path = output_dir / "results.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    if plan_path.exists():
        existing_plan = _read_json(plan_path)
        if _plan_identity(existing_plan) != _plan_identity(plan):
            raise ObservationAssemblyError(
                "Existing output plan does not match this replay/model/input state"
            )
    else:
        _write_json(plan_path, plan)

    existing_results = {
        row["job_id"]: row
        for row in _read_jsonl(results_path)
        if row.get("input_hash")
    }
    jobs = [
        job
        for job in scan["jobs"]
        if not (
            existing_results.get(job.job_id, {}).get("input_hash") == job.input_hash
            and existing_results.get(job.job_id, {}).get("status")
            in {"ready", "empty"}
        )
    ]
    completed = dict(existing_results)
    model_id = str(plan["model_id"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_assembly_job,
                job,
                model_id,
                max_tokens=max_tokens,
                timeout=timeout,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            result, attempt = future.result()
            completed[result["job_id"]] = result
            _append_jsonl(attempts_path, [attempt])
            ordered = [
                completed[job.job_id]
                for job in scan["jobs"]
                if job.job_id in completed
            ]
            _write_jsonl(results_path, ordered)
            print(
                f"[{len(ordered)}/{len(scan['jobs'])}] "
                f"{result['job_id']} {result['status']}"
            )
    return verify_generation_artifact(output_dir)


def verify_generation_artifact(output_dir: Path) -> Dict[str, Any]:
    """Validate causal identity, exact evidence partition, and board inspection."""
    output_dir = Path(output_dir)
    plan = _read_json(output_dir / "plan.json")
    results = _read_jsonl(output_dir / "results.jsonl")
    if plan.get("schema") != RUN_SCHEMA:
        raise ObservationAssemblyError("Unexpected generation plan schema")
    plan_jobs = plan.get("jobs")
    if not isinstance(plan_jobs, list):
        raise ObservationAssemblyError("Generation plan jobs must be a list")
    expected = {job["job_id"]: job for job in plan_jobs}
    if len(expected) != len(plan_jobs):
        raise ObservationAssemblyError("Generation plan contains duplicate jobs")
    actual = {result.get("job_id"): result for result in results}
    if len(actual) != len(results):
        raise ObservationAssemblyError("Generation results contain duplicate jobs")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ObservationAssemblyError(
            f"Generation result coverage mismatch: missing={missing}, extra={extra}"
        )

    paragraph_count = 0
    empty_count = 0
    omitted_count = 0
    total_cost = 0.0
    for job_id, job in expected.items():
        result = actual[job_id]
        for field in (
            "schema",
            "input_hash",
            "model_id",
            "generator_version",
            "replay_index",
            "previous_replay_index",
            "anchor_kind",
            "decision_ids",
            "board_state_hash",
            "window_start_s",
            "window_end_s",
        ):
            expected_value = {
                "schema": RESULT_SCHEMA,
                "model_id": plan.get("model_id"),
                "generator_version": plan.get("generator_version"),
            }.get(field, job.get(field))
            if result.get(field) != expected_value:
                raise ObservationAssemblyError(f"{field} mismatch for {job_id}")
        if result.get("status") not in {"ready", "empty"}:
            raise ObservationAssemblyError(f"Unsuccessful result for {job_id}")
        called_tools = result.get("called_tools") or []
        if not called_tools or called_tools[0] != "inspect_board":
            raise ObservationAssemblyError(f"Board inspection missing for {job_id}")
        paragraphs = result.get("paragraphs")
        omitted = result.get("omitted_evidence_ids")
        if not isinstance(paragraphs, list) or not isinstance(omitted, list):
            raise ObservationAssemblyError(f"Invalid evidence output for {job_id}")
        cited = [item for paragraph in paragraphs for item in paragraph["evidence_ids"]]
        if len(cited) != len(set(cited)) or len(omitted) != len(set(omitted)):
            raise ObservationAssemblyError(f"Duplicate evidence output for {job_id}")
        if set(cited).intersection(omitted) or set(cited).union(omitted) != set(
            job["evidence_ids"]
        ):
            raise ObservationAssemblyError(f"Evidence partition mismatch for {job_id}")
        if result["status"] == "ready" and not paragraphs:
            raise ObservationAssemblyError(f"Ready result has no paragraphs for {job_id}")
        if result["status"] == "empty":
            if paragraphs:
                raise ObservationAssemblyError(f"Empty result has paragraphs for {job_id}")
            empty_count += 1
        for paragraph in paragraphs:
            if paragraph.get("kind") not in PARAGRAPH_KINDS:
                raise ObservationAssemblyError(f"Invalid paragraph kind for {job_id}")
            if not (
                job["source_start_s"]
                <= paragraph["start_s"]
                <= paragraph["end_s"]
                <= job["source_end_s"]
                <= job["window_end_s"]
            ):
                raise ObservationAssemblyError(
                    f"Paragraph crosses packet bounds for {job_id}"
                )
            if (
                paragraph.get("subject_replay_index") != job["replay_index"]
                or paragraph.get("available_replay_index") != job["replay_index"]
            ):
                raise ObservationAssemblyError(
                    f"Paragraph violates strict causal attachment for {job_id}"
                )
        paragraph_count += len(paragraphs)
        omitted_count += len(omitted)
        cost = (result.get("usage") or {}).get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total_cost += float(cost)

    return {
        "schema": "narrator-observation-assembly-verification-v1",
        "game_id": plan["game_id"],
        "model_id": plan["model_id"],
        "decision_count": plan["decision_count"],
        "anchor_count": plan["anchor_count"],
        "job_count": len(expected),
        "ready_count": len(expected) - empty_count,
        "empty_count": empty_count,
        "paragraph_count": paragraph_count,
        "evidence_count": plan["evidence_count"],
        "omitted_evidence_count": omitted_count,
        "total_cost": round(total_cost, 6),
    }
