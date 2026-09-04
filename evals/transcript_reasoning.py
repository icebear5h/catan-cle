"""Generate board-grounded narrator reasoning from paired YouTube captions."""

from __future__ import annotations

from cle.replay.runtime.access import get_game_engine

import contextlib
import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from cle.game_engine.models.enums import CITY, SETTLEMENT
from playground.game_viewer.app import app
from playground.game_viewer.commentary.references import (
    build_corner_index,
    ground_text_references,
)
from cle.replay.activity import (
    format_visible_replay_activity,
    select_recent_activity_rows,
)
from cle.replay.runtime.step_executor import replay_step_logic
from playground.game_viewer.replay.transcript import (
    build_paired_transcript_window,
    paired_transcript_fingerprint,
)
from playground.game_viewer.state import server_state
from playground.openrouter_client import query_text_with_tools

RUN_SCHEMA = "narrator-reasoning-run-v1"
ATTEMPT_SCHEMA = "narrator-reasoning-attempt-v1"
RESULT_SCHEMA = "narrator-reasoning-result-v1"
PROMPT_VERSION = "narrator-reasoning-v2"
DEFAULT_MODEL = "openai/gpt-5.6-sol"
MAX_PARAGRAPHS = 3
MAX_PARAGRAPH_CHARS = 1_200

_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_board",
            "description": (
                "Inspect the authoritative public Catan board and public event context "
                "at this exact pre-action transcript cursor. Hidden hands and the next "
                "recorded action are unavailable."
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
                "Resolve one number-based location phrase that appears verbatim in the "
                "caption evidence, such as '8 4 10'. Ambiguity is preserved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {
                        "type": "string",
                        "description": "Exact location phrase from the supplied captions.",
                    }
                },
                "required": ["reference"],
                "additionalProperties": False,
            },
        },
    },
]

_SYSTEM_PROMPT = f"""You are an editorial reconstruction agent for human Catan commentary.
Your job is to turn noisy YouTube ASR captions into faithful, coherent first-person
reasoning paragraphs. You are reconstructing what the narrator expressed, not writing
a stronger strategy and not predicting the next replay action.

Rules ({PROMPT_VERSION}):
- Call inspect_board before answering. Use inspect_location for number-based board
  references when its result can clarify the wording.
- Use only supplied captions and tool facts from this exact cursor. The next recorded
  action and all future events are unavailable.
- Remove greetings, filler, repeated rolling-caption text, false starts, and verbal
  tics. Repair grammar and sentence boundaries.
- Preserve the narrator's uncertainty, alternatives, commitments, and mistakes. Do
  not silently improve the strategy or invent motives, resources, routes, players,
  visual pointers, or outcomes.
- Resolve phrases such as "this spot" only when the captions plus tool evidence make
  the referent unique. Otherwise keep the uncertainty explicit or omit the unsupported
  referent.
- Do not turn opponent speculation into fact. Use engine colors only when identity is
  supported; otherwise retain a neutral reference such as "another player."
- Return zero paragraphs when the captions contain only greetings, filler, reactions,
  or unrelated chatter.
- Each paragraph must be prose, not bullets, and must cite the caption evidence IDs it
  actually reconstructs. Return at most {MAX_PARAGRAPHS} paragraphs.

Return only this JSON shape:
{{
  "paragraphs": [
    {{
      "text": "one coherent first-person paragraph",
      "evidence_ids": ["u0", "u1"],
      "uncertainties": ["brief unresolved point, if any"]
    }}
  ]
}}"""


@dataclass(frozen=True)
class ReasoningJob:
    """One transcript interval and its immutable public board inspection tools."""

    job_id: str
    game_id: str
    replay_index: int
    input_hash: str
    window_start_s: float
    window_end_s: float
    utterances: Tuple[Dict[str, Any], ...]
    board_snapshot: Dict[str, Any]
    locations: Dict[str, Dict[str, Any]]

    def plan_record(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "game_id": self.game_id,
            "replay_index": self.replay_index,
            "input_hash": self.input_hash,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "source_start_s": min(
                utterance["start_s"] for utterance in self.utterances
            ),
            "source_end_s": max(
                utterance["end_s"] for utterance in self.utterances
            ),
            "source_start_index": min(
                utterance["source_start_index"] for utterance in self.utterances
            ),
            "source_end_index": max(
                utterance["source_end_index"] for utterance in self.utterances
            ),
            "utterance_count": len(self.utterances),
            "board_state_hash": self.board_snapshot["board_state_hash"],
        }


class NarratorReasoningError(ValueError):
    """Raised when generation inputs or artifacts violate their contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _color_name(color: Any) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_edge(edge: Sequence[int]) -> Tuple[int, int]:
    return min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1]))


def _colonist_mapping(mapping: Dict[str, Any]) -> Dict[int, int]:
    return {
        int(key.removeprefix("_")): int(value)
        for key, value in mapping.items()
    }


def _public_player_summaries(game_state: Any) -> List[Dict[str, Any]]:
    board = game_state.board
    unique_roads = {
        (*_normalize_edge(edge), _color_name(color))
        for edge, color in board.roads.items()
    }
    summaries = []
    for player_index, color in enumerate(game_state.colors):
        color_name = _color_name(color)
        buildings = game_state.buildings_by_color.get(color, {})
        summaries.append(
            {
                "color": color_name,
                "public_victory_points": game_state.player_state.get(
                    f"P{player_index}_VICTORY_POINTS", 0
                ),
                "settlements": len(buildings.get(SETTLEMENT, [])),
                "cities": len(buildings.get(CITY, [])),
                "roads": sum(road_color == color_name for _, _, road_color in unique_roads),
                "longest_road_length": game_state.player_state.get(
                    f"P{player_index}_LONGEST_ROAD_LENGTH", 0
                ),
                "played_knights": game_state.player_state.get(
                    f"P{player_index}_PLAYED_KNIGHT", 0
                ),
            }
        )
    return summaries


def build_public_board_snapshot(state: Any) -> Dict[str, Any]:
    """Return public board facts without hidden hands or the upcoming action."""
    game_state = get_game_engine(state).state
    board = game_state.board
    catan_map = board.map
    corner_to_node = _colonist_mapping(state.corner_to_node_map)
    node_to_corner = {node: corner for corner, node in corner_to_node.items()}
    edge_to_nodes = {
        int(key.removeprefix("_")): _normalize_edge(value)
        for key, value in state.edge_to_edge_map.items()
    }
    nodes_to_edge = {nodes: edge for edge, nodes in edge_to_nodes.items()}
    coordinate_by_tile_identity = {
        id(tile): tuple(coordinate) for coordinate, tile in catan_map.land_tiles.items()
    }

    tiles = []
    for coordinate, tile in sorted(catan_map.land_tiles.items()):
        tiles.append(
            {
                "tile_id": int(tile.id),
                "coordinate": list(coordinate),
                "resource": tile.resource,
                "number": tile.number,
                "robber": tuple(coordinate) == tuple(board.robber_coordinate),
            }
        )

    buildings = []
    for node, (color, building) in sorted(board.buildings.items()):
        adjacent_tiles = []
        for tile in catan_map.adjacent_tiles[node]:
            adjacent_tiles.append(
                {
                    "tile_id": int(tile.id),
                    "coordinate": list(coordinate_by_tile_identity[id(tile)]),
                    "resource": tile.resource,
                    "number": tile.number,
                }
            )
        ports = []
        for resource, port_nodes in catan_map.port_nodes.items():
            if node in port_nodes:
                ports.append("3:1" if resource is None else f"2:1 {resource}")
        buildings.append(
            {
                "colonist_corner_id": node_to_corner.get(int(node)),
                "engine_node_id": int(node),
                "color": _color_name(color),
                "building": _color_name(building),
                "adjacent_tiles": sorted(adjacent_tiles, key=lambda item: item["tile_id"]),
                "ports": sorted(ports),
            }
        )

    roads = []
    seen_roads = set()
    for edge, color in board.roads.items():
        normalized = _normalize_edge(edge)
        if normalized in seen_roads:
            continue
        seen_roads.add(normalized)
        roads.append(
            {
                "colonist_edge_id": nodes_to_edge.get(normalized),
                "engine_nodes": list(normalized),
                "color": _color_name(color),
            }
        )
    roads.sort(key=lambda item: (item["colonist_edge_id"] is None, item["colonist_edge_id"]))

    activity_rows, activity_window = select_recent_activity_rows(
        state.replay_data.get("parsed_actions", []), state.replay_index
    )
    public_activity = [
        format_visible_replay_activity(
            action,
            state.replay_data,
            game_state.colors,
            observer_color=None,
        )
        for action in activity_rows
    ]
    narrator = (state.replay_data.get("paired_transcript") or {}).get("narrator") or {}
    narrator_index = state.replay_data.get("colonist_color_to_engine_idx", {}).get(
        str(narrator.get("colonist_color"))
    )
    narrator_color = (
        _color_name(game_state.colors[narrator_index])
        if isinstance(narrator_index, int) and 0 <= narrator_index < len(game_state.colors)
        else None
    )

    public_payload = {
        "game_id": str(state.replay_data.get("game_id")),
        "replay_index": state.replay_index,
        "phase": (
            "initial_placement" if game_state.is_initial_build_phase else "main_game"
        ),
        "turn_number": game_state.num_turns,
        "current_player_color": _color_name(game_state.current_color()),
        "narrator_color": narrator_color,
        "tiles": tiles,
        "buildings": buildings,
        "roads": roads,
        "players": _public_player_summaries(game_state),
        "recent_public_activity": public_activity,
        "activity_window": activity_window,
    }
    public_payload["board_state_hash"] = _stable_hash(public_payload)
    return public_payload


def _serialize_grounding(result: Any) -> Dict[str, Any]:
    candidates = []
    for candidate in result.candidates:
        corner = candidate.corner
        candidates.append(
            {
                "colonist_corner_id": corner.colonist_corner_id,
                "engine_node_id": corner.engine_node_id,
                "numbers": list(corner.numbers),
                "ports": list(corner.ports),
                "coast": corner.coast,
                "occupied_by": candidate.occupied_by,
                "building": candidate.building,
                "adjacent_tiles": [
                    {
                        "tile_id": tile.tile_id,
                        "number": tile.number,
                        "resource": tile.resource,
                        "coordinate": list(tile.coordinate),
                    }
                    for tile in corner.tiles
                ],
            }
        )
    return {
        "reference": result.mention.surface,
        "normalized_number_options": [list(item) for item in result.mention.number_options],
        "direction_word": result.mention.direction,
        "status": result.status,
        "candidates": candidates,
        "note": (
            "Direction words are retained as speech evidence but are not used to guess "
            "among otherwise ambiguous candidates."
        ),
    }


def build_location_inspections(
    state: Any,
    utterances: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Ground only number references that occur in this transcript interval."""
    corner_index = build_corner_index(get_game_engine(state).state.board.map)
    inspections = {}
    for utterance in utterances:
        for result in ground_text_references(
            str(utterance["text"]),
            corner_index,
            get_game_engine(state).state,
            include_legality=False,
        ):
            inspections[" ".join(result.mention.surface.lower().split())] = (
                _serialize_grounding(result)
            )
    return inspections


def build_reasoning_job(state: Any) -> Optional[ReasoningJob]:
    """Build one immutable job from the current pre-action replay cursor."""
    window = build_paired_transcript_window(state.replay_data, state.replay_index)
    if window is None or not window.get("segments"):
        return None

    utterances = tuple(
        {
            "evidence_id": f"u{index}",
            "start_s": segment["start_s"],
            "end_s": segment["end_s"],
            "text": segment["text"],
            "source_start_index": segment["source_start_index"],
            "source_end_index": segment["source_end_index"],
            "source_segment_count": segment["source_segment_count"],
        }
        for index, segment in enumerate(window["segments"])
    )
    board_snapshot = build_public_board_snapshot(state)
    locations = build_location_inspections(state, utterances)
    input_payload = {
        "prompt_version": PROMPT_VERSION,
        "game_id": str(state.replay_data.get("game_id")),
        "replay_index": state.replay_index,
        "window_start_s": window["window_start_s"],
        "window_end_s": window["window_end_s"],
        "utterances": utterances,
        "board_state_hash": board_snapshot["board_state_hash"],
        "locations": locations,
    }
    game_id = str(state.replay_data.get("game_id"))
    return ReasoningJob(
        job_id=f"{game_id}:{state.replay_index}",
        game_id=game_id,
        replay_index=state.replay_index,
        input_hash=_stable_hash(input_payload),
        window_start_s=float(window["window_start_s"]),
        window_end_s=float(window["window_end_s"]),
        utterances=utterances,
        board_snapshot=board_snapshot,
        locations=locations,
    )


def _step_replay_without_lookahead(state: Any) -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        result = replay_step_logic(state, lambda: None, allow_lookahead=False)
    if isinstance(result, tuple):
        payload, status = result
        raise NarratorReasoningError(
            f"Replay step failed with status {status}: {payload}\n{output.getvalue()[-2000:]}"
        )
    if result.get("error"):
        raise NarratorReasoningError(
            f"Replay step failed: {result}\n{output.getvalue()[-2000:]}"
        )


def scan_reasoning_jobs(game_id: str) -> Dict[str, Any]:
    """Replay a game once and capture every nonempty transcript interval."""
    app.config["TESTING"] = True
    server_state.reset()
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            response = app.test_client().post("/api/load-replay", json={"game_id": game_id})
        if response.status_code != 200:
            raise NarratorReasoningError(
                f"Could not load replay {game_id}: {response.get_json()}\n"
                f"{output.getvalue()[-2000:]}"
            )

        paired_transcript = server_state.replay_data.get("paired_transcript")
        if not paired_transcript:
            raise NarratorReasoningError(f"Replay {game_id} has no paired transcript")
        jobs = []
        total_events = int(server_state.replay_data.get("total_events", 0))
        while True:
            job = build_reasoning_job(server_state)
            if job is not None:
                jobs.append(job)
            if server_state.replay_index >= total_events:
                break
            _step_replay_without_lookahead(server_state)

        semantic_errors = [
            issue
            for issue in server_state.replay_semantic_issues
            if issue.get("severity") == "error"
        ]
        if semantic_errors:
            raise NarratorReasoningError(
                f"Replay reconstruction reported {len(semantic_errors)} semantic errors"
            )
        replay_path = Path(server_state.replay_data["file"])
        return {
            "game_id": str(game_id),
            "total_events": total_events,
            "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
            "transcript_sha256": paired_transcript_fingerprint(paired_transcript),
            "narrator": deepcopy(paired_transcript.get("narrator") or {}),
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
        "narrator": scan["narrator"],
        "job_count": len(jobs),
        "jobs": [job.plan_record() for job in jobs],
    }


def _user_prompt(job: ReasoningJob) -> str:
    transcript_lines = "\n".join(
        (
            f"[{utterance['evidence_id']} | {utterance['start_s']:.1f}-"
            f"{utterance['end_s']:.1f}s | source "
            f"{utterance['source_start_index']}-{utterance['source_end_index']}] "
            f"{utterance['text']}"
        )
        for utterance in job.utterances
    )
    references = sorted(item["reference"] for item in job.locations.values())
    reference_text = ", ".join(references) if references else "none"
    return f"""<narrator_commentary game_id="{job.game_id}" replay_index="{job.replay_index}">
<availability>All captions end before the replay action at this cursor. No future action
or future board state may be used.</availability>
<known_location_phrases>{reference_text}</known_location_phrases>
<captions>
{transcript_lines}
</captions>
</narrator_commentary>

Inspect the board, then reconstruct only the substantive reasoning in these captions."""


def _tool_handler(job: ReasoningJob, name: str, arguments: Dict[str, Any]) -> Any:
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
                "reason": "The phrase is not a number-based reference in this caption window.",
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
        raise NarratorReasoningError(f"Model response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise NarratorReasoningError("Model response must be a JSON object")
    return parsed


def parse_reasoning_response(job: ReasoningJob, raw_response: str) -> List[Dict[str, Any]]:
    """Validate model prose and derive all timestamps from cited source evidence."""
    parsed = _parse_json_object(raw_response)
    raw_paragraphs = parsed.get("paragraphs")
    if not isinstance(raw_paragraphs, list):
        raise NarratorReasoningError("Model response paragraphs must be a list")
    if len(raw_paragraphs) > MAX_PARAGRAPHS:
        raise NarratorReasoningError(
            f"Model returned more than {MAX_PARAGRAPHS} paragraphs"
        )

    evidence_by_id = {
        utterance["evidence_id"]: utterance for utterance in job.utterances
    }
    paragraphs = []
    for paragraph_index, raw_paragraph in enumerate(raw_paragraphs):
        if not isinstance(raw_paragraph, dict):
            raise NarratorReasoningError("Each paragraph must be an object")
        text = raw_paragraph.get("text")
        if not isinstance(text, str) or not text.strip():
            raise NarratorReasoningError("Each paragraph must contain nonempty text")
        text = " ".join(text.split())
        if len(text) > MAX_PARAGRAPH_CHARS:
            raise NarratorReasoningError(
                f"Paragraph exceeds {MAX_PARAGRAPH_CHARS} characters"
            )

        evidence_ids = raw_paragraph.get("evidence_ids")
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or not all(isinstance(item, str) and item for item in evidence_ids)
        ):
            raise NarratorReasoningError(
                "Each paragraph must cite at least one evidence ID"
            )
        evidence_ids = list(dict.fromkeys(evidence_ids))
        unknown_ids = sorted(set(evidence_ids) - set(evidence_by_id))
        if unknown_ids:
            raise NarratorReasoningError(
                f"Paragraph cites unknown evidence IDs: {', '.join(unknown_ids)}"
            )
        evidence = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]

        uncertainties = raw_paragraph.get("uncertainties", [])
        if not isinstance(uncertainties, list) or not all(
            isinstance(item, str) and item.strip() for item in uncertainties
        ):
            raise NarratorReasoningError("Paragraph uncertainties must be strings")
        uncertainties = [" ".join(item.split()) for item in uncertainties]
        paragraph_identity = {
            "job_id": job.job_id,
            "paragraph_index": paragraph_index,
            "text": text,
            "evidence_ids": evidence_ids,
        }
        paragraphs.append(
            {
                "paragraph_id": _stable_hash(paragraph_identity)[:16],
                "text": text,
                "evidence_ids": evidence_ids,
                "uncertainties": uncertainties,
                "start_s": min(item["start_s"] for item in evidence),
                "end_s": max(item["end_s"] for item in evidence),
                "source_start_index": min(
                    item["source_start_index"] for item in evidence
                ),
                "source_end_index": max(item["source_end_index"] for item in evidence),
            }
        )
    return paragraphs


def generate_reasoning_job(
    job: ReasoningJob,
    model_id: str = DEFAULT_MODEL,
    *,
    max_tokens: int = 1_200,
    timeout: float = 180.0,
    query: Callable[..., Dict[str, Any]] = query_text_with_tools,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one forced board-inspection tool loop and validate its paragraphs."""
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
            raise NarratorReasoningError("Model did not inspect the board first")
        paragraphs = parse_reasoning_response(job, response.get("content") or "")
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job.job_id,
            "game_id": job.game_id,
            "replay_index": job.replay_index,
            "input_hash": job.input_hash,
            "model_id": model_id,
            "generator_version": PROMPT_VERSION,
            "status": "ready" if paragraphs else "empty",
            "recorded_at": recorded_at,
            "board_state_hash": job.board_snapshot["board_state_hash"],
            "window_start_s": job.window_start_s,
            "window_end_s": job.window_end_s,
            "called_tools": called_tools,
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
        result = {
            "schema": RESULT_SCHEMA,
            "job_id": job.job_id,
            "game_id": job.game_id,
            "replay_index": job.replay_index,
            "input_hash": job.input_hash,
            "model_id": model_id,
            "generator_version": PROMPT_VERSION,
            "status": "error",
            "recorded_at": recorded_at,
            "board_state_hash": job.board_snapshot["board_state_hash"],
            "window_start_s": job.window_start_s,
            "window_end_s": job.window_end_s,
            "called_tools": response.get("called_tools") or [],
            "paragraphs": [],
            "usage": response.get("usage") or {},
            "latency_ms": response.get("latency_ms"),
            "error": {"type": type(exc).__name__, "message": str(exc)},
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
            "error": result["error"],
        }
        return result, attempt


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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise NarratorReasoningError(f"Missing artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise NarratorReasoningError(f"Artifact is not an object: {path}")
    return data


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise NarratorReasoningError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def _plan_identity(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in plan.items()
        if key not in {"created_at"}
    }


def run_generation(
    output_dir: Path,
    scan: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    workers: int = 4,
    max_tokens: int = 1_200,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Resume a generation run and atomically materialize latest selected results."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = Path(output_dir)
    plan_path = output_dir / "plan.json"
    results_path = output_dir / "results.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    if plan_path.exists():
        existing_plan = _read_json(plan_path)
        if _plan_identity(existing_plan) != _plan_identity(plan):
            raise NarratorReasoningError(
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
            and existing_results.get(job.job_id, {}).get("status") in {"ready", "empty"}
        )
    ]
    completed = dict(existing_results)
    model_id = str(plan["model_id"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_reasoning_job,
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
            ordered_results = [
                completed[job.job_id]
                for job in scan["jobs"]
                if job.job_id in completed
            ]
            _write_jsonl(results_path, ordered_results)
            print(
                f"[{len(ordered_results)}/{len(scan['jobs'])}] "
                f"{result['job_id']} {result['status']}"
            )

    return verify_generation_artifact(output_dir)


def verify_generation_artifact(output_dir: Path) -> Dict[str, Any]:
    """Validate completeness, input identity, evidence bounds, and board-tool use."""
    output_dir = Path(output_dir)
    plan = _read_json(output_dir / "plan.json")
    results = _read_jsonl(output_dir / "results.jsonl")
    if plan.get("schema") != RUN_SCHEMA:
        raise NarratorReasoningError("Unexpected generation plan schema")
    plan_jobs = plan.get("jobs")
    if not isinstance(plan_jobs, list):
        raise NarratorReasoningError("Generation plan jobs must be a list")
    expected = {job["job_id"]: job for job in plan_jobs}
    if len(expected) != len(plan_jobs):
        raise NarratorReasoningError("Generation plan contains duplicate job IDs")
    actual = {result.get("job_id"): result for result in results}
    if len(actual) != len(results):
        raise NarratorReasoningError("Generation results contain duplicate job IDs")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise NarratorReasoningError(
            f"Generation result coverage mismatch: missing={missing}, extra={extra}"
        )

    paragraph_count = 0
    empty_count = 0
    total_cost = 0.0
    for job_id, job in expected.items():
        result = actual[job_id]
        if result.get("schema") != RESULT_SCHEMA:
            raise NarratorReasoningError(f"Unexpected result schema for {job_id}")
        if result.get("input_hash") != job.get("input_hash"):
            raise NarratorReasoningError(f"Input hash mismatch for {job_id}")
        if result.get("model_id") != plan.get("model_id"):
            raise NarratorReasoningError(f"Model mismatch for {job_id}")
        if result.get("status") not in {"ready", "empty"}:
            raise NarratorReasoningError(f"Unsuccessful result for {job_id}")
        called_tools = result.get("called_tools") or []
        if not called_tools or called_tools[0] != "inspect_board":
            raise NarratorReasoningError(f"Board inspection missing for {job_id}")
        paragraphs = result.get("paragraphs") or []
        if result["status"] == "ready" and not paragraphs:
            raise NarratorReasoningError(f"Ready result has no paragraphs for {job_id}")
        if result["status"] == "empty":
            if paragraphs:
                raise NarratorReasoningError(f"Empty result has paragraphs for {job_id}")
            empty_count += 1
        for paragraph in paragraphs:
            if not (
                job["source_start_s"] <= paragraph["start_s"] <= paragraph["end_s"]
                <= job["source_end_s"] <= job["window_end_s"]
            ):
                raise NarratorReasoningError(
                    f"Paragraph crosses transcript window for {job_id}"
                )
        paragraph_count += len(paragraphs)
        cost = (result.get("usage") or {}).get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total_cost += float(cost)

    return {
        "schema": "narrator-reasoning-verification-v1",
        "game_id": plan["game_id"],
        "model_id": plan["model_id"],
        "job_count": len(expected),
        "ready_count": len(expected) - empty_count,
        "empty_count": empty_count,
        "paragraph_count": paragraph_count,
        "total_cost": round(total_cost, 6),
    }
