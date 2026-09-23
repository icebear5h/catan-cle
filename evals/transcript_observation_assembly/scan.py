"""Replay scan that captures every assembly packet for a game."""

from __future__ import annotations

import contextlib
import hashlib
import io
from copy import deepcopy
from pathlib import Path

from cle.replay.contracts import optional_mapping_field
from evals.json_types import JsonDict, as_dict
from evals.transcript_observation_assembly.anchors import load_decision_anchors
from evals.transcript_observation_assembly.artifacts import utc_now
from evals.transcript_observation_assembly.job import (
    ObservationAssemblyError,
    ObservationAssemblyJob,
    ObservationScan,
)
from evals.transcript_observation_assembly.packets import build_packet_specs
from evals.transcript_observation_assembly.protocol import (
    DEFAULT_DECISION_ARTIFACT_DIR,
    PROMPT_VERSION,
    RUN_SCHEMA,
)
from evals.transcript_observation_assembly.specs import (
    _build_job_from_spec,
    _step_replay_without_lookahead,
)
from evals.transcript_reasoning.support import (
    int_field,
    json_value,
    path_field,
    require_replay_data,
)
from playground.game_viewer.app import app
from playground.game_viewer.replay.transcript import paired_transcript_fingerprint
from playground.game_viewer.state import server_state


def scan_observation_jobs(
    game_id: str,
    decision_artifact_dir: Path = DEFAULT_DECISION_ARTIFACT_DIR,
) -> ObservationScan:
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
        replay_data = require_replay_data(server_state)
        paired_transcript = optional_mapping_field(replay_data, "paired_transcript")
        if not paired_transcript:
            raise ObservationAssemblyError(f"Replay {game_id} has no transcript")
        total_events = int_field(replay_data.get("total_events", 0), "total_events")
        raw_narrator = paired_transcript.get("narrator") or {}
        narrator = deepcopy(as_dict(json_value(raw_narrator, "narrator"), "narrator"))
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
        jobs: list[ObservationAssemblyJob] = []
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
        replay_path = path_field(replay_data["file"], "replay file")
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


def build_generation_plan(scan: ObservationScan, model_id: str) -> JsonDict:
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
        "anchors": list(scan["anchors"]),
        "job_count": len(jobs),
        "jobs": [job.plan_record() for job in jobs],
    }

