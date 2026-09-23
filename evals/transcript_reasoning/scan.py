"""Replay scan that captures every nonempty transcript interval."""

from __future__ import annotations

import contextlib
import hashlib
import io
from copy import deepcopy

from cle.replay.contracts import optional_mapping_field
from evals.json_types import JsonDict, as_dict
from evals.transcript_reasoning.grounding import (
    _step_replay_without_lookahead,
    build_reasoning_job,
)
from evals.transcript_reasoning.job import (
    NarratorReasoningError,
    ReasoningJob,
    ReasoningScan,
)
from evals.transcript_reasoning.protocol import PROMPT_VERSION, RUN_SCHEMA
from evals.transcript_reasoning.support import (
    int_field,
    json_value,
    path_field,
    require_replay_data,
    utc_now,
)
from playground.game_viewer.app import app
from playground.game_viewer.replay.transcript import paired_transcript_fingerprint
from playground.game_viewer.state import server_state


def scan_reasoning_jobs(game_id: str) -> ReasoningScan:
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

        replay_data = require_replay_data(server_state)
        paired_transcript = optional_mapping_field(replay_data, "paired_transcript")
        if not paired_transcript:
            raise NarratorReasoningError(f"Replay {game_id} has no paired transcript")
        jobs: list[ReasoningJob] = []
        total_events = int_field(replay_data.get("total_events", 0), "total_events")
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
        narrator = paired_transcript.get("narrator") or {}
        replay_path = path_field(replay_data["file"], "replay file")
        return {
            "game_id": str(game_id),
            "total_events": total_events,
            "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
            "transcript_sha256": paired_transcript_fingerprint(paired_transcript),
            "narrator": deepcopy(as_dict(json_value(narrator, "narrator"), "narrator")),
            "jobs": jobs,
        }
    finally:
        server_state.reset()


def build_generation_plan(scan: ReasoningScan, model_id: str) -> JsonDict:
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

