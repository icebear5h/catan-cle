"""Cursor-safe loading for GPT-reconstructed narrator reasoning paragraphs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .transcript import paired_transcript_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_SCHEMA = "narrator-observation-assembly-run-v1"
RESULT_SCHEMA = "narrator-observation-assembly-result-v1"
WINDOW_SCHEMA = "paired-narrator-reasoning-v2"
PARAGRAPH_KINDS = frozenset(
    {
        "decision_reasoning",
        "board_observation",
        "opponent_assessment",
        "reaction",
        "reflection",
    }
)
ANCHOR_KINDS = frozenset({"decision", "observation", "complete"})

_CURATED_REASONING_RUNS = {
    "242781000": {
        "artifact_dir": PROJECT_ROOT
        / "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI"
        / "narrator_reasoning/gpt_5_6_sol_observation_v3_20260820",
        "model_id": "openai/gpt-5.6-sol",
        "model_label": "GPT-5.6",
        "generator_version": "narrator-observation-assembly-v1",
    }
}


class NarratorReasoningArtifactError(ValueError):
    """Raised when a persisted narrator-reasoning run violates its contract."""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise NarratorReasoningArtifactError(
            f"Missing narrator-reasoning artifact: {path.name}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NarratorReasoningArtifactError(
            f"Could not read narrator-reasoning artifact {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise NarratorReasoningArtifactError(
            f"Narrator-reasoning artifact {path.name} is not an object"
        )
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise NarratorReasoningArtifactError(
            f"Could not read narrator-reasoning artifact {path.name}: {exc}"
        ) from exc

    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NarratorReasoningArtifactError(
                f"Could not read {path.name}:{line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise NarratorReasoningArtifactError(
                f"{path.name}:{line_number} is not an object"
            )
        rows.append(row)
    return rows


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise NarratorReasoningArtifactError(
            f"Narrator-reasoning {label} mismatch: "
            f"expected {expected!r}, got {actual!r}"
        )


def _validate_plan_jobs(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    jobs = plan.get("jobs")
    if not isinstance(jobs, list):
        raise NarratorReasoningArtifactError("Narrator-reasoning plan jobs must be a list")
    if plan.get("job_count") != len(jobs):
        raise NarratorReasoningArtifactError(
            "Narrator-reasoning plan job_count does not match jobs"
        )

    indexed = {}
    replay_indices = set()
    for row_number, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning plan job {row_number} is not an object"
            )
        job_id = job.get("job_id")
        replay_index = job.get("replay_index")
        previous_index = job.get("previous_replay_index")
        input_hash = job.get("input_hash")
        anchor_kind = job.get("anchor_kind")
        decision_ids = job.get("decision_ids")
        evidence_ids = job.get("evidence_ids")
        if not isinstance(job_id, str) or not job_id:
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning plan job {row_number} has no job_id"
            )
        if job_id in indexed:
            raise NarratorReasoningArtifactError(f"Duplicate reasoning job_id {job_id}")
        if (
            not isinstance(replay_index, int)
            or isinstance(replay_index, bool)
            or replay_index < 0
            or replay_index in replay_indices
            or not isinstance(previous_index, int)
            or isinstance(previous_index, bool)
            or previous_index >= replay_index
        ):
            raise NarratorReasoningArtifactError(
                f"Invalid packet cursor for {job_id}"
            )
        if anchor_kind not in ANCHOR_KINDS:
            raise NarratorReasoningArtifactError(f"Invalid anchor kind for {job_id}")
        if not isinstance(decision_ids, list) or not all(
            isinstance(item, str) and item for item in decision_ids
        ):
            raise NarratorReasoningArtifactError(f"Invalid decision IDs for {job_id}")
        if anchor_kind == "decision" and not decision_ids:
            raise NarratorReasoningArtifactError(
                f"Decision packet {job_id} has no decision IDs"
            )
        if anchor_kind != "decision" and decision_ids:
            raise NarratorReasoningArtifactError(
                f"Non-decision packet {job_id} contains decision IDs"
            )
        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) != len(set(evidence_ids))
            or not all(isinstance(item, str) and item for item in evidence_ids)
            or job.get("utterance_count") != len(evidence_ids)
        ):
            raise NarratorReasoningArtifactError(f"Invalid evidence IDs for {job_id}")
        if not isinstance(input_hash, str) or len(input_hash) != 64:
            raise NarratorReasoningArtifactError(f"Invalid input hash for {job_id}")
        for field in (
            "window_start_s",
            "window_end_s",
            "source_start_s",
            "source_end_s",
        ):
            if not isinstance(job.get(field), (int, float)) or isinstance(
                job.get(field), bool
            ):
                raise NarratorReasoningArtifactError(
                    f"Invalid {field} for {job_id}"
                )
        if not (
            job["source_start_s"]
            <= job["source_end_s"]
            <= job["window_end_s"]
        ) or job["window_start_s"] > job["window_end_s"]:
            raise NarratorReasoningArtifactError(
                f"Invalid transcript availability or source bounds for {job_id}"
            )
        indexed[job_id] = job
        replay_indices.add(replay_index)
    return indexed


def _validate_plan_anchors(plan: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    anchors = plan.get("anchors")
    if not isinstance(anchors, list) or plan.get("anchor_count") != len(anchors):
        raise NarratorReasoningArtifactError(
            "Narrator-reasoning plan anchors are malformed"
        )
    indexed = {}
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise NarratorReasoningArtifactError("Reasoning anchor is not an object")
        replay_index = anchor.get("replay_index")
        anchor_kind = anchor.get("anchor_kind")
        decision_ids = anchor.get("decision_ids")
        if (
            not isinstance(replay_index, int)
            or isinstance(replay_index, bool)
            or replay_index < 0
            or replay_index in indexed
            or anchor_kind not in ANCHOR_KINDS
            or not isinstance(decision_ids, list)
            or not all(isinstance(item, str) and item for item in decision_ids)
        ):
            raise NarratorReasoningArtifactError("Invalid reasoning anchor")
        if (anchor_kind == "decision") != bool(decision_ids):
            raise NarratorReasoningArtifactError(
                "Reasoning anchor decision identity is inconsistent"
            )
        indexed[replay_index] = anchor
    return indexed


def _validate_paragraph(
    paragraph: Any,
    *,
    job_id: str,
    job: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(paragraph, dict):
        raise NarratorReasoningArtifactError(f"Paragraph for {job_id} is not an object")
    if any(
        not isinstance(paragraph.get(field), str) or not paragraph[field]
        for field in ("paragraph_id", "text")
    ) or paragraph.get("kind") not in PARAGRAPH_KINDS:
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} is missing identity, kind, or text"
        )
    if paragraph["kind"] == "decision_reasoning" and job["anchor_kind"] != "decision":
        raise NarratorReasoningArtifactError(
            f"Observation paragraph for {job_id} is mislabeled as decision reasoning"
        )
    evidence_ids = paragraph.get("evidence_ids")
    uncertainties = paragraph.get("uncertainties")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or len(evidence_ids) != len(set(evidence_ids))
        or not all(isinstance(item, str) and item for item in evidence_ids)
        or not set(evidence_ids).issubset(job["evidence_ids"])
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} has invalid evidence IDs"
        )
    if not isinstance(uncertainties, list) or not all(
        isinstance(item, str) and item for item in uncertainties
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} has invalid uncertainties"
        )
    for field in ("start_s", "end_s"):
        if not isinstance(paragraph.get(field), (int, float)) or isinstance(
            paragraph.get(field), bool
        ):
            raise NarratorReasoningArtifactError(
                f"Paragraph for {job_id} has invalid {field}"
            )
    if not (
        job["source_start_s"]
        <= paragraph["start_s"]
        <= paragraph["end_s"]
        <= job["source_end_s"]
        <= job["window_end_s"]
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} crosses its transcript window"
        )
    if not (
        job["source_start_index"]
        <= paragraph.get("source_start_index", -1)
        <= paragraph.get("source_end_index", -1)
        <= job["source_end_index"]
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} crosses its source-caption bounds"
        )
    if (
        paragraph.get("subject_replay_index") != job["replay_index"]
        or paragraph.get("available_replay_index") != job["replay_index"]
        or paragraph.get("anchor_kind") != job["anchor_kind"]
        or paragraph.get("decision_ids") != job["decision_ids"]
    ):
        raise NarratorReasoningArtifactError(
            f"Paragraph for {job_id} violates strict causal attachment"
        )
    return paragraph


def load_narrator_reasoning_artifact(
    artifact_dir: Path,
    *,
    expected_game_id: str,
    expected_model_id: str,
    expected_model_label: str,
    expected_generator_version: str,
    paired_transcript: Dict[str, Any],
) -> Dict[str, Any]:
    """Load and validate one persisted GPT narrator-reasoning run."""
    artifact_dir = Path(artifact_dir)
    plan = _read_json(artifact_dir / "plan.json")
    results = _read_jsonl(artifact_dir / "results.jsonl")
    _require_equal("plan schema", plan.get("schema"), RUN_SCHEMA)
    _require_equal("game", str(plan.get("game_id")), str(expected_game_id))
    _require_equal("model", plan.get("model_id"), expected_model_id)
    _require_equal(
        "generator version",
        plan.get("generator_version"),
        expected_generator_version,
    )
    _require_equal(
        "transcript fingerprint",
        plan.get("transcript_sha256"),
        paired_transcript_fingerprint(paired_transcript),
    )
    plan_jobs = _validate_plan_jobs(plan)
    anchors_by_replay_index = _validate_plan_anchors(plan)

    results_by_replay_index = {}
    loaded_job_ids = set()
    ready_count = 0
    empty_count = 0
    error_count = 0
    paragraph_count = 0
    for row_number, result in enumerate(results, start=1):
        job_id = result.get("job_id")
        if not isinstance(job_id, str) or job_id not in plan_jobs:
            raise NarratorReasoningArtifactError(
                f"results.jsonl:{row_number} references an unknown job"
            )
        if job_id in loaded_job_ids:
            raise NarratorReasoningArtifactError(
                f"Duplicate narrator-reasoning result for {job_id}"
            )
        loaded_job_ids.add(job_id)
        job = plan_jobs[job_id]
        _require_equal(f"result schema for {job_id}", result.get("schema"), RESULT_SCHEMA)
        _require_equal(f"game for {job_id}", str(result.get("game_id")), expected_game_id)
        _require_equal(f"model for {job_id}", result.get("model_id"), expected_model_id)
        _require_equal(
            f"generator version for {job_id}",
            result.get("generator_version"),
            expected_generator_version,
        )
        _require_equal(
            f"replay index for {job_id}",
            result.get("replay_index"),
            job.get("replay_index"),
        )
        _require_equal(
            f"input hash for {job_id}",
            result.get("input_hash"),
            job.get("input_hash"),
        )
        for field in (
            "previous_replay_index",
            "anchor_kind",
            "decision_ids",
        ):
            _require_equal(
                f"{field} for {job_id}",
                result.get(field),
                job.get(field),
            )
        _require_equal(
            f"board state for {job_id}",
            result.get("board_state_hash"),
            job.get("board_state_hash"),
        )
        _require_equal(
            f"window start for {job_id}",
            result.get("window_start_s"),
            job.get("window_start_s"),
        )
        _require_equal(
            f"window end for {job_id}",
            result.get("window_end_s"),
            job.get("window_end_s"),
        )

        status = result.get("status")
        if status not in {"ready", "empty", "error"}:
            raise NarratorReasoningArtifactError(
                f"Invalid narrator-reasoning status for {job_id}"
            )
        paragraphs = result.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning paragraphs for {job_id} must be a list"
            )
        validated_paragraphs = [
            _validate_paragraph(paragraph, job_id=job_id, job=job)
            for paragraph in paragraphs
        ]
        omitted_evidence_ids = result.get("omitted_evidence_ids")
        if not isinstance(omitted_evidence_ids, list) or not all(
            isinstance(item, str) and item for item in omitted_evidence_ids
        ):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning omissions for {job_id} must be evidence IDs"
            )
        cited_evidence_ids = [
            evidence_id
            for paragraph in validated_paragraphs
            for evidence_id in paragraph["evidence_ids"]
        ]
        if (
            len(cited_evidence_ids) != len(set(cited_evidence_ids))
            or len(omitted_evidence_ids) != len(set(omitted_evidence_ids))
            or set(cited_evidence_ids).intersection(omitted_evidence_ids)
            or set(cited_evidence_ids).union(omitted_evidence_ids)
            != set(job["evidence_ids"])
        ):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning evidence partition is invalid for {job_id}"
            )
        if result.get("evidence_count") != len(job["evidence_ids"]):
            raise NarratorReasoningArtifactError(
                f"Narrator-reasoning evidence count is invalid for {job_id}"
            )
        if status == "ready":
            called_tools = result.get("called_tools")
            if not called_tools or called_tools[0] != "inspect_board":
                raise NarratorReasoningArtifactError(
                    f"Ready result for {job_id} lacks board inspection"
                )
            if not validated_paragraphs:
                raise NarratorReasoningArtifactError(
                    f"Ready result for {job_id} has no paragraphs"
                )
            ready_count += 1
        elif status == "empty":
            if validated_paragraphs:
                raise NarratorReasoningArtifactError(
                    f"Empty result for {job_id} contains paragraphs"
                )
            called_tools = result.get("called_tools")
            if not called_tools or called_tools[0] != "inspect_board":
                raise NarratorReasoningArtifactError(
                    f"Empty result for {job_id} lacks board inspection"
                )
            empty_count += 1
        else:
            if validated_paragraphs:
                raise NarratorReasoningArtifactError(
                    f"Error result for {job_id} contains paragraphs"
                )
            error_count += 1
        paragraph_count += len(validated_paragraphs)
        results_by_replay_index[result["replay_index"]] = {
            "status": status,
            "recorded_at": result.get("recorded_at"),
            "error": result.get("error"),
            "anchor_kind": result.get("anchor_kind"),
            "decision_ids": list(result.get("decision_ids") or []),
            "paragraphs": validated_paragraphs,
        }

    expected_indices = {
        job["replay_index"]: job_id for job_id, job in plan_jobs.items()
    }
    return {
        "schema": WINDOW_SCHEMA,
        "artifact_schema": RUN_SCHEMA,
        "game_id": expected_game_id,
        "model_id": expected_model_id,
        "model_label": expected_model_label,
        "generator_version": expected_generator_version,
        "transcript_sha256": plan["transcript_sha256"],
        "narrator": plan.get("narrator") or {},
        "artifact_error": None,
        "complete": len(loaded_job_ids) == len(plan_jobs) and error_count == 0,
        "expected_window_count": len(plan_jobs),
        "loaded_window_count": len(loaded_job_ids),
        "ready_window_count": ready_count,
        "empty_window_count": empty_count,
        "error_window_count": error_count,
        "paragraph_count": paragraph_count,
        "decision_count": plan.get("decision_count", 0),
        "anchor_count": plan.get("anchor_count", 0),
        "anchors_by_replay_index": anchors_by_replay_index,
        "expected_job_id_by_replay_index": expected_indices,
        "results_by_replay_index": results_by_replay_index,
    }


def _artifact_error_collection(
    game_id: str,
    run: Dict[str, Any],
    paired_transcript: Dict[str, Any],
    error: Exception,
) -> Dict[str, Any]:
    return {
        "schema": WINDOW_SCHEMA,
        "artifact_schema": RUN_SCHEMA,
        "game_id": game_id,
        "model_id": run["model_id"],
        "model_label": run["model_label"],
        "generator_version": run["generator_version"],
        "transcript_sha256": paired_transcript_fingerprint(paired_transcript),
        "narrator": paired_transcript.get("narrator") or {},
        "artifact_error": str(error),
        "complete": False,
        "expected_window_count": 0,
        "loaded_window_count": 0,
        "ready_window_count": 0,
        "empty_window_count": 0,
        "error_window_count": 0,
        "paragraph_count": 0,
        "decision_count": 0,
        "anchor_count": 0,
        "anchors_by_replay_index": {},
        "expected_job_id_by_replay_index": {},
        "results_by_replay_index": {},
    }


def load_paired_narrator_reasoning(
    game_id: str,
    paired_transcript: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Load optional narrator reasoning, failing soft so replay loading still works."""
    run = _CURATED_REASONING_RUNS.get(str(game_id))
    if run is None or paired_transcript is None:
        return None
    try:
        return load_narrator_reasoning_artifact(
            run["artifact_dir"],
            expected_game_id=str(game_id),
            expected_model_id=run["model_id"],
            expected_model_label=run["model_label"],
            expected_generator_version=run["generator_version"],
            paired_transcript=paired_transcript,
        )
    except NarratorReasoningArtifactError as exc:
        return _artifact_error_collection(str(game_id), run, paired_transcript, exc)
    except (AttributeError, KeyError, TypeError) as exc:
        schema_error = NarratorReasoningArtifactError(
            f"Malformed narrator-reasoning artifact schema: {exc}"
        )
        return _artifact_error_collection(
            str(game_id), run, paired_transcript, schema_error
        )


def build_paired_narrator_reasoning_window(
    replay_data: Dict[str, Any], replay_index: int
) -> Optional[Dict[str, Any]]:
    """Expose subject-grouped assembly only at its strict causal cursor."""
    collection = replay_data.get("paired_narrator_reasoning")
    if not collection:
        return None

    anchors = collection.get("anchors_by_replay_index", {})
    current_anchor = anchors.get(replay_index) or {}
    base = {
        "schema": collection.get("schema", WINDOW_SCHEMA),
        "game_id": collection.get("game_id"),
        "replay_index": replay_index,
        "model_id": collection.get("model_id"),
        "model_label": collection.get("model_label"),
        "generator_version": collection.get("generator_version"),
        "narrator": collection.get("narrator") or {},
        "artifact_error": collection.get("artifact_error"),
        "complete": bool(collection.get("complete", False)),
        "strict_causal": True,
        "decision_count": collection.get("decision_count", 0),
        "anchor_count": collection.get("anchor_count", 0),
        "expected_window_count": collection.get("expected_window_count", 0),
        "loaded_window_count": collection.get("loaded_window_count", 0),
        "anchor_kind": current_anchor.get("anchor_kind"),
        "decision_ids": list(current_anchor.get("decision_ids") or []),
    }
    empty_payload = {
        "error": None,
        "paragraphs": [],
        "history_paragraphs": [],
        "history_groups": [],
    }
    if collection.get("artifact_error"):
        return {**base, **empty_payload, "status": "artifact_error"}
    if replay_index < 0:
        return {**base, **empty_payload, "status": "unavailable"}

    results_by_replay_index = collection.get("results_by_replay_index", {})
    available_results = [
        (available_index, result)
        for available_index, result in sorted(results_by_replay_index.items())
        if available_index <= replay_index
    ]
    history_groups = [
        {
            "subject_replay_index": available_index,
            "available_replay_index": available_index,
            "anchor_kind": result.get("anchor_kind"),
            "decision_ids": list(result.get("decision_ids") or []),
            "paragraphs": list(result.get("paragraphs") or []),
        }
        for available_index, result in available_results
        if result.get("paragraphs")
    ]
    history_paragraphs = [
        paragraph
        for group in history_groups
        for paragraph in group["paragraphs"]
    ]
    history_paragraphs.sort(
        key=lambda paragraph: (
            paragraph["available_replay_index"],
            paragraph["start_s"],
            paragraph["end_s"],
            paragraph["paragraph_id"],
        )
    )
    history_payload = {
        "history_paragraphs": history_paragraphs,
        "history_groups": history_groups,
    }
    result = results_by_replay_index.get(replay_index)
    if result is not None:
        return {
            **base,
            **history_payload,
            "status": result["status"],
            "recorded_at": result.get("recorded_at"),
            "error": result.get("error"),
            "anchor_kind": result.get("anchor_kind"),
            "decision_ids": list(result.get("decision_ids") or []),
            "paragraphs": list(result.get("paragraphs") or []),
        }

    expected = collection.get("expected_job_id_by_replay_index", {})
    if replay_index in expected:
        return {
            **base,
            **history_payload,
            "status": "pending",
            "error": None,
            "paragraphs": [],
        }

    total_events = replay_data.get("total_events", 0)
    status = (
        "complete"
        if isinstance(total_events, int) and replay_index >= total_events
        else "no_commentary"
    )
    return {
        **base,
        **history_payload,
        "status": status,
        "error": None,
        "paragraphs": [],
    }
