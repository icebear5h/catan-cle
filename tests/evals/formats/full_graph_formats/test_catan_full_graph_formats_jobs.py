"""Probe construction, job building, resume, and output locks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals.catan_board_bench.ascii_variations import full_fact_digest
from evals.catan_board_bench.full_graph_format_probe import (
    build_full_graph_format_probe,
)
from evals.catan_board_bench.full_graph_formats import (
    FORMAT_EXTENSIONS,
    FORMAT_NAMES,
    parse_full_graph_format,
)
from scripts.board_bench.run.eval_catan_board_bench_full_graph_formats import (
    accepted_records,
    acquire_run_lock,
    admissible_record,
    build_jobs,
    build_plan,
    release_run_lock,
    response_record,
    validate_dataset_integrity,
    validate_request_contract,
)

from .support import SOURCE_DIR


def test_format_probe_preserves_source_questions_and_baseline(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "format_probe"
    metadata = build_full_graph_format_probe(
        output_dir,
        source_dir=SOURCE_DIR,
    )

    assert metadata["request_count"] == 360
    assert metadata["formats"] == list(FORMAT_NAMES)
    assert metadata["incident_list_format"] is False
    assert metadata["integrated_ascii_dynamic_state_only_in_diagram"] is True
    assert (output_dir / "qa.jsonl").read_bytes() == (SOURCE_DIR / "qa.jsonl").read_bytes()

    representation_files = list((output_dir / "representations").glob("*/*"))
    assert len(representation_files) == 72
    for sample_dir in sorted((output_dir / "representations").iterdir()):
        baseline = sample_dir / "tile_rows.txt"
        source_baseline = SOURCE_DIR / "representations" / sample_dir.name / "tile_rows.txt"
        assert baseline.read_bytes() == source_baseline.read_bytes()
        facts = json.loads((output_dir / "facts" / f"{sample_dir.name}.json").read_text())
        for format_name in FORMAT_NAMES:
            path = sample_dir / f"{format_name}{FORMAT_EXTENSIONS[format_name]}"
            parsed = parse_full_graph_format(
                format_name,
                path.read_text().rstrip("\n"),
            )
            assert full_fact_digest(parsed) == full_fact_digest(facts)


def test_format_evaluator_builds_360_leak_free_jobs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "format_probe"
    metadata: Any = build_full_graph_format_probe(
        dataset_dir,
        source_dir=SOURCE_DIR,
    )
    questions: Any = validate_dataset_integrity(dataset_dir, metadata)
    jobs: Any = build_jobs(
        dataset_dir,
        questions=questions,
        formats=FORMAT_NAMES,
        max_requests=None,
    )

    assert len(jobs) == 360
    assert len({(job["format"], job["qa"]["id"]) for job in jobs}) == 360
    assert all(job["qa"]["answer_text"] not in job["prompt"] for job in jobs)
    assert all("Required JSON shape:" in job["prompt"] for job in jobs)
    assert all(job["representation_sha256"] for job in jobs)
    assert all(job["board_presentation"].kind == "text" for job in jobs)
    assert all(
        job["board_presentation"].provenance.identity_space
        == "opaque_board_local_ids"
        for job in jobs
    )
    assert all(
        job["board_presentation"].content in job["prompt"]
        for job in jobs
    )
    with pytest.raises(ValueError, match="duplicate formats"):
        build_jobs(
            dataset_dir,
            questions=questions,
            formats=("tile_rows", "tile_rows"),
            max_requests=None,
        )
    with pytest.raises(ValueError, match="must be positive"):
        build_jobs(
            dataset_dir,
            questions=questions,
            formats=("tile_rows",),
            max_requests=0,
        )

    args: Any = SimpleNamespace(
        dataset_dir=dataset_dir,
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path / "eval",
        concurrency=2,
        temperature=0.0,
        max_tokens=256,
        timeout=180.0,
    )
    plan: Any = build_plan(
        args,
        dataset_metadata=metadata,
        questions=questions,
        formats=FORMAT_NAMES,
        jobs=jobs,
        provider_order=("AkashML",),
    )
    assert plan["request_count"] == 360
    assert plan["request_settings"]["reasoning_disabled"] is True
    assert plan["request_settings"]["allow_fallbacks"] is False


def test_resume_recomputes_scores_and_keeps_earlier_valid_attempt() -> None:
    qa: Any = {
        "id": "q1",
        "sample_id": "board1",
        "category": "direction_to_tile",
        "question": "Which tile?",
        "answer": {"tile": "T01"},
        "answer_text": '{"tile":"T01"}',
        "fact_digest": "facts",
    }
    job: Any = {
        "format": "full_graph_json",
        "qa": qa,
        "prompt": "authoritative prompt",
        "representation_path": "board.json",
        "representation_sha256": "representation",
    }
    plan: Any = {
        "model": "qwen/qwen3.8-27b",
        "scorer": {"version": "v", "sha256": "score"},
        "request_settings": {
            "provider_order": ["AkashML"],
            "reasoning_disabled": True,
        },
    }
    result: Any = {
        "served_model": plan["model"],
        "provider": "AkashML",
        "response": qa["answer_text"],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        "native_reasoning": None,
    }
    valid: Any = response_record(job, result, plan)
    valid["score"] = {"correct": False, "corrupted": True}
    later_valid: Any = response_record(
        job,
        dict(result, response='{"tile":"T99"}'),
        plan,
    )
    later_invalid: Any = dict(valid, error="later transport failure")
    accepted: Any = accepted_records(
        [valid, later_valid, later_invalid],
        jobs=(job,),
        plan=plan,
    )

    assert admissible_record(valid, job=job, plan=plan)
    assert accepted[(job["format"], qa["id"])]["score"]["correct"]

    wrong_provider: Any = response_record(
        job,
        dict(result, provider="Chutes"),
        plan,
    )
    assert "unexpected provider" in wrong_provider["error"]
    hidden_reasoning: Any = response_record(
        job,
        dict(result, native_reasoning={"reasoning_content": "hidden"}),
        plan,
    )
    assert "reasoning control violation" in hidden_reasoning["error"]


def test_integrity_request_contract_and_output_lock_fail_closed(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "format_probe"
    metadata = build_full_graph_format_probe(
        dataset_dir,
        source_dir=SOURCE_DIR,
    )
    validate_dataset_integrity(dataset_dir, metadata)

    representation = dataset_dir / "representations" / "ascii_board_00" / "full_graph_json.json"
    original_representation = representation.read_bytes()
    representation.write_bytes(original_representation + b" ")
    with pytest.raises(SystemExit, match="Representation hash mismatch"):
        validate_dataset_integrity(dataset_dir, metadata)
    representation.write_bytes(original_representation)

    qa_path = dataset_dir / "qa.jsonl"
    original_qa = qa_path.read_bytes()
    qa_path.write_bytes(original_qa + b"\n")
    with pytest.raises(SystemExit, match="Generated locked file changed"):
        validate_dataset_integrity(dataset_dir, metadata)
    qa_path.write_bytes(original_qa)

    validate_request_contract("qwen/qwen3.8-27b", ("AkashML",))
    with pytest.raises(SystemExit, match="Exactly one"):
        validate_request_contract(
            "qwen/qwen3.8-27b",
            ("AkashML", "Chutes"),
        )
    with pytest.raises(SystemExit, match="reasoning-disable"):
        validate_request_contract("unknown/model", ("AkashML",))

    output_dir = tmp_path / "eval"
    output_dir.mkdir()
    lock_path, lock_fd = acquire_run_lock(output_dir)
    try:
        with pytest.raises(SystemExit, match="already locked"):
            acquire_run_lock(output_dir)
    finally:
        release_run_lock(lock_path, lock_fd)
    assert not lock_path.exists()
