"""Job factorial, preflight, admission, and provider contracts."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.board_bench.run.eval_catan_tile_prompt_ablation import (
    CONDITIONS,
    SCORER_VERSION,
    acquire_run_lock,
    admissible_record,
    build_jobs,
    build_plan,
    factorial_summary,
    release_run_lock,
    response_record,
    scorer_sha256,
    validate_request_contract,
)

from .support import load_fixture_questions


def test_builds_balanced_408_job_factorial(tmp_path: Path) -> None:
    _, _, questions = load_fixture_questions(tmp_path)
    jobs = build_jobs(questions, conditions=CONDITIONS, seed=38_271)

    assert len(questions) == 102
    assert len(jobs) == 408
    assert len({(job["condition"], job["qa"]["id"]) for job in jobs}) == 408
    assert Counter(job["condition"] for job in jobs) == {
        condition: 102 for condition in CONDITIONS
    }

    position_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for index, job in enumerate(jobs):
        position_counts[index % 4][job["condition"]] += 1
    for counts in position_counts.values():
        assert max(counts.values()) - min(counts.values()) <= 1

    with pytest.raises(ValueError, match="unique"):
        build_jobs(
            questions,
            conditions=("angle_labels", "angle_labels"),
            seed=1,
        )
    with pytest.raises(ValueError, match="positive"):
        build_jobs(
            questions,
            conditions=("angle_labels",),
            seed=1,
            max_requests=0,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        build_jobs(
            questions,
            conditions=CONDITIONS,
            seed=1,
            preflight=True,
            max_requests=1,
        )


def test_preflight_covers_conditions_and_both_output_grammars(tmp_path: Path) -> None:
    _, _, questions = load_fixture_questions(tmp_path)
    jobs = build_jobs(
        questions,
        conditions=CONDITIONS,
        seed=38_271,
        preflight=True,
    )

    assert len(jobs) == 4
    assert {job["condition"] for job in jobs} == set(CONDITIONS)
    assert sum(job["qa"]["truth"]["number"] is None for job in jobs) == 2
    assert sum(job["qa"]["truth"]["number"] is not None for job in jobs) == 2
    for wrapper in ("angle", "plain"):
        wrapper_jobs = [job for job in jobs if job["condition"].startswith(wrapper)]
        assert sum(job["qa"]["truth"]["number"] is None for job in wrapper_jobs) == 1
    for guidance in ("labels", "described"):
        guidance_jobs = [job for job in jobs if job["condition"].endswith(guidance)]
        assert sum(job["qa"]["truth"]["number"] is None for job in guidance_jobs) == 1
    assert factorial_summary(
        {
            (job["condition"], job["qa"]["id"]): {
                "score": {"pair_correct": True}
            }
            for job in jobs
        }
    ) == {"complete": False}


def test_plan_and_admission_fail_closed(tmp_path: Path) -> None:
    qa_path, image_root, questions = load_fixture_questions(tmp_path)
    jobs: Any = build_jobs(
        questions,
        conditions=CONDITIONS,
        seed=38_271,
        max_requests=1,
    )
    args: Any = SimpleNamespace(
        model="qwen/qwen3.8-27b",
        qa_path=qa_path,
        image_root=image_root,
        output_dir=tmp_path / "run",
        seed=38_271,
        preflight=False,
        temperature=0.0,
        max_tokens=32,
        timeout=180.0,
        concurrency=1,
    )
    plan: Any = build_plan(
        args,
        questions=questions,
        conditions=CONDITIONS,
        provider_order=("AkashML",),
        jobs=jobs,
    )
    job: Any = jobs[0]
    wrong_but_admissible = "<ORE> 2" if job["condition"].startswith("angle") else "ORE 2"
    result: Any = {
        "response": wrong_but_admissible,
        "served_model": "qwen/qwen3.8-27b",
        "provider": "AkashML",
        "finish_reason": "stop",
        "native_reasoning": {},
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        "submitted_image_sha256": job["qa"]["image_sha256"],
    }
    record = response_record(job, result, plan)

    assert plan["request_count"] == 1
    assert plan["scorer"] == {
        "version": SCORER_VERSION,
        "sha256": scorer_sha256(),
    }
    assert admissible_record(record, job, plan)

    corrupted: Any = dict(record, score={"pair_correct": True})
    assert not admissible_record(corrupted, job, plan)

    reasoning_result: Any = dict(
        result,
        native_reasoning={"reasoning": "hidden"},
    )
    reasoning_record: Any = response_record(job, reasoning_result, plan)
    assert "reasoning control violation" in reasoning_record["error"]
    assert not admissible_record(reasoning_record, job, plan)

    wrong_provider: Any = dict(result, provider="Chutes")
    provider_record: Any = response_record(job, wrong_provider, plan)
    assert "provider mismatch" in provider_record["error"]

    invalid_reasoning_values = (
        None,
        False,
        "0",
        0.5,
        "malformed",
        float("nan"),
    )
    invalid_usages: Any = [
        {},
        *[
            {"completion_tokens_details": {"reasoning_tokens": value}}
            for value in invalid_reasoning_values
        ],
    ]
    for invalid_usage in invalid_usages:
        invalid_result: Any = dict(result, usage=invalid_usage)
        invalid_record: Any = response_record(job, invalid_result, plan)
        assert "reasoning-token usage field is missing or invalid" in invalid_record[
            "error"
        ]
        assert not admissible_record(invalid_record, job, plan)


def test_provider_contract_and_output_lock(tmp_path: Path) -> None:
    validate_request_contract("qwen/qwen3.8-27b", ("AkashML",))
    with pytest.raises(SystemExit, match="Exactly one"):
        validate_request_contract(
            "qwen/qwen3.8-27b",
            ("AkashML", "Chutes"),
        )
    with pytest.raises(SystemExit, match="reasoning-disable"):
        validate_request_contract("unknown/model", ("AkashML",))
    with pytest.raises(SystemExit, match="reasoning-disable"):
        validate_request_contract("fake/qwen3.8-impostor", ("AkashML",))

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    lock_path, lock_fd = acquire_run_lock(output_dir)
    try:
        with pytest.raises(SystemExit, match="already locked"):
            acquire_run_lock(output_dir)
    finally:
        release_run_lock(lock_path, lock_fd)
    assert not lock_path.exists()
