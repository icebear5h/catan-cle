from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.eval_catan_tile_prompt_ablation as tile_ablation
from scripts.eval_catan_tile_prompt_ablation import (
    CONDITIONS,
    DEFAULT_QA_PATH,
    SCORER_VERSION,
    acquire_run_lock,
    admissible_record,
    build_jobs,
    build_plan,
    build_prompt,
    expected_response,
    extract_message_text,
    factorial_summary,
    load_questions,
    release_run_lock,
    response_record,
    score_response,
    scorer_sha256,
    validate_request_contract,
)


def materialize_images(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "isolated_visuals"
    rows = [
        json.loads(line)
        for line in DEFAULT_QA_PATH.read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["category"] != "isolated_tile_resource_number":
            continue
        image_path = image_root / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image:{row['sample_id']}".encode())
    qa_path = tmp_path / "qa.jsonl"
    qa_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return qa_path, image_root


def load_fixture_questions(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    qa_path, image_root = materialize_images(tmp_path)
    return qa_path, image_root, load_questions(qa_path, image_root)


def test_prompts_cross_wrapper_and_description_factors() -> None:
    angle_labels = build_prompt("angle_labels")
    plain_labels = build_prompt("plain_labels")
    angle_described = build_prompt("angle_described")
    plain_described = build_prompt("plain_described")

    for resource in ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "DESERT"):
        assert f"<{resource}>" in angle_labels
        assert f"<{resource}>" in angle_described
        assert f"<{resource}>" not in plain_labels
        assert f"<{resource}>" not in plain_described
        assert resource in plain_labels
        assert resource in plain_described

    assert "Visual guide" not in angle_labels
    assert "Visual guide" not in plain_labels
    assert "evergreen tree" in angle_described
    assert "stacked brick blocks" in plain_described
    assert "NO_NUMBER" not in angle_described
    assert "FOREST" not in plain_described

    assert angle_labels.replace("<", "").replace(">", "") == plain_labels
    assert (
        angle_described.replace("<", "").replace(">", "")
        == plain_described
    )


def test_strict_condition_scoring_rejects_aliases_extras_and_wrong_wrappers() -> None:
    wood = {"resource": "WOOD", "number": 2}
    desert = {"resource": "DESERT", "number": None}

    exact_angle = score_response(wood, "<WOOD> 2", "angle_labels")
    assert exact_angle["pair_correct"] is True
    assert exact_angle["protocol_exact"] is True
    assert exact_angle["wrapper_rescued_content_correct"] is False

    wrong_wrapper = score_response(wood, "WOOD 2", "angle_labels")
    assert wrong_wrapper["protocol_valid"] is False
    assert wrong_wrapper["pair_correct"] is False
    assert wrong_wrapper["wrapper_rescued_content_correct"] is True

    for invalid in (
        "<FOREST> 2",
        "<WOOD> 2.",
        "Answer: <WOOD> 2",
        "```<WOOD> 2```",
        " <WOOD> 2",
        "<WOOD> 2\n",
    ):
        assert score_response(wood, invalid, "angle_labels")[
            "protocol_valid"
        ] is False

    assert extract_message_text({"content": " <WOOD> 2\n"}) == " <WOOD> 2\n"

    wrong_resource = score_response(wood, "<ORE> 2", "angle_labels")
    assert wrong_resource["protocol_valid"] is True
    assert wrong_resource["number_correct"] is True
    assert wrong_resource["resource_correct"] is False
    assert wrong_resource["pair_correct"] is False

    assert score_response(desert, "<DESERT>", "angle_labels")["pair_correct"]
    assert score_response(desert, "DESERT", "plain_labels")["pair_correct"]
    assert not score_response(desert, "<DESERT> 2", "angle_labels")[
        "protocol_valid"
    ]
    assert expected_response(wood, "plain_described") == "WOOD 2"
    assert expected_response(desert, "angle_described") == "<DESERT>"


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
    jobs = build_jobs(
        questions,
        conditions=CONDITIONS,
        seed=38_271,
        max_requests=1,
    )
    args = SimpleNamespace(
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
    plan = build_plan(
        args,
        questions=questions,
        conditions=CONDITIONS,
        provider_order=("AkashML",),
        jobs=jobs,
    )
    job = jobs[0]
    wrong_but_admissible = "<ORE> 2" if job["condition"].startswith("angle") else "ORE 2"
    result = {
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

    corrupted = dict(record, score={"pair_correct": True})
    assert not admissible_record(corrupted, job, plan)

    reasoning_result = dict(
        result,
        native_reasoning={"reasoning": "hidden"},
    )
    reasoning_record = response_record(job, reasoning_result, plan)
    assert "reasoning control violation" in reasoning_record["error"]
    assert not admissible_record(reasoning_record, job, plan)

    wrong_provider = dict(result, provider="Chutes")
    provider_record = response_record(job, wrong_provider, plan)
    assert "provider mismatch" in provider_record["error"]

    invalid_reasoning_values = (
        None,
        False,
        "0",
        0.5,
        "malformed",
        float("nan"),
    )
    invalid_usages = [
        {},
        *[
            {"completion_tokens_details": {"reasoning_tokens": value}}
            for value in invalid_reasoning_values
        ],
    ]
    for invalid_usage in invalid_usages:
        invalid_result = dict(result, usage=invalid_usage)
        invalid_record = response_record(job, invalid_result, plan)
        assert "reasoning-token usage field is missing or invalid" in invalid_record[
            "error"
        ]
        assert not admissible_record(invalid_record, job, plan)


def test_scorer_hash_covers_condition_sets_and_regex_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = scorer_sha256()
    monkeypatch.setattr(
        tile_ablation,
        "ANGLE_CONDITIONS",
        frozenset({"angle_labels"}),
    )
    assert scorer_sha256() != baseline

    monkeypatch.undo()
    monkeypatch.setattr(
        tile_ablation,
        "ANGLE_NUMBER_RE",
        tile_ablation.re.compile(
            tile_ablation.ANGLE_NUMBER_RE.pattern,
            tile_ablation.re.IGNORECASE,
        ),
    )
    assert scorer_sha256() != baseline


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
