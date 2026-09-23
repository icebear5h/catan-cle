from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any

import pytest

from evals.catan_board_bench.ascii_variations import (
    ASCII_VARIANTS,
    build_ascii_variation_dataset,
    full_fact_digest,
    full_public_graph_facts,
    parse_ascii_variant,
    render_ascii_variant,
    score_strict_json_answer,
)
from scripts.board_bench.run.eval_catan_board_bench_ascii_variations import (
    build_jobs,
    build_plan,
    call_openrouter_text,
    response_record,
    successful_record,
)

CONTRACT_DIR = Path("evals/catan_board_bench/datasets/catan_board_bench_100/contracts")


def test_full_graph_variants_round_trip_one_information_set() -> None:
    contract = json.loads((CONTRACT_DIR / "sample_007.json").read_text())
    facts, aliases = full_public_graph_facts(contract, sample_id="roundtrip")

    assert len(facts["tiles"]) == 19
    assert len(facts["nodes"]) == 54
    assert len(facts["edges"]) == 72
    assert len(facts["ports"]) == 9
    assert "players" not in facts
    assert "settlement_count" not in json.dumps(facts)
    assert any(
        original_id != alias.removeprefix("T") for original_id, alias in aliases["tiles"].items()
    )

    expected_digest = full_fact_digest(facts)
    for variant in ASCII_VARIANTS:
        rendered = render_ascii_variant(variant, facts, sample_id="roundtrip")
        assert full_fact_digest(parse_ascii_variant(rendered)) == expected_digest


def test_ascii_smoke_dataset_is_balanced_and_game_diverse(tmp_path: Path) -> None:
    metadata = build_ascii_variation_dataset(
        tmp_path,
        contract_dir=CONTRACT_DIR,
    )
    questions = [json.loads(line) for line in (tmp_path / "qa.jsonl").read_text().splitlines()]
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]

    assert metadata["request_count"] == 360
    assert len(questions) == 60
    assert len({row["id"] for row in questions}) == 60
    assert set(Counter(row["category"] for row in questions).values()) == {6}
    assert len({row["source_game_id"] for row in manifest}) == 12
    for category in ("node_state", "edge_state", "port_occupancy"):
        values = {row["target"]["occupied"] for row in questions if row["category"] == category}
        assert values == {True, False}
    production_values = {
        row["target"]["has_payouts"] for row in questions if row["category"] == "roll_production"
    }
    assert production_values == {True, False}

    for row in manifest:
        sample_id = row["sample_id"]
        expected_digest = row["fact_digest"]
        for variant in ASCII_VARIANTS:
            rendered = (tmp_path / "representations" / sample_id / f"{variant}.txt").read_text()
            assert full_fact_digest(parse_ascii_variant(rendered)) == expected_digest


def test_ascii_evaluator_builds_one_leak_free_job_per_pair(tmp_path: Path) -> None:
    build_ascii_variation_dataset(tmp_path, contract_dir=CONTRACT_DIR)
    questions = [json.loads(line) for line in (tmp_path / "qa.jsonl").read_text().splitlines()]
    jobs: Any = build_jobs(
        tmp_path,
        questions=questions,
        variants=ASCII_VARIANTS,
        max_requests=None,
    )

    assert len(jobs) == 360
    assert len({(job["variant"], job["qa"]["id"]) for job in jobs}) == 360
    assert all(job["qa"]["answer_text"] not in job["prompt"] for job in jobs)
    assert all("Required JSON shape:" in job["prompt"] for job in jobs)


def test_strict_json_scorer_rejects_extras_and_protocol_noise() -> None:
    expected = {
        "occupants": [
            {
                "node": "N44",
                "color": "<BLUE>",
                "building": "<SETTLEMENT>",
            }
        ]
    }
    exact = '{"occupants":[{"node":"N44","color":"<BLUE>","building":"<SETTLEMENT>"}]}'
    extra = (
        '{"occupants":[{"node":"N40","color":"<BLUE>",'
        '"building":"<SETTLEMENT>"},{"node":"N44","color":"<BLUE>",'
        '"building":"<SETTLEMENT>"}]}'
    )
    wrong_tuple = '{"occupants":[{"node":"N44","color":"<RED>","building":"<SETTLEMENT>"}]}'

    assert score_strict_json_answer(expected, exact)["correct"]
    assert not score_strict_json_answer(expected, extra)["correct"]
    assert not score_strict_json_answer(expected, wrong_tuple)["correct"]
    duplicate_key = '{"occupants":[],"occupants":' + exact.split(":", 1)[1]

    assert not score_strict_json_answer(expected, f"Answer: {exact}")["json_valid"]
    assert not score_strict_json_answer(expected, duplicate_key)["json_valid"]
    assert not score_strict_json_answer(expected, "1")["correct"]


def test_openrouter_result_fails_closed_on_model_and_reasoning_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "id": "request-1",
        "provider": "AkashML",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": '{"tile":"T01"}',
                    "reasoning_content": "hidden trace",
                    "reasoning_details": [{"type": "reasoning.text"}],
                },
            }
        ],
        "usage": {
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return payload

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> bool | None:
            return False

        def post(self, url: str, *, headers: dict[str, str], json: object) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(
        "scripts.board_bench.run.eval_catan_board_bench_ascii_variations.httpx.Client",
        FakeClient,
    )
    result = call_openrouter_text(
        "key",
        "qwen/qwen3.8-27b",
        prompt="prompt",
        temperature=0.0,
        max_tokens=256,
        timeout=180.0,
        provider_order=("AkashML",),
    )

    assert result["served_model"] is None
    assert result["native_reasoning"] == {
        "reasoning_content": "hidden trace",
        "reasoning_details": [{"type": "reasoning.text"}],
    }


def test_resume_rejects_stale_prompt_provider_and_reasoning() -> None:
    qa = {
        "id": "q1",
        "sample_id": "board1",
        "category": "direction_to_tile",
        "question": "Which tile?",
        "answer": {"tile": "T01"},
        "answer_text": '{"tile":"T01"}',
        "fact_digest": "facts",
    }
    job: Any = {
        "variant": "flat_sorted",
        "qa": qa,
        "prompt": "authoritative prompt",
        "representation_path": "board.txt",
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
        "requested_model": plan["model"],
        "served_model": plan["model"],
        "provider": "AkashML",
        "response": qa["answer_text"],
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        "native_reasoning": None,
    }
    record: Any = response_record(job, result, plan)

    assert record["error"] is None
    assert successful_record(record, job=job, plan=plan)

    stale_prompt: Any = dict(record, prompt_sha256="wrong")
    wrong_provider: Any = dict(record, provider="Chutes")
    hidden_reasoning: Any = dict(record, native_reasoning="private")
    assert not successful_record(stale_prompt, job=job, plan=plan)
    assert not successful_record(wrong_provider, job=job, plan=plan)
    assert not successful_record(hidden_reasoning, job=job, plan=plan)

    rejected: Any = response_record(job, dict(result, provider="Chutes"), plan)
    assert "unexpected provider" in rejected["error"]
    missing_model: Any = response_record(job, dict(result, served_model=None), plan)
    assert "unexpected served model" in missing_model["error"]
    reasoning_content: Any = response_record(
        job,
        dict(result, native_reasoning={"reasoning_content": "hidden"}),
        plan,
    )
    assert "reasoning control violation" in reasoning_content["error"]


def test_plan_hash_commits_expected_answers(tmp_path: Path) -> None:
    build_ascii_variation_dataset(tmp_path, contract_dir=CONTRACT_DIR)
    questions: Any = [json.loads(line) for line in (tmp_path / "qa.jsonl").read_text().splitlines()][:1]
    jobs = build_jobs(
        tmp_path,
        questions=questions,
        variants=("flat_sorted",),
        max_requests=None,
    )
    args: Any = SimpleNamespace(
        dataset_dir=tmp_path,
        model="qwen/qwen3.8-27b",
        output_dir=tmp_path / "eval",
        concurrency=2,
        temperature=0.0,
        max_tokens=256,
        timeout=180.0,
    )
    original = build_plan(
        args,
        questions=questions,
        variants=("flat_sorted",),
        jobs=jobs,
        provider_order=("AkashML",),
    )
    changed_questions: Any = json.loads(json.dumps(questions))
    changed_questions[0]["answer"] = {"tile": "T99"}
    changed = build_plan(
        args,
        questions=changed_questions,
        variants=("flat_sorted",),
        jobs=jobs,
        provider_order=("AkashML",),
    )

    assert original["manifest_sha256"] != changed["manifest_sha256"]


def test_strict_json_scorer_allows_only_order_variation_for_sets() -> None:
    expected = {"tiles": ["T01", "T04", "T09"]}
    reordered = '{"tiles":["T09","T01","T04"]}'
    duplicate = '{"tiles":["T01","T04","T09","T09"]}'
    extra_key = '{"tiles":["T01","T04","T09"],"note":"yes"}'

    assert score_strict_json_answer(expected, reordered)["correct"]
    assert not score_strict_json_answer(expected, duplicate)["correct"]
    assert not score_strict_json_answer(expected, extra_key)["correct"]
