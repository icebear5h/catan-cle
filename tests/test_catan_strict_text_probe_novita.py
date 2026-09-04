import json
from types import SimpleNamespace

from scripts import eval_catan_board_bench_full_graph_formats as base_evaluator
from scripts.eval_catan_strict_text_probe_novita import (
    DEFAULT_DATASET_DIR,
    FORMAT_NAME,
    admissible_record,
    build_plan,
    call_novita_text,
    configured_text_evaluator,
    response_record,
)


def test_text_runner_uses_locked_winner_and_strict_scoring(tmp_path):
    with configured_text_evaluator():
        metadata = json.loads((DEFAULT_DATASET_DIR / "metadata.json").read_text())
        base_evaluator.validate_dataset_metadata(metadata)
        questions = base_evaluator.validate_dataset_integrity(DEFAULT_DATASET_DIR, metadata)
        jobs = base_evaluator.build_jobs(
            DEFAULT_DATASET_DIR,
            questions=questions,
            formats=[FORMAT_NAME],
            max_requests=None,
        )
    args = SimpleNamespace(
        model="qwen/qwen3.8-max",
        dataset_dir=DEFAULT_DATASET_DIR,
        output_dir=tmp_path / "run",
        concurrency=1,
        temperature=0.0,
        max_tokens=256,
        timeout=240.0,
        request_interval=0.1,
    )
    plan = build_plan(args, metadata=metadata, questions=questions, jobs=jobs)
    job = jobs[0]
    result = {
        "response": job["qa"]["answer_text"],
        "served_model": args.model,
        "provider": "novita",
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        "latency_ms": 10,
    }
    record = response_record(job, result, plan)

    assert len(questions) == len(jobs) == 60
    assert plan["format"] == "indexed_tile_rows"
    assert job["board_presentation"].kind == "text"
    assert job["board_presentation"].format == "indexed_tile_rows/v3"
    assert job["board_presentation"].provenance.identity_space == (
        "opaque_board_local_ids"
    )
    assert job["board_presentation"].content in job["prompt"]
    expected_text = (
        DEFAULT_DATASET_DIR
        / "representations"
        / job["qa"]["sample_id"]
        / "indexed_tile_rows.txt"
    ).read_text().rstrip("\n")
    assert job["prompt"] == base_evaluator.build_prompt(
        FORMAT_NAME,
        expected_text,
        job["qa"],
    )
    assert plan["request_settings"]["image_input"] is False
    assert record["score"]["correct"] is True
    assert admissible_record(record, job=job, plan=plan)


def test_novita_text_call_disables_thinking(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "model": "qwen/qwen3.8-max",
                "choices": [{"message": {"content": '{"tile":"T03"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    class FakeClient:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(
        "scripts.eval_catan_strict_text_probe_novita.httpx.Client",
        FakeClient,
    )
    result = call_novita_text(
        "secret",
        "qwen/qwen3.8-max",
        prompt="prompt",
        temperature=0.0,
        max_tokens=256,
        timeout=240.0,
    )

    assert captured["payload"]["enable_thinking"] is False
    assert captured["payload"]["messages"][1]["content"] == "prompt"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert result["response"] == '{"tile":"T03"}'
    assert result["provider"] == "novita"
