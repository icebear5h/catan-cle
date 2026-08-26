import json
import multiprocessing
import time
from types import SimpleNamespace

import pytest
from PIL import Image

import scripts.eval_catan_strict_vision_probe as vision_eval
from scripts.eval_catan_strict_vision_probe import (
    admissible_record,
    build_jobs,
    build_plan,
    call_job_with_hard_deadline,
    response_record,
    validate_dataset,
)
from scripts.render_catan_strict_vision_probe import (
    DEFAULT_SOURCE_DIR,
    canonical_id_map,
    canonicalize_question,
    render_strict_vision_probe,
    validate_source_dataset,
)


ALIAS_PATH = DEFAULT_SOURCE_DIR / "aliases/ascii_board_00.json"


def test_question_projection_inverts_text_aliases_to_engine_ids():
    aliases = json.loads(ALIAS_PATH.read_text())
    mapping = canonical_id_map(aliases)
    manifest = json.loads((DEFAULT_SOURCE_DIR / "manifest.jsonl").read_text().splitlines()[0])
    question = json.loads((DEFAULT_SOURCE_DIR / "qa.jsonl").read_text().splitlines()[0])

    projected = canonicalize_question(question, mapping, manifest_row=manifest)

    assert set(mapping) == {
        *(f"T{index:02d}" for index in range(19)),
        *(f"N{index:02d}" for index in range(54)),
        *(f"E{index:02d}" for index in range(72)),
        *(f"P{index:02d}" for index in range(9)),
    }
    assert len(set(mapping.values())) == 154
    assert projected["id"] == question["id"]
    assert projected["source_answer"] == question["answer"]
    assert projected["answer"] != question["answer"]
    assert projected["identity_projection"] == "canonical_engine_ids"
    assert "fact_digest" not in projected


def test_rendered_probe_uses_raw_images_and_strict_scoring(tmp_path):
    source_metadata, source_manifest, source_questions = validate_source_dataset(DEFAULT_SOURCE_DIR)
    output_dir = tmp_path / "strict_vision_probe"

    rendered_metadata = render_strict_vision_probe(
        DEFAULT_SOURCE_DIR,
        output_dir,
        image_size=256,
        view_padding_factor=0.933134,
        target_board_canvas_fraction=0.9,
    )
    metadata, manifest, questions = validate_dataset(output_dir)
    jobs = build_jobs(output_dir, manifest=manifest, questions=questions)

    assert source_metadata["question_count"] == 60
    assert len(source_manifest) == 12
    assert len(source_questions) == len(questions) == len(jobs) == 60
    assert rendered_metadata == metadata
    assert metadata["render_variant"]["image_annotation"] is None
    assert metadata["identity_projection"]["image_contains_entity_labels"] is False
    assert len(manifest) == 12
    assert all(row["identity_projection"] == "canonical_engine_ids" for row in questions)
    assert all("Visual coordinate atlas" not in job["prompt"] for job in jobs)
    with Image.open(output_dir / manifest[0]["image_path"]) as image:
        assert image.size == (256, 256)
    assert "unannotated 256px board screenshots" in (output_dir / "README.md").read_text()

    args = SimpleNamespace(
        model="test/model",
        dataset_dir=output_dir,
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
        "served_model": "test/model",
        "provider": "novita",
        "usage": {"completion_tokens_details": {"reasoning_tokens": 0}},
        "latency_ms": 12,
    }
    record = response_record(job, result, plan)

    assert plan["request_settings"]["image_annotation"] is None
    assert record["score"]["correct"] is True
    assert admissible_record(record, job=job, plan=plan)


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="monkeypatched child requires fork",
)
def test_sequential_request_has_a_hard_wall_clock_deadline(monkeypatch):
    def slow_call(_api_key, _args, _job):
        time.sleep(0.2)
        return {"response": "too late"}

    monkeypatch.setattr(vision_eval, "call_job", slow_call)
    args = SimpleNamespace(timeout=0.02)

    result = call_job_with_hard_deadline("key", args, {"qa": {"id": "q"}})

    assert "hard wall-clock timeout" in result["error"]
    assert result["latency_ms"] < 150
