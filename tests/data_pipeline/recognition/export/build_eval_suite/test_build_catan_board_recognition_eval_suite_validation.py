"""Corrected-source integrity and matched validation."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.board_recognition.build_catan_board_recognition_eval_suite import (
    assert_aligned,
    build_rows,
    read_jsonl,
    write_jsonl,
)
from sft.analysis.behavior_diagnostics import records_fingerprint

from .support import CORRECTED_SOURCE, ROOT


def test_corrected_validation_source_integrity_balance_and_eval_rows(tmp_path: Path) -> None:
    source: Any = ROOT / CORRECTED_SOURCE
    if not (source / "metadata.json").is_file():
        pytest.skip("corrected replay supplement is not available locally")
    metadata = json.loads((source / "metadata.json").read_text())
    assert metadata["schema"] == "catan_spatial_robber_supplement/v1"
    assert Path(metadata["source_dataset"]).resolve() == ROOT
    assert Path(metadata["image_root"]).resolve() == ROOT / "images"
    assert (
        metadata["source_manifest_sha256"]
        == hashlib.sha256((ROOT / "manifest.jsonl").read_bytes()).hexdigest()
    )
    assert metadata["split_counts"]["validation"] == 312
    assert metadata["task_counts"]["validation"] == {"spatial_grounding": 120, "robber": 192}
    for key in ("annotations", "audit"):
        path = source / metadata["files"]["validation"][key]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == metadata["files"]["validation"][f"{key}_sha256"]
        )
    annotations: Any = read_jsonl(source / "validation.jsonl")
    audits: Any = read_jsonl(source / "audit/validation.jsonl")
    states: Any = {state["sample_id"]: state for state in read_jsonl(ROOT / "manifest.jsonl")}
    assert len(annotations) == len(audits) == 312
    assert Counter(audit["task_family"] for audit in audits) == {
        "spatial_grounding": 120,
        "robber": 192,
    }
    for row, audit in zip(annotations, audits, strict=True):
        state: Any = states[audit["state_id"]]
        assert_aligned(row, audit)
        assert audit["split"] == state["split"] == "validation"
        assert audit["density_bin"] == state["density_bin"]
        assert row["images"] == [audit["image_name"]] == [Path(state["image_path"]).name]
        assert row["curriculum_stage"] == audit["curriculum_stage"]
        assert (ROOT / "images" / row["images"][0]).is_file()

    result: Any = build_rows(
        tmp_path / "missing-base", "validation", supplement_dir=source, spatial_only=True
    )
    selected: Any = [
        (row, audit)
        for row, audit in zip(annotations, audits, strict=True)
        if audit["task_family"] == "spatial_grounding"
    ]
    assert len(result) == len({row["id"] for row in result}) == 120
    assert len({row["metadata"]["query_id"] for row in result}) == 120
    assert {row["id"] for row in result}.isdisjoint(audit["query_id"] for audit in audits)
    task_counts = Counter(row["metadata"]["task_type"] for row in result)
    assert task_counts == {
        f"{entity}_{task}_{answer}": 10
        for entity, tasks in [
            ("node", ["direction", "adjacent", "connected"]),
            ("tile", ["direction", "adjacent"]),
        ]
        for task in tasks
        for answer in (["yes", "no", "token"] if task == "direction" else ["yes", "no"])
    }
    entity_counts = Counter(row["metadata"]["entity_type"] for row in result)
    assert entity_counts == {"node": 70, "tile": 50}
    answers: Any = Counter()
    choices: Any = Counter()
    state_choices: Any = Counter()
    image_names: Any = set()
    for enriched, (row, audit) in zip(result, selected, strict=True):
        assert enriched["messages"] == row["messages"]
        assert enriched["images"] == row["images"]
        assert enriched["metadata"]["query_id"] == audit["query_id"]
        assert enriched["metadata"]["source_supplement"] == str(source)
        assert enriched["metadata"]["source_split"] == "validation"
        assert enriched["metadata"]["density_bin"] == "empty"
        answer: Any = row["messages"][1]["content"]
        answers[answer if answer in {"yes", "no"} else "token"] += 1
        if audit["task_type"].endswith("_token"):
            tokens: Any = re.findall(r"<[NT]\d{2}>", row["messages"][0]["content"])
            assert len(tokens) == len(set(tokens)) == 2
            position: Any = tokens.index(answer) + 1
            choices[(enriched["metadata"]["entity_type"], position)] += 1
            state_choices[(audit["state_id"], enriched["metadata"]["entity_type"], position)] += 1
        image_names.update(row["images"])
    assert answers == {"yes": 50, "no": 50, "token": 20}
    assert choices == {("node", 1): 5, ("node", 2): 5, ("tile", 1): 5, ("tile", 2): 5}
    assert len(state_choices) == 20 and set(state_choices.values()) == {1}
    assert len(image_names) == 5
    for state_id in {row["metadata"]["state_id"] for row in result}:
        state = states[state_id]
        image_path: Any = ROOT / state["image_path"]
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == state["sha256"]["image"]
        with Image.open(image_path) as image:
            assert list(image.size) == state["image_size"]
            image.verify()
    legacy = [{**row, "id": row["metadata"]["query_id"]} for row in result]
    assert records_fingerprint(result) != records_fingerprint(legacy)
    eval_path = ROOT / "evals/spatial_choice_order_validation_v1.jsonl"
    if eval_path.exists():
        assert read_jsonl(eval_path) == result
        expected_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in result).encode()
        assert eval_path.read_bytes() == expected_bytes
    print(
        json.dumps(
            {
                "source_rows": len(annotations),
                "eval_rows": len(result),
                "task_counts": dict(sorted(task_counts.items())),
                "entity_counts": dict(entity_counts),
                "answers": dict(answers),
                "choice_positions": {
                    f"{entity}:{position}": count for (entity, position), count in choices.items()
                },
                "images": len(image_names),
            },
            sort_keys=True,
        )
    )


def test_answer_only_real_validation_is_matched_and_preserves_original_bytes(tmp_path: Path) -> None:
    original_path = ROOT / "evals/spatial_choice_order_validation_v1.jsonl"
    if not original_path.is_file() or not (ROOT / CORRECTED_SOURCE / "metadata.json").is_file():
        pytest.skip("original spatial eval and corrected source are not available locally")
    original_bytes = original_path.read_bytes()
    original: Any = read_jsonl(original_path)
    rebuilt = build_rows(
        ROOT, "validation", supplement_dir=ROOT / CORRECTED_SOURCE, spatial_only=True
    )
    rebuilt_path = tmp_path / "rebuilt.jsonl"
    write_jsonl(rebuilt_path, rebuilt)
    assert rebuilt_path.read_bytes() == original_bytes
    control: Any = build_rows(
        ROOT,
        "validation",
        supplement_dir=ROOT / CORRECTED_SOURCE,
        spatial_only=True,
        answer_only=True,
    )
    assert len(original) == len(control) == len({row["id"] for row in control}) == 120
    assert len({row["metadata"]["query_id"] for row in control}) == 120
    assert {row["id"] for row in original}.isdisjoint(row["id"] for row in control)
    choices = Counter()
    instructions: Any = Counter()
    for base, row in zip(original, control, strict=True):
        instruction: Any = (
            "Answer with only one of the two tokens shown in the question."
            if base["metadata"]["task_type"].endswith("_direction_token")
            else "Answer with exactly one word: yes or no."
        )
        instructions[instruction] += 1
        assert row == {
            **base,
            "id": base["id"] + ":answer_only",
            "metadata": {**base["metadata"], "prompt_variant": "answer_only"},
            "messages": [
                {
                    **base["messages"][0],
                    "content": base["messages"][0]["content"] + "\n" + instruction,
                },
                *base["messages"][1:],
            ],
        }
        if base["metadata"]["task_type"].endswith("_direction_token"):
            tokens = re.findall(r"<[NT]\d{2}>", row["messages"][0]["content"])
            assert len(tokens) == 2
            choices[
                (row["metadata"]["entity_type"], tokens.index(row["messages"][1]["content"]) + 1)
            ] += 1
    assert instructions == {
        "Answer with exactly one word: yes or no.": 100,
        "Answer with only one of the two tokens shown in the question.": 20,
    }
    assert choices == {("node", 1): 5, ("node", 2): 5, ("tile", 1): 5, ("tile", 2): 5}
    assert original_path.read_bytes() == original_bytes
    output = ROOT / "evals/spatial_choice_order_answer_only_validation_v1.jsonl"
    if output.exists():
        assert read_jsonl(output) == control
        assert (
            output.read_bytes()
            == "".join(json.dumps(row, sort_keys=True) + "\n" for row in control).encode()
        )
        print(f"answer_only_sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    print(f"original_sha256={hashlib.sha256(original_bytes).hexdigest()}")
