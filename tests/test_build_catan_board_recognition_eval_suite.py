import hashlib
import json
import re
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_catan_board_recognition_eval_suite import (
    assert_aligned,
    build_rows,
    compact_metadata,
    main,
    read_jsonl,
    write_jsonl,
)
from sft.behavior_diagnostics import records_fingerprint


ROOT = Path(__file__).resolve().parents[1] / "artifacts/generated/board_recognition/replay_v1"
CORRECTED_SOURCE = "spatial_robber_choice_order_v1"


@pytest.fixture
def supplement(tmp_path):
    source = tmp_path / CORRECTED_SOURCE
    audits = [
        {
            "query_id": "node-question",
            "task_family": "spatial_grounding",
            "task_type": "node_direction_token",
            "prompt": "Which node is above the other: <N00> or <N01>?",
            "answer": "<N01>",
        },
        {
            "query_id": "robber-question",
            "task_family": "robber",
            "task_type": "robber_presence_positive",
            "prompt": "Is the robber on <T00>?",
            "answer": "yes",
        },
        {
            "query_id": "tile-question",
            "task_family": "spatial_grounding",
            "task_type": "tile_adjacent_no",
            "prompt": "Are <T00> and <T18> adjacent tiles?",
            "answer": "no",
        },
    ]
    rows = []
    for audit in audits:
        audit.update(state_id="board", split="validation", image_name="nested/board.png")
        rows.append(
            {
                "images": [audit["image_name"]],
                "messages": [
                    {"role": "user", "content": f"<image>\n{audit['prompt']}"},
                    {"role": "assistant", "content": audit["answer"]},
                ],
            }
        )
    write_jsonl(source / "validation.jsonl", rows)
    write_jsonl(source / "audit/validation.jsonl", audits)
    return source, rows, audits


def test_spatial_only_uses_explicit_source_without_base_and_preserves_rows(tmp_path, supplement):
    source, rows, audits = supplement
    missing_root = tmp_path / "no-base-dataset"
    result = build_rows(missing_root, "validation", supplement_dir=source, spatial_only=True)

    assert not missing_root.exists()
    assert len(result) == 2
    assert result == build_rows(
        missing_root, "validation", supplement_dir=source, spatial_only=True
    )
    for enriched, index, entity in zip(result, [0, 2], ["node", "tile"], strict=True):
        assert enriched["images"] == rows[index]["images"]
        assert enriched["messages"] == rows[index]["messages"]
        assert enriched["id"] == f"{CORRECTED_SOURCE}:validation:{audits[index]['query_id']}"
        assert enriched["metadata"] == {
            **compact_metadata(audits[index], suite="spatial_robber", category="spatial_grounding"),
            "entity_type": entity,
            "source_supplement": str(source.resolve()),
            "source_split": "validation",
        }
    assert len({row["id"] for row in result}) == 2
    legacy = [{**row, "id": row["metadata"]["query_id"]} for row in result]
    assert records_fingerprint(result) != records_fingerprint(legacy)


def test_default_build_keeps_base_and_robber_rows_and_legacy_ids(tmp_path, supplement):
    source, rows, audits = supplement
    base = {
        "images": ["base.png"],
        "messages": [
            {"role": "user", "content": "<image>\nWhat resource is on <T00>?"},
            {"role": "assistant", "content": "WOOD"},
        ],
    }
    audit = {
        "query_id": "base-question",
        "entity_type": "tile",
        "head": "tile.resource",
        "semantic_prompt": "What resource is on <T00>?",
        "semantic_answer": "WOOD",
    }
    write_jsonl(tmp_path / "ms_swift_bidirectional_v1/mixed/validation.jsonl", [base])
    write_jsonl(
        tmp_path / "ms_swift_bidirectional_v1/mixed_index/validation.jsonl",
        [{"mixed_index": 0, "row_kind": "forward", "query_id": audit["query_id"]}],
    )
    write_jsonl(tmp_path / "ms_swift_semantic_v1/audit/validation.jsonl", [audit])
    write_jsonl(tmp_path / "ms_swift_bidirectional_v1/audit/validation.jsonl", [])
    write_jsonl(tmp_path / "spatial_robber_v1/validation.jsonl", rows)
    write_jsonl(tmp_path / "spatial_robber_v1/audit/validation.jsonl", audits)

    result = build_rows(tmp_path, "validation")
    assert [row["id"] for row in result] == [audit["query_id"], *[a["query_id"] for a in audits]]
    assert result[0] == {
        **base,
        "id": audit["query_id"],
        "metadata": {
            **compact_metadata(audit, suite="bidirectional", category="tile.resource"),
            "row_kind": "forward",
        },
    }
    assert [row["messages"] for row in result[1:]] == [row["messages"] for row in rows]
    explicit = build_rows(tmp_path, "validation", supplement_dir=source)
    assert len(explicit) == 4
    assert explicit[0] == result[0]
    assert explicit[2]["metadata"]["task_family"] == "robber"
    narrow = build_rows(tmp_path, "validation", spatial_only=True)
    assert len(narrow) == 2
    assert all(row["id"].startswith("spatial_robber_v1:validation:") for row in narrow)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt", "Wrong question", "prompt mismatch"),
        ("answer", "yes", "answer mismatch"),
        ("image_name", "wrong.png", "image mismatch"),
        ("split", "train", "split mismatch"),
        ("task_type", "edge_direction_token", "unknown spatial entity"),
    ],
)
def test_spatial_alignment_failures(tmp_path, supplement, field, value, error):
    source, _, audits = supplement
    audits[0][field] = value
    write_jsonl(source / "audit/validation.jsonl", audits, overwrite=True)
    with pytest.raises(ValueError, match=error):
        build_rows(tmp_path, "validation", supplement_dir=source, spatial_only=True)


@pytest.mark.parametrize("duplicate", [False, True])
def test_spatial_rejects_length_mismatch_and_duplicate_query_ids(tmp_path, supplement, duplicate):
    source, rows, audits = supplement
    if duplicate:
        rows.append(rows[0])
        audits.append(audits[0])
    else:
        audits.pop()
    write_jsonl(source / "validation.jsonl", rows, overwrite=True)
    write_jsonl(source / "audit/validation.jsonl", audits, overwrite=True)
    with pytest.raises(
        ValueError, match="duplicate query_id" if duplicate else "different lengths"
    ):
        build_rows(tmp_path, "validation", supplement_dir=source, spatial_only=True)


def test_writer_is_deterministic_and_requires_explicit_overwrite(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    rows = [{"id": "one", "metadata": {"category": "spatial_grounding"}}]
    assert write_jsonl(first, iter(rows)) == write_jsonl(second, rows) == 1
    original = first.read_bytes()
    assert original == second.read_bytes()
    with pytest.raises(FileExistsError):
        write_jsonl(first, [])
    assert first.read_bytes() == original
    assert write_jsonl(first, [], overwrite=True) == 0
    assert first.read_bytes() == b""


@pytest.mark.parametrize("answer_only", [False, True])
def test_cli_spatial_only_and_overwrite_policy(tmp_path, supplement, monkeypatch, answer_only):
    source, _, _ = supplement
    output = tmp_path / "eval.jsonl"
    smoke = tmp_path / "smoke.jsonl"
    argv = [
        "build_eval",
        "--root",
        str(tmp_path / "missing-base"),
        "--supplement-dir",
        str(source),
        "--spatial-only",
        "--output",
        str(output),
        "--smoke-output",
        str(smoke),
        "--smoke-rows",
        "1",
        *(["--answer-only"] if answer_only else []),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    assert len(read_jsonl(output)) == 2
    assert all(
        (row["metadata"].get("prompt_variant") == "answer_only") == answer_only
        for row in read_jsonl(output)
    )
    assert len(read_jsonl(smoke)) == 1
    before = output.read_bytes(), smoke.read_bytes()
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        main()
    assert (output.read_bytes(), smoke.read_bytes()) == before
    monkeypatch.setattr(sys, "argv", [*argv, "--overwrite"])
    assert main() == 0
    assert (output.read_bytes(), smoke.read_bytes()) == before


def test_cli_preflights_smoke_output_before_writing(tmp_path, supplement, monkeypatch):
    source, _, _ = supplement
    output = tmp_path / "new.jsonl"
    smoke = tmp_path / "existing.jsonl"
    write_jsonl(smoke, [{"id": "existing"}])
    before = smoke.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eval",
            "--supplement-dir",
            str(source),
            "--spatial-only",
            "--output",
            str(output),
            "--smoke-output",
            str(smoke),
        ],
    )
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        main()
    assert not output.exists()
    assert smoke.read_bytes() == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_cli_rejects_same_output_and_smoke_path(tmp_path, monkeypatch, overwrite):
    output = tmp_path / "eval.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eval",
            "--output",
            str(output),
            "--smoke-output",
            str(output),
            *(["--overwrite"] if overwrite else []),
        ],
    )
    with pytest.raises(ValueError, match="different paths"):
        main()
    assert not output.exists()


def test_answer_only_requires_spatial_only_before_reads_or_writes(tmp_path, monkeypatch, capsys):
    missing_root = tmp_path / "missing"
    with pytest.raises(ValueError, match="--answer-only requires --spatial-only"):
        build_rows(missing_root, "validation", answer_only=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eval",
            "--answer-only",
            "--overwrite",
            "--output",
            str(missing_root / "eval.jsonl"),
            "--smoke-output",
            str(missing_root / "smoke.jsonl"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--answer-only requires --spatial-only" in capsys.readouterr().err
    assert not missing_root.exists()


@pytest.mark.parametrize("answer", ["yes", "no"])
def test_answer_only_is_neutral_and_does_not_mutate_sources(
    tmp_path, supplement, monkeypatch, answer
):
    source, rows, audits = supplement
    audits[2].update(task_type=f"tile_adjacent_{answer}", answer=answer)
    rows[2]["messages"][1]["content"] = answer
    sources = {source / "validation.jsonl": rows, source / "audit/validation.jsonl": audits}
    before_sources = deepcopy(sources)
    monkeypatch.setattr(
        "scripts.build_catan_board_recognition_eval_suite.read_jsonl", sources.__getitem__
    )
    original = build_rows(tmp_path, "validation", supplement_dir=source, spatial_only=True)
    before_original = deepcopy(original)
    control = build_rows(
        tmp_path, "validation", supplement_dir=source, spatial_only=True, answer_only=True
    )
    assert sources == before_sources
    assert original == before_original
    instructions = [
        "Answer with only one of the two tokens shown in the question.",
        "Answer with exactly one word: yes or no.",
    ]
    for base, row, instruction in zip(original, control, instructions, strict=True):
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
        assert row["messages"][0] is not base["messages"][0]
    assert records_fingerprint(original) != records_fingerprint(control)


def test_corrected_validation_source_integrity_balance_and_eval_rows(tmp_path):
    source = ROOT / CORRECTED_SOURCE
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
    annotations = read_jsonl(source / "validation.jsonl")
    audits = read_jsonl(source / "audit/validation.jsonl")
    states = {state["sample_id"]: state for state in read_jsonl(ROOT / "manifest.jsonl")}
    assert len(annotations) == len(audits) == 312
    assert Counter(audit["task_family"] for audit in audits) == {
        "spatial_grounding": 120,
        "robber": 192,
    }
    for row, audit in zip(annotations, audits, strict=True):
        state = states[audit["state_id"]]
        assert_aligned(row, audit)
        assert audit["split"] == state["split"] == "validation"
        assert audit["density_bin"] == state["density_bin"]
        assert row["images"] == [audit["image_name"]] == [Path(state["image_path"]).name]
        assert row["curriculum_stage"] == audit["curriculum_stage"]
        assert (ROOT / "images" / row["images"][0]).is_file()

    result = build_rows(
        tmp_path / "missing-base", "validation", supplement_dir=source, spatial_only=True
    )
    selected = [
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
    answers = Counter()
    choices = Counter()
    state_choices = Counter()
    image_names = set()
    for enriched, (row, audit) in zip(result, selected, strict=True):
        assert enriched["messages"] == row["messages"]
        assert enriched["images"] == row["images"]
        assert enriched["metadata"]["query_id"] == audit["query_id"]
        assert enriched["metadata"]["source_supplement"] == str(source)
        assert enriched["metadata"]["source_split"] == "validation"
        assert enriched["metadata"]["density_bin"] == "empty"
        answer = row["messages"][1]["content"]
        answers[answer if answer in {"yes", "no"} else "token"] += 1
        if audit["task_type"].endswith("_token"):
            tokens = re.findall(r"<[NT]\d{2}>", row["messages"][0]["content"])
            assert len(tokens) == len(set(tokens)) == 2
            position = tokens.index(answer) + 1
            choices[(enriched["metadata"]["entity_type"], position)] += 1
            state_choices[(audit["state_id"], enriched["metadata"]["entity_type"], position)] += 1
        image_names.update(row["images"])
    assert answers == {"yes": 50, "no": 50, "token": 20}
    assert choices == {("node", 1): 5, ("node", 2): 5, ("tile", 1): 5, ("tile", 2): 5}
    assert len(state_choices) == 20 and set(state_choices.values()) == {1}
    assert len(image_names) == 5
    for state_id in {row["metadata"]["state_id"] for row in result}:
        state = states[state_id]
        image_path = ROOT / state["image_path"]
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


def test_answer_only_real_validation_is_matched_and_preserves_original_bytes(tmp_path):
    original_path = ROOT / "evals/spatial_choice_order_validation_v1.jsonl"
    if not original_path.is_file() or not (ROOT / CORRECTED_SOURCE / "metadata.json").is_file():
        pytest.skip("original spatial eval and corrected source are not available locally")
    original_bytes = original_path.read_bytes()
    original = read_jsonl(original_path)
    rebuilt = build_rows(
        ROOT, "validation", supplement_dir=ROOT / CORRECTED_SOURCE, spatial_only=True
    )
    rebuilt_path = tmp_path / "rebuilt.jsonl"
    write_jsonl(rebuilt_path, rebuilt)
    assert rebuilt_path.read_bytes() == original_bytes
    control = build_rows(
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
    instructions = Counter()
    for base, row in zip(original, control, strict=True):
        instruction = (
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
