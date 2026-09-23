"""Real-source matched-panel integration, geometry identity, and strict scoring."""
import copy
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from data_pipeline.board_recognition.sources import canonical_sha256, file_sha256
from sft.board.board_fluency_scoring import TRANSPORT_TOKENS
from sft.board.coordinate_comparison import (
    ATLAS_ATOM,
    COORDINATE_ATOM,
    DEFAULT_SOURCE,
    INCIDENCE_RELATIONS,
    PAIR_QUOTAS,
    REPRESENTATIONS,
    SCHEMA,
    build_comparison_rows,
    coordinate_mapping,
    load_source_rows,
    mapping_artifact,
    mapping_sha256,
    paired_summary,
    project_text,
    read_jsonl,
    score_coordinate_comparison,
    select_source_cases,
    validate_comparison_rows,
)
from sft.board.symbolic_board_tasks import atlas_geometry, symbolic_answer, symbolic_prompt
from sft.scripts.builders.build_coordinate_comparison import build_dataset
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator


@pytest.fixture(scope="module")
def sources() -> tuple[dict[str, object], ...]:
    return load_source_rows(DEFAULT_SOURCE)[0]


@pytest.fixture(scope="module")
def rows(sources: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
    return build_comparison_rows(sources)


def test_all_154_mapping_points_are_exact_integer_reversible_identities() -> None:
    atlas, mapping = atlas_geometry(), coordinate_mapping()
    inverse = mapping_artifact()["coordinates_to_atlas"]
    assert len(mapping) == len(inverse) == 154
    assert Counter(t[0] for t in inverse) == {"N": 54, "T": 19, "E": 72, "P": 9}
    assert all(COORDINATE_ATOM.fullmatch(atom) and inverse[atom] == token for token, atom in mapping.items())
    for token, (x, y) in atlas["positions"].items():
        assert mapping[token] == f"{token[1]}({2*x},{2*y})"
    for edge, (a, b) in atlas["edges"].items():
        x, y = [atlas["positions"][a][i] + atlas["positions"][b][i] for i in (0, 1)]
        assert mapping[edge] == f"E({x},{y})"
    for port in atlas["raw"]["ports"]:
        a, b = [f"<N{n:02d}>" for n in port["attached_nodes"]]
        x, y = [atlas["positions"][a][i] + atlas["positions"][b][i] for i in (0, 1)]
        assert mapping[port["token"]] == f"P({x},{y})"
    assert mapping_sha256() == canonical_sha256(mapping_artifact())
    mapping.clear()
    assert len(coordinate_mapping()) == 154  # Detached API cannot corrupt the cached crosswalk.


def test_complete_panel_roundtrips_source_pair_contract_and_no_prompt_leakage(rows: list[dict[str, object]], sources: tuple[dict[str, object], ...]) -> None:
    before = copy.deepcopy(sources)
    assert rows == build_comparison_rows(sources)
    assert sources == before
    report = validate_comparison_rows(rows)
    assert report["rows"] == len(set(report["ids"])) == 400
    assert report["pairs"] == 200 and report["by_representation"] == {"atlas": 200, "coordinates": 200}
    assert report["by_operation"] == {task: 2 * count for task, count in PAIR_QUOTAS.items()}
    assert report["by_family"] == {"direction": 128, "adjacency": 64, "incidence": 32, "ownership": 176}
    assert report["incidence_relations"] == dict.fromkeys(INCIDENCE_RELATIONS, 2)
    assert report["pairs_by_split"] == {"test": 198, "validation": 2}
    mapping = coordinate_mapping()
    inverse = {v: k for k, v in mapping.items()}
    originals = {s["row"]["id"]: s["row"] for s in sources}
    for a, c in zip(rows[::2], rows[1::2], strict=True):
        assert a["metadata"]["pair_id"] == c["metadata"]["pair_id"]
        for row in (a, c):
            m = row["metadata"]
            original = originals[m["source_id"]]
            assert all(m[k] == v for k, v in original["metadata"].items())
            assert m["source"]["row_sha256"] == canonical_sha256(original)
            gold = row["messages"][1]["content"]
            score = score_coordinate_comparison(gold, gold, m)
            assert score["correct"] and score["format_valid"] and score["scoring"] == SCHEMA
            assert score["canonical_expected_normalized"] == symbolic_answer(row["task_type"], m["target"])
            assert score["symbolic_scoring"] == row["task_type"]
        ap, cp = a["messages"][0]["content"], c["messages"][0]["content"]
        assert ATLAS_ATOM.search(cp) is None
        assert "fixed learned Catan atlas" not in cp
        assert not any(word in cp for word in ("canonical_answer", "query_sha256", "attached_nodes", "crosswalk"))
        target = a["metadata"]["target"]
        question = symbolic_prompt(a["task_type"], target).rsplit("\n", 1)[-1]
        if target["state"] is None:
            question = question.removeprefix("Use the fixed learned Catan atlas. ")
            assert ap.split("Entity inventory: ", 1)[1].split("\n", 1)[0].split() == list(mapping)
            inventory = cp.split("Entity inventory: ", 1)[1].split("\n", 1)[0].split()
            assert [inverse[atom] for atom in inventory] == list(mapping)
        else:
            assert "Entity inventory:" not in ap + cp
            board = cp.split("Board: ", 1)[1].split("\n", 1)[0]
            assert COORDINATE_ATOM.sub(lambda match: inverse[match[0]], board) == target["state"]["board"]
        assert ap.endswith(question) and cp.endswith(project_text(question, "coordinates"))
    json.dumps(report, allow_nan=False)


def test_missing_quotas_and_mismatched_pair_provenance_prompt_and_gold_fail(rows: list[dict[str, object]], sources: tuple[dict[str, object], ...]) -> None:
    missing_port = [s for s in sources if not (s["row"]["split"] == "validation"
                    and s["row"]["metadata"]["target"]["query"].get("token") == "<P03>")]
    with pytest.raises(ValueError, match="quota"):
        select_source_cases(missing_port)
    for field, value in (("pair_id", "wrong"), ("source_split", "train"), ("mapping_sha256", "0" * 64),
                         ("canonical_answer", "wrong"), ("canonical_fact_sha256", "0" * 64)):
        changed = copy.deepcopy(rows)
        changed[1]["metadata"][field] = value
        with pytest.raises(ValueError):
            validate_comparison_rows(changed)
    changed = copy.deepcopy(rows)
    dynamic = next(r for r in changed if r["metadata"]["target"]["state"] is not None)
    dynamic["metadata"]["provenance"]["state_id"] += "-changed"
    with pytest.raises(ValueError, match="provenance hash"):
        validate_comparison_rows(changed)
    changed = copy.deepcopy(rows)
    changed[1]["messages"][0]["content"] += "\nAnswer: no"
    with pytest.raises(ValueError, match="prompt"):
        validate_comparison_rows(changed)
    with pytest.raises(ValueError, match="400"):
        validate_comparison_rows(rows[:-1])
    changed = copy.deepcopy(rows)
    changed[1]["metadata"]["source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="paired|file hashes"):
        validate_comparison_rows(changed)


def test_strict_answers_and_trailing_versus_interior_transport_tokens(rows: list[dict[str, object]]) -> None:
    for rep in REPRESENTATIONS:
        row = next(r for r in rows if r["metadata"]["representation"] == rep
                   and r["task_type"] == "symbolic_neighbors" and r["metadata"]["target"]["query"]["token"][1] == "N")
        m, gold = row["metadata"], row["messages"][1]["content"]
        first = gold.split()[0]
        reversed_gold = "\n".join(reversed(gold.split()))
        assert score_coordinate_comparison(gold, reversed_gold, m)["correct"]
        for suffix in TRANSPORT_TOKENS:
            assert score_coordinate_comparison(gold, gold + " " + suffix, m)["correct"]
            score = score_coordinate_comparison(gold, first + suffix + " " + gold, m)
            assert not score["correct"] and not score["format_valid"]
        assert score_coordinate_comparison(gold, gold + "<|im_end|> </s> <pad>", m)["correct"]
        other_arm = project_text(m["canonical_answer"], "coordinates" if rep == "atlas" else "atlas")
        malformed = ["", "Answer: " + gold, gold + " explanation", gold + " " + first,
                     other_arm, first + " " + other_arm, gold + "<|im_end|> junk", "NONE " + first,
                     "N(999,999)", "<N99>", "T(0,0)", "<T00>", "```" + gold + "```"]
        if rep == "coordinates":
            malformed += ["N(+0,0)", "N(-0,0)", "N(00,0)", "N(0.0,0)", "N(0, 0)",
                          "N(０,0)", first.lower(), first + ",", "(" + first + ")"]
        for response in malformed:
            score = score_coordinate_comparison(gold, response, m)
            assert not score["correct"] and not score["format_valid"], (rep, response, score)
        wrong = score_coordinate_comparison(gold, "NONE", m)
        assert not wrong["correct"] and wrong["format_valid"]
        with pytest.raises(ValueError, match="gold"):
            score_coordinate_comparison("bogus expected", gold, m)
        for task in ("symbolic_direction", "symbolic_direction_choice", "symbolic_piece_owner"):
            scalar = next(r for r in rows if r["task_type"] == task and r["metadata"]["representation"] == rep
                          and r["metadata"]["answer"] != "NONE")
            m, gold = scalar["metadata"], scalar["metadata"]["answer"]
            for bad in (gold + " extra", gold + " " + gold, gold + "<|im_end|> extra", "NONE extra"):
                score = score_coordinate_comparison(gold, bad, m)
                assert not score["correct"] and not score["format_valid"]
            assert score_coordinate_comparison(gold, gold + "<|im_end|>", m)["correct"]
    assert score_coordinate_comparison("", "", {"schema": SCHEMA + "-other"}) is None
    with pytest.raises(ValueError):
        score_coordinate_comparison("", "", {"schema": SCHEMA})


def test_paired_summary_rescores_raw_outputs_and_requires_two_matching_arms(rows: list[dict[str, object]]) -> None:
    records = [{"id": r["id"], "metadata": copy.deepcopy(r["metadata"]),
                "expected": r["messages"][1]["content"], "response": r["messages"][1]["content"],
                "score": {"correct": False, "format_valid": False}} for r in rows[:8]]
    # One of each outcome, with intentionally untrustworthy recorded booleans.
    for index in (3, 4, 6, 7):
        records[index]["response"] = "extra prose"
        records[index]["score"] = {"correct": True, "format_valid": True}
    summary = paired_summary(records)
    assert {key: summary[key] for key in ("both", "atlas_only", "coordinates_only", "neither")} == dict.fromkeys(
        ("both", "atlas_only", "coordinates_only", "neither"), 1)
    assert summary["pairs"] == 4 and summary["rescored_from_raw"]
    assert summary["by_representation"]["atlas"]["format_invalid"] == 2
    assert summary["by_representation"]["coordinates"]["accuracy"] == 0.5
    assert len(summary["outcome_pair_ids"]["atlas_only"]) == 1
    assert sum(group["pairs"] for group in summary["by_task"].values()) == 4
    json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError, match="incomplete pair"):
        paired_summary(records[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        paired_summary(records + records[:1])
    changed = copy.deepcopy(records)
    changed[1]["metadata"]["source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="paired"):
        paired_summary(changed)


def test_native_evaluator_dispatches_and_summarizes_both_representations(rows: list[dict[str, object]]) -> None:
    records = []
    for row in rows:
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        gold = evaluator.expected_text(row)
        score = evaluator.score_response(gold, gold + "<|im_end|>", metadata=metadata)
        assert score["scoring"] == SCHEMA and score["correct"]
        records.append({"id": row["id"], "metadata": metadata, "expected": gold,
                        "response": gold + "<|im_end|>", "score": score})
    summary = evaluator.summarize(records)
    assert summary["correct"] == 400
    assert summary["coordinate_comparison"]["both"] == 200
    assert summary["by_representation"]["atlas"]["total"] == 200
    assert summary["by_representation"]["coordinates"]["total"] == 200


def test_builder_writes_inspectable_immutable_artifacts(tmp_path: Path, rows: list[dict[str, object]]) -> None:
    before = {name: file_sha256(DEFAULT_SOURCE / name) for name in ("test.jsonl", "validation.jsonl", "manifest.json")}
    output = tmp_path / "panel"
    result = build_dataset(output)
    assert result["rows"] == 400 and read_jsonl(output / "paired.jsonl") == rows
    assert {p.name for p in output.iterdir()} == {"paired.jsonl", "mapping.json", "manifest.json", "preview.md"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mapping_sha256"] == canonical_sha256(json.loads((output / "mapping.json").read_text()))
    assert "manifest.json" not in manifest["files"]
    for name, receipt in manifest["files"].items():
        assert file_sha256(output / name) == receipt["sha256"]
    preview = (output / "preview.md").read_text()
    assert len(re.findall(r"^### (atlas|coordinates) ", preview, re.MULTILINE)) == 16
    for area in ("direction", "adjacency", "incidence", "ownership"):
        assert f"## {area} /" in preview
    assert all(file_sha256(DEFAULT_SOURCE / name) == digest for name, digest in before.items())
    with pytest.raises(FileExistsError):
        build_dataset(output)
