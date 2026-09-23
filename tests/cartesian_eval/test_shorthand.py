"""Real-parent integration checks for exact geometry, sparse facts, and oracle binding."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sft.cartesian_eval.contracts import (
    DIMENSIONS,
    compact,
    digest,
    object_list,
    object_map,
    parse_json,
    sha256,
    text,
)
from sft.cartesian_eval.geometry import ATLAS_ATOM, coordinate_points, parse_atom
from sft.cartesian_eval.geometry import LEGEND as ROOT_LEGEND
from sft.cartesian_eval.shorthand import (
    PARENT_SHA256,
    SCHEMA,
    VERSION,
    parent_row,
    score_response,
    validate_rows,
)
from sft.cartesian_eval.shorthand_dataset import DEFAULT_SOURCE, build_dataset, build_rows
from sft.cartesian_eval.shorthand_geometry import (
    EMPTY_DEFAULT,
    LEGEND,
    coordinate_mapping,
    decode,
    encode,
    mapping_artifact,
    project_text,
    root_response,
)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return build_rows()


def content(row: dict[str, Any], position: int = 0) -> str:
    return text(object_map(object_list(row["messages"])[position])["content"])


def test_all_154_exact_points_and_simple_aliases() -> None:
    mapping, points = coordinate_mapping(), coordinate_points()
    assert len(mapping) == len(set(mapping.values())) == 154
    assert Counter(p.family for p in points.values()) == {"T": 19, "N": 54, "E": 72, "P": 9}
    assert mapping["<T00>"] == "T(0,0)"
    assert mapping["<T01>"] == "T(4h,0)"
    assert mapping["<N01>"] == "N(2h,0.5)"
    assert mapping["<E00_01>"] == "E(h,0.75)"
    for token, point in points.items():
        atom = mapping[token]
        assert decode(atom) == parse_atom(point.atom()) == point
        assert encode(decode(atom)) == atom
        assert project_text(atom, reverse=True) == point.atom()
        assert point.x_root * 4 == Fraction((point.x_root * 4).numerator)
        assert point.y.denominator in {1, 2, 4}
        coefficient = point.x_root * 4
        alias = f"{point.family}({coefficient.numerator * 2}*h/2,{point.y.numerator}/{point.y.denominator})"
        assert decode(alias) == point
    assert decode("N(2*h,2/4)") == decode("N(4h/2,.50)") == points["<N01>"]
    assert decode("E(h/2,0.75)").x_root == Fraction(1, 8)
    assert decode("N(1/2*h,1/2)").x_root == Fraction(1, 8)
    for invalid in ("N(h/0,1)", "N(2h,1/0)", "N(*h,1)", "N(h+h,1)",
                    "N(h,1e0)", "N(0.8660254,0.5)", "N(h, 1)", "N(" + "1" * 513 + "h,1)"):
        with pytest.raises(ValueError):
            decode(invalid)
    with pytest.raises(ValueError, match="unknown"):
        root_response("E(h/2,0.75)")


def test_all_200_parent_cases_gold_and_visible_facts(rows: list[dict[str, Any]]) -> None:
    raw = DEFAULT_SOURCE.read_bytes()
    parents = [parse_json(line) for line in raw.decode().splitlines()]
    assert sha256(raw) == PARENT_SHA256
    report = validate_rows(rows)
    assert report["rows"] == report["gold_roundtrip_rows"] == 200
    assert report["by_split"] == {"test": 198, "validation": 2}
    assert (report["static_rows"], report["dynamic_rows"]) == (112, 88)
    assert report["by_family"] == {"direction": 64, "adjacency": 32, "incidence": 16, "ownership": 88}
    assert report["mapping_sha256"] == digest(mapping_artifact())
    for row, parent in zip(rows, parents, strict=True):
        metadata, original = object_map(row["metadata"]), object_map(parent["metadata"])
        assert row["schema"] == SCHEMA and metadata["version"] == VERSION
        assert row["id"] == row["row_id"] == parent["id"]
        assert metadata["cartesian_reference"] == {"id": parent["id"], "metadata": original}
        assert parent_row(metadata) == parent
        assert all(metadata[key] == original[key] for key in DIMENSIONS)
        gold = content(row, 1)
        assert gold == metadata["answer"] == project_text(content(parent, 1))
        for response in (gold, content(parent, 1)):
            score = object_map(score_response(gold, response, metadata))
            assert score["correct"] is score["format_valid"] is True
            assert score["scoring"] == SCHEMA and score["response_normalized"] == gold
        prompt, dense = content(row), content(parent)
        assert prompt.startswith(LEGEND + "\n") and dense.startswith(ROOT_LEGEND + "\n")
        assert ATLAS_ATOM.search(prompt + gold) is None
        assert metadata["prompt_sha256"] == sha256(prompt.encode())
        if "Entity inventory: " in dense:
            inventory = prompt.split("Entity inventory: ")[1].split("\n")[0].split()
            assert inventory == list(coordinate_mapping().values())
            assert len(inventory) == 154 and EMPTY_DEFAULT not in prompt
            assert project_text(prompt.removeprefix(LEGEND), reverse=True) == dense.removeprefix(ROOT_LEGEND)
        else:
            assert prompt.count(EMPTY_DEFAULT) == 1
            source_board = dense.split("Board: ")[1].split("\n")[0]
            sparse_board = prompt.split("Board: ")[1].split("\n")[0]
            source = dict(clause.split(" ", 1) for clause in source_board.split("; "))
            actual = dict(clause.split(" ", 1) for clause in
                          project_text(sparse_board, reverse=True).split("; "))
            assert len(source) == 155
            assert {key: actual.get(key, "empty") for key in source} == source
            assert actual.items() <= source.items()
            assert [key for key in source if key in actual] == list(actual)
            assert all(key.startswith(("N(", "E(")) and source[key] == "empty"
                       for key in source.keys() - actual.keys())
            assert not any(key.startswith(("N(", "E(")) and value == "empty"
                           for key, value in actual.items())
            assert sum(key.startswith("T(") for key in actual) == 19
            assert sum(key.startswith("P(") for key in actual) == 9
            assert actual["robber"] == source["robber"]
            assert any(value == "desert none" for value in actual.values())
            # Participants and the entire question are byte-identical after reversing notation.
            restored = project_text(prompt.removeprefix(LEGEND), reverse=True)
            restored = restored.replace(project_text(sparse_board, reverse=True), source_board)
            assert restored.replace(EMPTY_DEFAULT + "\n", "") == dense.removeprefix(ROOT_LEGEND)
    assert sha256(DEFAULT_SOURCE.read_bytes()) == PARENT_SHA256


def test_complete_response_admission_and_color_scope(rows: list[dict[str, Any]]) -> None:
    row = next(row for row in rows if row["task_type"] == "symbolic_neighbors")
    metadata, gold = object_map(row["metadata"]), content(row, 1)
    radical_alias = root_response(gold).split()[0]
    aliases = " ".join(f"{p.family}({8 * p.x_root}*h/2,{p.y})" for p in map(decode, gold.split()))
    for response in ("\n".join(reversed(gold.split())), aliases,
                     root_response(gold).replace("sqrt(3)", "√3"), gold + "<|im_end|> </s> <pad>"):
        assert object_map(score_response(gold, response, metadata))["correct"] is True
    for response in (gold + " " + radical_alias, "N(999h,999)", "<N00>", gold + " <N00>",
                     "Answer: " + gold, gold + " explanation", gold + "<|im_end|> extra",
                     "N(2h,0.5000000001)", "N(2h,0.5) garbage", "N(0, 1)", ""):
        score = object_map(score_response(gold, response, metadata))
        assert score["correct"] is score["format_valid"] is False, response
    wrong = object_map(score_response(gold, "NONE", metadata))
    assert wrong["correct"] is False and wrong["format_valid"] is True
    owner = next(row for row in rows if row["task_type"] == "symbolic_piece_owner" and content(row, 1) != "NONE")
    m, color = object_map(owner["metadata"]), content(owner, 1)
    assert object_map(score_response(color, color, m))["correct"] is True
    assert object_map(score_response(color, color + " " + color, m))["format_valid"] is False
    assert object_map(score_response(color, "ULTRAVIOLET", m))["correct"] is False
    boolean = next(row for row in rows if row["task_type"] == "symbolic_direction")
    answer = content(boolean, 1)
    assert object_map(score_response(answer, "no" if answer == "yes" else "yes",
                                     object_map(boolean["metadata"])))["correct"] is False
    assert score_response("", "", {"schema": "unrelated"}) is None


def test_gold_reference_membership_and_prompt_tampering(rows: list[dict[str, Any]]) -> None:
    m = object_map(rows[0]["metadata"])
    gold = text(m["answer"])
    wrong = "no" if gold == "yes" else "yes"
    with pytest.raises(ValueError, match="gold"):
        score_response(wrong, wrong, m)
    for key, value in (("answer", wrong), ("mapping_sha256", "0" * 64),
                       ("parent_sha256", "0" * 64), ("source_id", "another-case")):
        with pytest.raises(ValueError, match="projection"):
            score_response(gold, gold, {**m, key: value})
    reference = deepcopy(object_map(m["cartesian_reference"]))
    parent = object_map(reference["metadata"])
    atlas_reference = object_map(parent["atlas_reference"])
    atlas = {**object_map(atlas_reference["metadata"]), "canonical_answer": wrong, "answer": wrong}
    reference["metadata"] = {**parent, "atlas_reference": {**atlas_reference, "metadata": atlas}}
    with pytest.raises(ValueError, match="immutable source"):
        score_response(gold, gold, {**m, "cartesian_reference": reference})
    for malformed in (rows[:-1], [rows[1], rows[0], *rows[2:]], [rows[0], rows[0], *rows[2:]]):
        with pytest.raises(ValueError):
            validate_rows(malformed)
    changed = deepcopy(rows)
    changed[0]["messages"] = [{"role": "user", "content": "Answer yes"},
                               {"role": "assistant", "content": gold}]
    with pytest.raises(ValueError, match="prompt/gold"):
        validate_rows(changed)


def test_exclusive_builder_and_artifact_receipts(tmp_path: Path) -> None:
    destination = tmp_path / "panel"
    result = build_dataset(destination)
    assert result["rows"] == 200 and result["mapping_entities"] == 154
    manifest = parse_json((destination / "manifest.json").read_text())
    rows = [parse_json(line) for line in (destination / "eval.jsonl").read_text().splitlines()]
    assert validate_rows(rows) == manifest["validation"]
    assert object_map(manifest["parent"])["sha256"] == PARENT_SHA256
    assert object_map(manifest["intent"])["model_run_performed"] is False
    assert manifest["added_tokenizer_tokens"] == []
    mapping = parse_json((destination / "mapping.json").read_text())
    assert digest(mapping) == manifest["mapping_sha256"]
    for name, receipt in object_map(manifest["files"]).items():
        assert sha256((destination / name).read_bytes()) == object_map(receipt)["sha256"]
    preview = (destination / "preview.md").read_text()
    assert preview.count("\n## symbolic_") == 8 and PARENT_SHA256 in preview
    assert "Gold:" in preview and "Component counts:" in preview
    with pytest.raises(FileExistsError):
        build_dataset(destination)
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path / "missing" / "panel")
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text(compact(rows[0]) + "\n")
    with pytest.raises(ValueError, match="parent source hash"):
        build_dataset(tmp_path / "rejected", source=corrupt)
    assert not (tmp_path / "rejected").exists()
