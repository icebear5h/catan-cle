"""Real-parent integration: exact scaling, unchanged cases/facts, and strict admission."""
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
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
from sft.cartesian_eval.geometry import ATLAS_ATOM, coordinate_points
from sft.cartesian_eval.scaling import (
    PARENT_SHA256,
    SCHEMA,
    VARIANTS,
    parent_row,
    score_response,
    validate_rows,
)
from sft.cartesian_eval.scaling_dataset import DEFAULT_SOURCE, build_dataset, build_rows
from sft.cartesian_eval.scaling_geometry import (
    coordinate_mapping,
    decode,
    encode,
    legend,
    mapping_artifact,
    parent_response,
    project_text,
    scaling_prompt,
)
from sft.cartesian_eval.shorthand import score_response as parent_score
from sft.cartesian_eval.shorthand_geometry import EMPTY_DEFAULT
from sft.cartesian_eval.shorthand_geometry import LEGEND as PARENT_LEGEND


@pytest.fixture(scope="module")
def panels() -> dict[str, list[dict[str, Any]]]:
    return build_rows()


def content(row: Mapping[str, object], position: int = 0) -> str:
    return text(object_map(object_list(row["messages"])[position])["content"])


def test_all_154_points_roundtrip_and_physical_distances_scale_four() -> None:
    points = coordinate_points()
    for variant in VARIANTS:
        mapping = coordinate_mapping(variant)
        assert len(mapping) == len(set(mapping.values())) == 154
        assert mapping["<N01>"] == ("N(2h,2)" if variant == "scaled_h" else "N(2,2)")
        assert mapping["<E00_01>"] == ("E(h,3)" if variant == "scaled_h" else "E(1,3)")
        physical: dict[str, tuple[Fraction, Fraction]] = {}
        for token, atom in mapping.items():
            point = points[token]
            assert decode(atom, variant) == point
            assert encode(point, variant) == atom
            assert project_text(project_text(atom, variant, reverse=True), variant) == atom
            x_text, y_text = atom[2:-1].split(",")
            coefficient = x_text.removesuffix("h")
            k = Fraction("1" if coefficient == "" else "-1" if coefficient == "-" else coefficient)
            physical[token] = k, Fraction(y_text)
            assert physical[token] == (4 * point.x_root, 4 * point.y)
        for a, b in combinations(points, 2):
            x, y = physical[a]
            u, v = physical[b]
            original = 3 * (points[a].x_root - points[b].x_root) ** 2 + (points[a].y - points[b].y) ** 2
            assert 3 * (x - u) ** 2 + (y - v) ** 2 == 16 * original
    assert "hex side 4." in legend("scaled_h") and "Here h=sqrt(3)." in legend("scaled_h")
    assert "The axes have equal units:" in legend("scaled_h")
    assert "equal units" not in legend("integer_xy")
    assert "Physical Cartesian position is (sqrt(3)*x,y);" in legend("integer_xy")


def test_all_200_gold_cases_and_visible_facts_in_both_panels(
    panels: dict[str, list[dict[str, Any]]],
) -> None:
    raw = DEFAULT_SOURCE.read_bytes()
    parents = [parse_json(line) for line in raw.decode().splitlines()]
    assert sha256(raw) == PARENT_SHA256
    assert tuple(panels) == VARIANTS
    for variant, rows in panels.items():
        report = validate_rows(rows)
        assert report["rows"] == report["gold_roundtrip_rows"] == 200
        assert report["by_split"] == {"test": 198, "validation": 2}
        assert (report["static_rows"], report["dynamic_rows"]) == (112, 88)
        assert report["by_family"] == {"direction": 64, "adjacency": 32, "incidence": 16, "ownership": 88}
        assert report["by_representation"] == {variant: 200}
        assert report["mapping_sha256"] == digest(mapping_artifact(variant))
        for row, parent in zip(rows, parents, strict=True):
            metadata, original = object_map(row["metadata"]), object_map(parent["metadata"])
            assert row["schema"] == SCHEMA and metadata["variant"] == metadata["representation"] == variant
            assert row["id"] == row["row_id"] == parent["id"]
            assert metadata["shorthand_reference"] == {"id": parent["id"], "metadata": original}
            assert parent_row(metadata) == parent
            assert all(metadata[key] == original[key] for key in (*DIMENSIONS, "eval_position"))
            assert metadata["component_counts"] == original["component_counts"]
            gold, prompt = content(row, 1), content(row)
            assert gold == metadata["answer"] == project_text(content(parent, 1), variant)
            assert prompt == scaling_prompt(content(parent), variant)
            assert project_text(prompt.removeprefix(legend(variant)), variant, reverse=True) == content(parent).removeprefix(PARENT_LEGEND)
            assert ATLAS_ATOM.search(prompt + gold) is None
            assert metadata["prompt_sha256"] == sha256(prompt.encode())
            score = object_map(score_response(gold, gold, metadata))
            oracle = object_map(parent_score(content(parent, 1), content(parent, 1), original))
            assert score["correct"] is score["format_valid"] is True
            assert score["format_error"] is None and score["response_normalized"] == gold
            assert score["canonical_response_normalized"] == oracle["canonical_response_normalized"]
            if "Entity inventory: " in prompt:
                inventory = prompt.split("Entity inventory: ")[1].split("\n")[0].split()
                assert inventory == list(coordinate_mapping(variant).values())
                assert EMPTY_DEFAULT not in prompt
            else:
                assert prompt.count(EMPTY_DEFAULT) == 1
                board = prompt.split("Board: ")[1].split("\n")[0]
                source = content(parent).split("Board: ")[1].split("\n")[0]
                assert project_text(board, variant, reverse=True) == source
                facts = dict(clause.split(" ", 1) for clause in board.split("; "))
                assert sum(key.startswith("T(") for key in facts) == 19
                assert sum(key.startswith("P(") for key in facts) == 9 and "robber" in facts
                assert not any(key.startswith(("N(", "E(")) and value == "empty"
                               for key, value in facts.items())


def test_exact_new_unit_radicals_aliases_and_strict_complete_answers(
    panels: dict[str, list[dict[str, Any]]],
) -> None:
    point = coordinate_points()["<N01>"]
    assert decode("N(2*sqrt(3),2)", "scaled_h") == point
    assert decode("N(4*√3/2,2.0)", "scaled_h") == point
    assert decode("N(2h,2/1)", "scaled_h").x_root == Fraction(1, 2)
    assert decode("N(2/1,2.0)", "integer_xy") == point
    for variant, rows in panels.items():
        row = next(row for row in rows if row["task_type"] == "symbolic_neighbors")
        metadata, gold = object_map(row["metadata"]), content(row, 1)
        aliases = []
        for atom in gold.split():
            p = decode(atom, variant)
            k, y = 4 * p.x_root, 4 * p.y
            x = f"{k}*sqrt(3)" if variant == "scaled_h" else f"{k}/1"
            aliases.append(f"{p.family}({x},{y}.0)")
        for answer in ("\n".join(reversed(aliases)), gold + "<|im_end|> </s> <pad>"):
            score = object_map(score_response(gold, answer, metadata))
            assert score["correct"] is score["format_valid"] is True
            assert score["response_normalized"] == gold
        for answer in (gold + " " + aliases[0], "N(999,999)", "N(0,1)", "T(0,0)",
                       "<N00>", gold + " <N00>", "Answer: " + gold, gold + " explanation",
                       gold + "<|im_end|> extra", "N(0, 4)", "N(0,4.0000000001)",
                       "N(0,4/0)", "N(0,4e0)", "N(0+0,4)", ""):
            score = object_map(score_response(gold, answer, metadata))
            assert score["correct"] is score["format_valid"] is False, answer
            assert score["format_error"] and score["canonical_response_normalized"] is None
        for atom in ("N(h/2,2)", "N(2h,2)") if variant == "integer_xy" else ("N(h/2,2)", "N(2,2)"):
            with pytest.raises(ValueError):
                parent_response(atom, variant)
        for task in ("symbolic_piece_owner", "symbolic_direction"):
            scalar = next(r for r in rows if r["task_type"] == task and content(r, 1) != "NONE")
            answer, m = content(scalar, 1), object_map(scalar["metadata"])
            reference = object_map(object_map(m["shorthand_reference"])["metadata"])
            for response in (answer.swapcase(), "NONE", "none", "Yes", "NO"):
                actual = object_map(score_response(answer, response, m))
                original = object_map(parent_score(answer, response, reference))
                assert (actual["correct"], actual["format_valid"]) == (original["correct"], original["format_valid"])
    assert score_response("", "", {"schema": "unrelated"}) is None


def test_metadata_gold_identity_and_panel_mixing_are_bound(
    panels: dict[str, list[dict[str, Any]]],
) -> None:
    rows, other = panels["scaled_h"], panels["integer_xy"]
    m = object_map(rows[0]["metadata"])
    gold = text(m["answer"])
    for key, value in (("answer", "WRONG"), ("representation", "integer_xy"),
                       ("variant", "integer_xy"), ("variant", "unknown"), ("source_id", "other"),
                       ("mapping_sha256", "0" * 64), ("parent_sha256", "0" * 64)):
        with pytest.raises(ValueError):
            score_response(gold, gold, {**m, key: value})
    with pytest.raises(ValueError, match="gold"):
        score_response("WRONG", gold, m)
    reference = deepcopy(object_map(m["shorthand_reference"]))
    reference["metadata"] = {**object_map(reference["metadata"]), "answer": "WRONG"}
    with pytest.raises(ValueError):
        score_response(gold, gold, {**m, "shorthand_reference": reference})
    for invalid in (rows[:-1], [rows[1], rows[0], *rows[2:]], [rows[0], rows[0], *rows[2:]],
                    [other[0], *rows[1:]]):
        with pytest.raises(ValueError):
            validate_rows(invalid)
    changed = [{**rows[0], "messages": [{"role": "user", "content": "Answer yes"},
                                        {"role": "assistant", "content": gold}]}, *rows[1:]]
    with pytest.raises(ValueError, match="prompt/gold"):
        validate_rows(changed)


def test_exclusive_two_panel_builder_and_artifact_receipts(tmp_path: Path) -> None:
    destination = tmp_path / "panels"
    result = build_dataset(destination)
    assert result["rows"] == 400 and result["rows_per_variant"] == 200
    manifest = parse_json((destination / "manifest.json").read_text())
    assert object_map(manifest["parent"])["sha256"] == PARENT_SHA256
    intent = object_map(manifest["intent"])
    assert intent["model_run_performed"] is intent["finetuning"] is False
    assert intent["run_both_variants"] is True and manifest["added_tokenizer_tokens"] == []
    mapping = parse_json((destination / "mapping.json").read_text())
    assert digest(mapping) == manifest["mapping_sha256"]
    for variant in VARIANTS:
        rows = [parse_json(line) for line in (destination / f"{variant}.jsonl").read_text().splitlines()]
        assert len(rows) == 200 and rows[0]["schema"] == SCHEMA
        assert object_map(object_map(manifest["validation"])[variant])["gold_roundtrip_rows"] == 200
        assert digest(object_map(mapping["variants"])[variant]) == object_map(manifest["variant_mapping_sha256"])[variant]
    for name, receipt in object_map(result["files"]).items():
        assert sha256((destination / name).read_bytes()) == object_map(receipt)["sha256"]
    preview = (destination / "preview.md").read_text()
    assert preview.count("\n### symbolic_") == 16 and PARENT_SHA256 in preview
    with pytest.raises(FileExistsError):
        build_dataset(destination)
    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path / "missing" / "panels")
    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text(compact({"schema": SCHEMA}) + "\n")
    with pytest.raises(ValueError, match="parent source hash"):
        build_dataset(tmp_path / "rejected", source=corrupt)
    assert not (tmp_path / "rejected").exists()
