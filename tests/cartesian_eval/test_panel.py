"""Real-source panel, independent physical geometry, and strict answer contracts."""
from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from typing import Any, cast

import pytest

from sft.board.coordinate_comparison import score_coordinate_comparison
from sft.board.symbolic_board_tasks import atlas_geometry
from sft.cartesian_eval import SCHEMA, score_response, validate_rows
from sft.cartesian_eval.contracts import object_list, object_map, parse_json, text
from sft.cartesian_eval.dataset import DEFAULT_SOURCE, build_rows
from sft.cartesian_eval.geometry import (
    ATLAS_ATOM,
    canonical_response,
    coordinate_mapping,
    coordinate_points,
    parse_atom,
    project_text,
)
from sft.json_types import as_dict


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return build_rows()


def test_all_entities_have_exact_physical_geometry_and_unit_roads() -> None:
    # Typed boundary to the existing legacy geometry function, not a second oracle.
    atlas = cast(Callable[[], dict[str, Any]], atlas_geometry)()
    base = cast(dict[str, tuple[int, int]], atlas["positions"])
    edges = cast(dict[str, tuple[str, str]], atlas["edges"])
    points, mapping = coordinate_points(), coordinate_mapping()
    assert len(mapping) == len(set(mapping.values())) == 154
    assert Counter(p.family for p in points.values()) == {"N": 54, "T": 19, "E": 72, "P": 9}
    assert mapping["<T00>"] == "T(0,0)"
    assert mapping["<N00>"] == "N(0,1)"
    assert mapping["<N01>"] == "N(sqrt(3)/2,1/2)"
    assert mapping["<E00_01>"] == "E(sqrt(3)/4,3/4)"
    for token, (x, y_down) in base.items():
        assert points[token].x_root == Fraction(x, 2)
        assert points[token].y == Fraction(-y_down, 2)
    for edge, (a, b) in edges.items():
        p, q, midpoint = points[a], points[b], points[edge]
        assert 3 * (p.x_root - q.x_root) ** 2 + (p.y - q.y) ** 2 == Fraction(1)
        assert midpoint.x_root == (p.x_root + q.x_root) / 2
        assert midpoint.y == (p.y + q.y) / 2
    ports = object_list(object_map(atlas["raw"])["ports"])
    for value in ports:
        port = object_map(value)
        first, second = (points[f"<N{nid:02d}>"] for nid in cast(list[int], port["attached_nodes"]))
        p = points[text(port["token"])]
        assert (p.x_root, p.y) == ((first.x_root + second.x_root) / 2, (first.y + second.y) / 2)
    assert all(parse_atom(mapping[t]) == p for t, p in points.items())
    mapping.clear()
    points.clear()
    assert len(coordinate_mapping()) == len(coordinate_points()) == 154


def test_exact_200_reference_cases_prompts_oracles_and_axis_orientation(
    rows: list[dict[str, Any]],
) -> None:
    source = [parse_json(line) for line in DEFAULT_SOURCE.read_text().splitlines()][::2]
    report = validate_rows(rows)
    assert report["rows"] == report["gold_roundtrip_rows"] == 200
    assert report["by_split"] == {"test": 198, "validation": 2}
    assert report["by_representation"] == {"cartesian": 200}
    assert report["by_family"] == {"direction": 64, "adjacency": 32, "incidence": 16, "ownership": 88}
    points = coordinate_points()
    equal_axis = 0
    directions: set[str] = set()
    for row, original in zip(rows, source, strict=True):
        m, atlas = object_map(row["metadata"]), object_map(original["metadata"])
        assert object_map(m["atlas_reference"])["metadata"] == atlas
        assert row["id"] == f"cartesian_eval_v2/{atlas['source_id']}"
        assert all(row[k] == original[k] for k in ("task_type", "split", "task_role", "training_family"))
        target = object_map(atlas["target"])
        prompt = text(object_map(object_list(row["messages"])[0])["content"])
        gold = text(m["answer"])
        assert ATLAS_ATOM.search(prompt + gold) is None
        assert all(s not in prompt for s in ("learned", "atlas", "canonical_answer", "crosswalk"))
        if target["state"] is None:
            inventory = prompt.split("Entity inventory: ", 1)[1].split("\n", 1)[0]
            assert inventory.split() == list(coordinate_mapping().values())
        else:
            assert "Entity inventory:" not in prompt
            board = prompt.split("Board: ", 1)[1].split("\n", 1)[0]
            assert board == project_text(text(object_map(target["state"])["board"]))
            assert len(board.split(";")) == 155
        score = object_map(score_response(gold, gold, m))
        delegated = object_map(score_coordinate_comparison(
            text(atlas["answer"]), canonical_response(gold), as_dict(atlas),
        ))
        assert score["correct"] is score["format_valid"] is True
        assert delegated["correct"] is delegated["format_valid"] is True
        assert score["scoring"] == SCHEMA
        assert score["canonical_expected_normalized"] == atlas["canonical_answer"]
        if row["task_type"] == "symbolic_direction":
            q = object_map(target["query"])
            a, b = points[text(q["a"])], points[text(q["b"])]
            direction = text(q["direction"])
            directions.add(direction)
            answer = {"left": a.x_root < b.x_root, "right": a.x_root > b.x_root,
                      "above": a.y > b.y, "below": a.y < b.y}[direction]
            assert gold == ("yes" if answer else "no")
            tied = a.x_root == b.x_root if direction in {"left", "right"} else a.y == b.y
            equal_axis += int(tied)
            assert not tied or gold == "no"
    assert equal_axis > 0 and directions == {"left", "right", "above", "below"}


def _alias(atom: str) -> str:
    p = parse_atom(atom)
    x, y = p.x_root, p.y
    return (f"{p.family}({x.numerator * 2}/{x.denominator * 2}*√3,"
            f"{y.numerator * 2}/{y.denominator * 2})")


def test_equivalent_exact_spellings_and_complete_response_rejection(
    rows: list[dict[str, Any]],
) -> None:
    point = parse_atom("N(sqrt(3)/2,1/2)")
    for atom in ("N(√3/2,2/4)", "N(1/2*sqrt(3),3/6)", "N(2*sqrt(3)/4,1/2)"):
        assert parse_atom(atom) == point
    row = next(r for r in rows if r["task_type"] == "symbolic_neighbors")
    m = object_map(row["metadata"])
    gold = text(m["answer"])
    aliases = [_alias(atom) for atom in gold.split()]
    equivalent = "\n".join(reversed(aliases))
    for response in (equivalent, gold.replace("sqrt(3)", "√3"), gold + "<|im_end|> </s> <pad>"):
        score = object_map(score_response(gold, response, m))
        assert score["correct"] is score["format_valid"] is True
        assert score["response_normalized"] == gold
    bad = (gold + " " + aliases[0], "N(0.8660254,1/2)", "N(1,1/2)", "N(sqrt(12)/4,1/2)",
           "N(sqrt(3)/0,1/2)", "N(999*sqrt(3),999)", "T(0,0)", "<N00>",
           gold + " <N00>", gold + " explanation", "Answer: " + gold,
           gold + "<|im_end|> extra", "<|im_end|>" + gold, "N(0, 1)", "")
    for response in bad:
        score = object_map(score_response(gold, response, m))
        assert score["correct"] is score["format_valid"] is False, response
    wrong = object_map(score_response(gold, "NONE", m))
    assert wrong["format_valid"] is True and wrong["correct"] is False
    scalar = next(object_map(r["metadata"]) for r in rows if r["task_type"] == "symbolic_direction")
    expected = text(scalar["answer"])
    assert object_map(score_response(expected, "no" if expected == "yes" else "yes", scalar))["correct"] is False
    assert object_map(score_response(expected, expected + " yes", scalar))["format_valid"] is False
    assert score_response("", "", {"schema": "unrelated"}) is None
