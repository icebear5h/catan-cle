"""Measured polarity, rendered-roster leakage and conditional density support."""

import json
import re
from collections import Counter

from sft.board.symbolic_board_tasks import (
    STATIC_TASKS,
)

from .support import BuiltDataset


def test_real_routes_and_atomic_constraints_have_measured_polarity(built_dataset: BuiltDataset) -> None:
    files = built_dataset[-1]
    for split in ("train", "validation", "test"):
        rows = files[split]
        for task in ("symbolic_direction", "symbolic_near", "symbolic_local_constraint"):
            counts = Counter(r["messages"][1]["content"] for r in rows if r["task_type"] == task)
            assert counts["yes"] == counts["no"]
        routes = [json.loads(r["messages"][1]["content"]) for r in rows if r["task_type"] == "symbolic_shortest_route"]
        assert sum(r["nodes"] is None for r in routes) == len(routes) * 3 // 8
        assert sum(r["edges"] == [] for r in routes) == len(routes) // 8
        assert any(r["edges"] and len(r["edges"]) >= 2 for r in routes)


def test_rendered_roster_position_cannot_predict_atomic_labels(built_dataset: BuiltDataset) -> None:
    # Deliberately derive both queried participant and predicate from the rendered
    # question; a balanced metadata table cannot mask a leaky model presentation.
    for split in ("train", "validation", "test"):
        tables = {}
        for row in built_dataset[-1][split]:
            task = row["task_type"]
            if task not in ("symbolic_owned_incident_roads", "symbolic_local_constraint", "symbolic_owned_roads"):
                continue
            prompt, response = [m["content"] for m in row["messages"]]
            roster = re.search(r"Participants: ([A-Z_ ]+)\.\n", prompt).group(1).split()
            question = prompt.rsplit("\n", 1)[-1]
            mode = "all"
            if task == "symbolic_local_constraint":
                mode = ("has_owned_incident_road" if "incident road" in question else
                        "no_adjacent_building" if "edge-adjacent" in question else "empty")
            match = re.search(r"existing ([A-Z_]+) (?:incident road|road edges|road edge)", question)
            position = roster.index(match.group(1)) if match else None
            key = (task, mode, position)
            tables.setdefault(key, Counter())[response not in ("no", "NONE")] += 1
        for counts in tables.values():
            assert counts[True] == counts[False] > 0
        for task, mode in (("symbolic_owned_incident_roads", "all"),
                           ("symbolic_local_constraint", "has_owned_incident_road")):
            assert {p for t, m, p in tables if (t, m) == (task, mode)} == {0, 1, 2, 3}


def test_global_dynamic_variety_and_conditional_density_support(built_dataset: BuiltDataset) -> None:
    metadata, files = built_dataset[-2:]
    train = [r for r in files["train"] if r["task_type"] not in STATIC_TASKS]
    ids = {r["metadata"]["provenance"]["state_id"] for r in train}
    assert len(ids) >= 1000  # v1 reused only 306 states across 1600 presentations.
    assert len(ids) == metadata["component_sampling"]["train"]["unique_states"]
    roads = [r for r in train if r["task_type"] == "symbolic_owned_roads"]
    positives = [r for r in roads if r["messages"][1]["content"] != "NONE"]
    negatives = [r for r in roads if r["messages"][1]["content"] == "NONE"]
    assert {r["metadata"]["provenance"]["density_bin"] for r in positives} == {"setup", "sparse", "dense"}
    assert all(r["metadata"]["provenance"]["density_bin"] != "empty" for r in negatives)
    assert all(set(c["eligible_states_by_density"]) == {"empty", "setup", "sparse", "dense"}
               for c in metadata["component_sampling"]["train"]["conditional_support"])
    # Cross-tabs retain the actual source-density condition as well as the visible roster.
    assert all("density" in c for c in metadata["component_profiles"]["train"]["rendered_roster_crosstabs"])
