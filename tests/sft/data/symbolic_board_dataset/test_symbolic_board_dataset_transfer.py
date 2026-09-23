"""Transfer split selection, weighting, declarations and historical exposure."""

import copy
from pathlib import Path

import pytest

from data_pipeline.board_recognition.sources import file_sha256
from sft.board.symbolic_board_tasks import (
    TRANSFER_TASKS,
    atlas_geometry,
    decode_state,
    score_symbolic_task,
)
from sft.scripts.builders.build_symbolic_board_dataset import (
    known_direction_exposure,
    read_json,
    transfer_projection,
    validate_row_declarations,
)

from .support import BuiltDataset


def test_real_settlement_transfer_against_independent_complete_predicate(built_dataset: BuiltDataset) -> None:
    rows = built_dataset[-1]["transfer_test"]
    atlas = atlas_geometry()
    checked = 0
    for row in rows:
        if row["task_type"] != "symbolic_settlement_locations":
            continue
        target = row["metadata"]["target"]
        q = target["query"]
        if q["near"] is not None:
            continue
        data = decode_state(target["state"])
        occupied = set(data["buildings"])
        expected = set()
        for node, neighbors in atlas["graph"].items():
            has_road = any(data["roads"].get(edge) == q["color"]
                           for edge, endpoints in atlas["edges"].items() if node in endpoints)
            if node not in occupied and not neighbors & occupied and (q["phase"] == "setup" or has_road):
                expected.add(node)
        assert set(row["messages"][1]["content"].split()) - {"NONE"} == expected
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("level", ["row", "metadata", "both"])
@pytest.mark.parametrize("field,value", [
    ("task_role", "train"), ("training_family", "symbolic_owned_roads"),
    ("task_type", "symbolic_owned_roads"), ("split", "train"),
])
def test_all_transfer_declarations_fail_closed(built_dataset: BuiltDataset, level: str,
                                            field: str, value: str) -> None:
    row = copy.deepcopy(built_dataset[-1]["transfer_test"][0])
    if level in ("row", "both"):
        row[field] = value
    if level in ("metadata", "both"):
        row["metadata"][field] = value
    with pytest.raises(ValueError):
        validate_row_declarations(row, "transfer_test")


def test_missing_role_and_forged_transfer_weights_rejected(built_dataset: BuiltDataset) -> None:
    files = built_dataset[-1]
    row = copy.deepcopy(files["transfer_test"][0])
    row.pop("task_role")
    with pytest.raises(ValueError):
        validate_row_declarations(row, "transfer_test")
    row = copy.deepcopy(files["transfer_test"][0])
    row["metadata"]["task_role"] = "train"
    with pytest.raises(ValueError):
        score_symbolic_task("", row["messages"][1]["content"], row["metadata"])


def test_transfer_dedup_selection_weights_and_missing_hard_cases(built_dataset: BuiltDataset) -> None:
    metadata, files = built_dataset[-2:]
    for split in ("transfer_validation", "transfer_test"):
        rows = files[split]
        projections = [transfer_projection(r["task_type"], r["metadata"]["target"]["state"],
                                           r["metadata"]["target"]["query"]) for r in rows]
        assert len(projections) == len(set(projections))
        report = metadata["transfer_selection"][split]
        assert report["status"] == "transfer_pilot_limited"
        assert report["population_rows"] > report["unique_relevant_queries"] > report["selected_rows"]
        assert report["selected_rows"] == len(rows) < 1000
        assert not report["aggregation"]["pooled_micro_headline"]
        for group in report["groups"]:
            pos, neg = group["positive"], group["negative"]
            if not group["mode"].startswith("setup/"):
                assert pos["selected"] == pos["unique"]
            assert neg["selected"] == min(neg["unique"], pos["selected"])
        for task in TRANSFER_TASKS:
            selected = [r for r in rows if r["task_type"] == task]
            if selected:
                assert sum(r["metadata"]["macro_weight"] for r in selected) == pytest.approx(1)
        coverage = report["population_graph_cases"]
        assert coverage["states"] == 64
        assert set(coverage["missing_coverage"]) == {"cycle", "effective_blocker", "tied_max_ge5"}
        assert all(coverage["counts"][key] == 0 for key in coverage["missing_coverage"])
    diagnostic = metadata["reserved_color_diagnostic_coverage"]
    assert not diagnostic["included_in_transfer"] and diagnostic["graph_cases"]["states"] == 64
    reserved = {p["state_id"] for p in diagnostic["provenance"]}
    assert all(r["metadata"]["provenance"]["state_id"] not in reserved
               for split in ("transfer_validation", "transfer_test") for r in files[split])


def test_known_history_pair_union_filtered_and_v1_preserved(built_dataset: BuiltDataset) -> None:
    output, _, _, metadata, files = built_dataset
    history = known_direction_exposure()
    known = {tuple(k.split()) for k in history["pair_sources"]}
    assert known
    pairs = read_json(output / "directional_pairs.json")["splits"]
    assert known <= {tuple(p) for p in pairs["train"]}
    for split in ("validation", "test"):
        assert not known & {tuple(p) for p in pairs[split]}
        assert all(not r["metadata"]["known_training_exposure"] for r in files[split]
                   if "directional_pair" in r["metadata"])
    assert "New-corpus holdout only" in history["claim"]
    receipt = metadata["preserved_v1"]
    if receipt["status"] == "preserved":
        for info in receipt["files"].values():
            assert file_sha256(Path(info["path"])) == info["sha256"]
