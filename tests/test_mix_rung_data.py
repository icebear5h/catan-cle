import json
from collections import Counter
from pathlib import Path

import pytest

from data_pipeline.board_recognition.mix_rung_data import apportion, draw_cells, estimate_tokens, export_mixed_rung
from data_pipeline.board_recognition.node_edge_readout import export_node_edge_readout
from data_pipeline.board_recognition.spatial_localization import SpatialLocalizationError
from data_pipeline.board_recognition.terrain_readout import export_terrain_readout


FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    root.rmdir()


def test_apportion_and_token_estimate():
    assert apportion({"a": 1, "b": 1, "c": 1}, 10) == {"a": 4, "b": 3, "c": 3}
    assert sum(apportion({"x": 0.5, "y": 0.25, "z": 0.25}, 7).values()) == 7
    assert estimate_tokens("red road") == 3
    assert estimate_tokens("<N00> empty; <N01> red settlement") == 2 + 4 + 1  # tokens, words and separator, end
    with pytest.raises(SpatialLocalizationError):
        apportion({"a": 0}, 3)


def test_mixed_rung_draws_quotas_by_density_and_balance(tmp_path: Path):
    pieces = FIXTURE_ROOT / "mix_test_pieces"
    terrain = FIXTURE_ROOT / "mix_test_terrain"
    _remove_tree(pieces)
    _remove_tree(terrain)
    try:
        export_node_edge_readout(FIXTURE_ROOT, output_dir=pieces, overwrite=True, validate_dataset=False, train_full_coverage=True)
        export_terrain_readout(FIXTURE_ROOT, output_dir=terrain, overwrite=True, validate_dataset=False)
        recipe = {
            "sources": {"pieces": str(pieces), "terrain": str(terrain)},
            "groups": [
                {"name": "piece_occupied", "source": "pieces", "categories": ["node.occupancy", "edge.owner"], "polarity": "positive",
                 "rows": 60, "density": {"setup": 1, "sparse": 1, "dense": 1}, "balance": ["piece", "color"],
                 "piece_shares": {"ROAD": 0.5, "SETTLEMENT": 0.25, "CITY": 0.25}},
                {"name": "piece_empty", "source": "pieces", "categories": ["node.occupancy", "edge.owner"], "polarity": "hard_negative",
                 "rows": 40, "density": {"empty": 0.1, "setup": 1, "sparse": 1, "dense": 1}, "balance": ["entity_type", "negative_kind"],
                 "negative_kind_shares": {"adjacent": 0.5, "cross_type": 0.15, "hop2": 0.1, "hop3": 0.05, "far": 0.2}},
                {"name": "terrain_short", "source": "terrain", "categories": ["tile.resource", "tile.number", "port.port_type"],
                 "rows": 30, "balance": ["category"], "category_shares": {"tile.resource": 0.4, "tile.number": 0.4, "port.port_type": 0.2}},
                {"name": "terrain_readout", "source": "terrain", "categories": ["terrain.readout"], "rows": 4},
                {"name": "node_readout", "source": "pieces", "categories": ["node.readout"], "rows": 3},
                {"name": "edge_readout", "source": "pieces", "categories": ["edge.readout"], "rows": 3},
            ],
            "eval_sample": [{"source": "pieces", "split": "validation", "every": 4}, {"source": "terrain", "split": "validation", "every": 3}],
        }
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(recipe))
        output = tmp_path / "mixed"
        metadata = export_mixed_rung(recipe_path, output)
        train = [json.loads(line) for line in (output / "stage1" / "train.jsonl").read_text().splitlines()]
        assert metadata["rows"] == len(train) == 60 + 40 + 30 + 4 + 3 + 3
        groups = metadata["groups"]
        assert groups["piece_occupied"]["taken"] == 60 and groups["piece_occupied"]["shortfall"] == 0
        assert set(groups["piece_occupied"]["by_density"]) <= {"setup", "sparse", "dense"}
        assert groups["piece_occupied"]["by_piece"]["ROAD"] >= groups["piece_occupied"]["by_piece"]["SETTLEMENT"]
        assert all(row["polarity"] == "positive" for row in train if row["mix_group"] == "piece_occupied")
        assert all(row["messages"][1]["content"] == "empty" for row in train if row["mix_group"] == "piece_empty")
        kinds = Counter(row["negative_kind"] for row in train if row["mix_group"] == "piece_empty")
        assert kinds["adjacent"] >= kinds["far"] >= kinds["hop3"]
        assert Counter(row["category"] for row in train if row["mix_group"] == "terrain_short")["port.port_type"] <= 8
        assert len({row["row_id"] for row in train}) == len(train)
        assert all((output / "images" / row["images"][0]).is_file() for row in train)
        evaluation = [json.loads(line) for line in (output / "stage1" / "eval_sample.jsonl").read_text().splitlines()]
        assert evaluation and {row["category"] for row in evaluation} >= {"node.occupancy", "tile.resource"}
        assert abs(sum(group["estimated_token_share"] for group in groups.values()) - 1.0) < 1e-6
        again = export_mixed_rung(recipe_path, tmp_path / "mixed_again")
        again_train = [json.loads(line) for line in (tmp_path / "mixed_again" / "stage1" / "train.jsonl").read_text().splitlines()]
        assert [row["row_id"] for row in again_train] == [row["row_id"] for row in train] and again["rows"] == metadata["rows"]
    finally:
        _remove_tree(pieces)
        _remove_tree(terrain)


def test_draw_cells_oversamples_short_cells_up_to_max_repeats():
    pool = [{"row_id": f"r{i}", "color": "WHITE" if i < 2 else "RED"} for i in range(6)]
    picked, remainder = draw_cells(pool, {("WHITE",): 5, ("RED",): 2}, ["color"], "salt", max_repeats=3)
    whites = [row for row in picked if row["color"] == "WHITE"]
    assert len(whites) == 5 and sorted(row.get("mix_repeat", 0) for row in whites) == [0, 0, 1, 1, 2]
    assert len({row["row_id"] for row in picked}) == len(picked)
    assert len([row for row in picked if row["color"] == "RED"]) == 2 and len(remainder) == 2
    once, _ = draw_cells(pool, {("WHITE",): 5}, ["color"], "salt")
    assert len(once) == 2
