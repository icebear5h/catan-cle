import json
from pathlib import Path

import pytest

from data_pipeline.board_recognition.terrain_readout import (
    READOUT_PROMPT,
    export_terrain_readout,
    layout_id,
    port_answer,
    readout_answer,
    rows_for_state,
    terrain_facts,
)
from data_pipeline.board_recognition.spatial_localization import SpatialLocalizationError


FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")


def _fixture_state(sample_id: str = "empty_setup_node_p000_base"):
    manifest = [json.loads(line) for line in (FIXTURE_ROOT / "manifest.jsonl").read_text().splitlines()]
    state = next(row for row in manifest if row["sample_id"] == sample_id)
    return state, json.loads((FIXTURE_ROOT / state["contract_path"]).read_text())


def test_layout_and_port_answers():
    assert layout_id("r_replay_189649315_s000000") == "r_replay_189649315"
    assert layout_id("r_replay_189649315_s000173") == "r_replay_189649315"
    assert layout_id("empty_setup_node_p000_base") == "empty_setup_node_p000_base"
    assert port_answer({"kind": "generic", "ratio": "3:1", "resource": None}) == "3:1 port"
    assert port_answer({"kind": "resource", "ratio": "2:1", "resource": "SHEEP"}) == "sheep port"


def test_rows_cover_every_tile_and_port_with_desert_and_duplicates():
    state, contract = _fixture_state()
    tiles, ports = terrain_facts(contract)
    rows = rows_for_state({**state, "split": "validation"}, contract)

    assert len(rows) == 19 * 2 + 9 + 1
    prompts = {row["messages"][0]["content"].replace("<image>\n", ""): row["messages"][1]["content"] for row in rows}
    assert sum(1 for prompt in prompts if prompt.endswith("resource?")) == 19
    assert sum(1 for prompt in prompts if prompt.endswith("number?")) == 19
    assert sum(1 for prompt in prompts if prompt.endswith("port?")) == 9
    desert = next(tile for tile in tiles if tile["resource"] == "desert")
    assert prompts[f"{desert['token']} resource?"] == "desert" and prompts[f"{desert['token']} number?"] == "none"
    assert all(prompts[f"{port['token']} port?"].endswith(" port") for port in ports)
    assert {row["task_type"] for row in rows} == {"tile_resource", "tile_number", "port_type", "terrain_readout"}
    readout = next(row for row in rows if row["task_type"] == "terrain_readout")
    assert readout["messages"][0]["content"] == f"<image>\n{READOUT_PROMPT}"
    answer = readout["messages"][1]["content"]
    assert answer == readout_answer(tiles, ports)
    parts = answer.split("; ")
    assert len(parts) == 28 and parts[0].startswith("<T00> ") and parts[-1].startswith("<P08> ")
    assert all(row["schema"] == "catan_terrain_readout_row/v1" and row["layout_id"] == state["sample_id"] and "spatial_targets" not in row for row in rows)
    assert len({row["row_id"] for row in rows}) == len(rows)


def test_readout_answer_keeps_duplicate_tiles_distinct():
    tiles = [{"token": "<T00>", "resource": "wheat", "number": "8"}, {"token": "<T01>", "resource": "wheat", "number": "8"}]
    ports = [{"token": "<P00>", "answer": "3:1 port"}]
    assert readout_answer(tiles, ports) == "<T00> wheat 8; <T01> wheat 8; <P00> 3:1 port"
    with pytest.raises(SpatialLocalizationError):
        terrain_facts({"tiles": [], "ports": []})


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    root.rmdir()


def test_export_splits_by_layout():
    output = FIXTURE_ROOT / "terrain_readout_test_output"
    _remove_tree(output)
    try:
        metadata = export_terrain_readout(FIXTURE_ROOT, output_dir=output, overwrite=True, validate_dataset=False)
        assert metadata["rows_per_image"] == 48
        assert set(metadata["layouts_by_split"]) == {"train", "validation", "test"}
        train = [json.loads(line) for line in (output / "stage1" / "train.jsonl").read_text().splitlines()]
        validation = [json.loads(line) for line in (output / "stage1" / "validation.jsonl").read_text().splitlines()]
        assert train and all(row["split"] == "train" for row in train)
        assert not ({row["layout_id"] for row in train} & {row["layout_id"] for row in validation})
        assert all((FIXTURE_ROOT / "images" / row["images"][0]).is_file() for row in train[:20])
    finally:
        _remove_tree(output)
