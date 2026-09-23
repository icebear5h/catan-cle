"""Real corpus quotas, splits, labels, and launcher schema."""
import copy
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

from cle.game_engine.models.enums import RESOURCES
from data_pipeline.board_recognition.full_board_readout import board_answer
from data_pipeline.board_recognition.replay_dataset import read_jsonl
from data_pipeline.board_recognition.sources import file_sha256
from evals.catan_board_bench.tokens import atlas_tokens
from sft.board.spatial_tasks import (
    atlas_node_graph,
    dice_production,
    local_node_tiles,
    node_tile_tokens,
    shortest_node_path,
)
from sft.scripts.builders.build_spatial_continuation_dataset import (
    BANK_QUOTAS,
    DEFAULT_ROOT,
    FROZEN_BOARD_EVAL,
    NEW_PANELS,
    STEP_QUOTAS,
    build_dataset,
)
from sft.scripts.eval.eval_qwen_vl_adapter import score_response
from sft.scripts.train.train_trl_catan_vision import _message_pair

from .support import DatasetInputs, JsonDict


def test_real_corpus_exact_steps_quotas_and_uniform_stage_omission(local_dataset: tuple[Path, DatasetInputs, JsonDict, list[JsonDict], dict[str, list[JsonDict]]]) -> None:
    _, _, metadata, train, panels = local_dataset
    assert len(train) == 1024
    expected = {k: v * 8 for k, v in STEP_QUOTAS.items()}
    assert Counter(r["training_family"] for r in train) == expected == metadata["row_quotas"]
    batches = [train[i : i + 8] for i in range(0, len(train), 8)]
    assert len(batches) == len(metadata["step_order"]) == 128
    for step, batch in enumerate(batches):
        assert len({r["training_family"] for r in batch}) == 1
        assert all(r["metadata"]["optimizer_step"] == step for r in batch)
        assert metadata["step_order"][step]["row_ids"] == [r["row_id"] for r in batch]
    for i in range(0, 128, 8):
        assert Counter(batch[0]["training_family"] for batch in batches[i : i + 8]) == {
            family: 2 if family == "full_board_readout" else 1 for family in STEP_QUOTAS
        }
        assert len({r["metadata"]["block_id"] for batch in batches[i : i + 8] for r in batch}) == 1
    all_rows = train + [r for rows in panels.values() for r in rows]
    assert len({r["id"] for r in all_rows}) == len({r["row_id"] for r in all_rows}) == 1270
    assert all(
        r["id"] == r["row_id"] and r["row_id"].startswith("spatial_continuation_v1/")
        for r in all_rows
    )
    assert all("curriculum_stage" not in r for r in all_rows)
    assert all(r["task_type"] == r["metadata"]["task_type"] for r in all_rows)
    assert Counter(r["task_type"] for r in train if r["task_type"] in BANK_QUOTAS) == BANK_QUOTAS
    assert metadata["global_row_shuffle"] is False


def test_real_corpus_splits_path_reversals_and_panel_coverage(local_dataset: tuple[Path, DatasetInputs, JsonDict, list[JsonDict], dict[str, list[JsonDict]]]) -> None:
    _, _, metadata, train, panels = local_dataset
    manifests = read_jsonl(DEFAULT_ROOT / "manifest.jsonl")
    heldout_maps = {s["board_map_sha256"] for s in manifests if s["split"] != "train"}
    assert not {r["metadata"]["board_map_sha256"] for r in train} & heldout_maps
    validation_ids = {s["sample_id"] for s in manifests if s["split"] == "validation"}
    assert {r["state_id"] for r in read_jsonl(FROZEN_BOARD_EVAL)} == validation_ids
    assert not {r["images"][0] for r in train} & {
        r["images"][0] for rows in panels.values() for r in rows
    }
    assert {k: len(v) for k, v in panels.items()} == dict(zip(NEW_PANELS, (54, 64, 64, 64)))
    for task, rows in panels.items():
        assert {r["metadata"]["state_id"] for r in rows} <= validation_ids
        assert len({r["images"][0] for r in rows}) == len(rows)
        assert len({r["metadata"]["board_map_sha256"] for r in rows}) == 5
    assert {r["metadata"]["target"]["node"] for r in panels["node_tiles"]} == set(
        atlas_node_graph()
    )
    assert (
        sorted(
            Counter(
                r["metadata"]["target"]["node"] for r in train if r["task_type"] == "node_tiles"
            ).values()
        )
        == [2] * 34 + [3] * 20
    )
    path_splits = metadata["splits"]["path_pairs"]
    pools = {s: {tuple(p) for p in data["pairs"]} for s, data in path_splits.items()}
    for left, right in combinations(pools.values(), 2):
        assert not left & right
    for split, rows in (("train", train), ("validation", panels["shortest_node_path"])):
        selected = set()
        for row in rows:
            if row["task_type"] != "shortest_node_path":
                continue
            target = row["metadata"]["target"]
            pair = tuple(sorted((target["end"], target["start"])))
            assert pair in pools[split] and pair not in selected
            selected.add(pair)
        assert selected == {tuple(p) for p in path_splits[split]["selected_pairs"]}
        assert set(path_splits[split]["selected_by_distance"]) == {str(i) for i in range(1, 12)}
        assert 0 < path_splits[split]["selected_tied"] < len(selected)
    assert metadata["coverage"]["train"]["unique_images"] == 1024
    assert metadata["coverage"]["all_unique_images"] == 1088
    retention = metadata["coverage"]["training_families"]["full_board_readout"]
    assert retention["by_density"] == dict.fromkeys(("empty", "setup", "sparse", "dense"), 64)
    assert retention["unique_board_maps"] >= 200


def test_real_corpus_labels_protocol_readout_retention_and_scorer(local_dataset: tuple[Path, DatasetInputs, JsonDict, list[JsonDict], dict[str, list[JsonDict]]]) -> None:
    _, _, _, train, panels = local_dataset
    sources = {
        r["state_id"]: r
        for r in read_jsonl(DEFAULT_ROOT / "full_board_readout_v1/stage1/train.jsonl")
    }
    contracts = {}
    for row in train + [r for rows in panels.values() for r in rows]:
        task, metadata = row["task_type"], row["metadata"]
        sid = metadata["state_id"]
        if sid not in contracts:
            contracts[sid] = json.loads(
                Path(metadata["provenance"]["paths"]["contract"]).read_text()
            )
        contract = contracts[sid]
        prompt, answer = row["messages"][0]["content"], row["messages"][1]["content"]
        assert prompt.count("<image>") == 1
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
        assert score_response(answer, answer, metadata=metadata)["correct"]
        assert _message_pair(row, line_number=1)[1] == answer
        assert not any(word in prompt for word in ("board_map_sha256", "contract_path", "state_id"))
        if task == "full_board_readout":
            assert row["messages"] == sources[sid]["messages"]
            assert row["images"] == sources[sid]["images"]
            assert answer == board_answer(contract)
            continue
        target = metadata["target"]
        assert "no explanation" in prompt.lower()
        if task in ("node_tiles", "local_node_tiles"):
            assert set(target) == {"node"}
            assert re.findall(r"<[NT]\d+>", prompt) == [target["node"]]
        if task == "node_tiles":
            assert answer.split() == node_tile_tokens(**target)
        elif task == "shortest_node_path":
            assert set(target) == {"start", "end"}
            assert answer.split() == shortest_node_path(**target)
            assert re.findall(r"<N\d+>", prompt) == [target["start"], target["end"]]
            assert "ignoring all buildings, roads, and the robber" in prompt
        elif task == "local_node_tiles":
            assert json.loads(answer) == local_node_tiles(contract, **target)
            assert answer not in prompt
        elif task == "dice_production":
            assert set(target) == {"color", "roll"}
            assert target["color"] in {p["color"] for p in contract["players"]}
            assert str(target["roll"]) in prompt
            assert target["color"].lower().replace("_", " ") in prompt
            label = json.loads(answer)
            assert label == dice_production(contract, **target)
            assert set(label) == {r.lower() for r in RESOURCES}
            assert all(type(n) is int for n in label.values())
            assert answer not in prompt
            facts = metadata["production"]
            assert facts["zero"] == (sum(label.values()) == 0)
            downgraded = copy.deepcopy(contract)
            for node in downgraded["nodes"]:
                if node["color"] == target["color"] and node["building"] == "CITY":
                    node["building"] = "SETTLEMENT"
            assert (dice_production(downgraded, **target) != label) == facts["city_relevant"]
            unblocked = copy.deepcopy(contract)
            desert = next(t for t in unblocked["tiles"] if t["resource"] is None)
            for tile in unblocked["tiles"]:
                tile["has_robber"] = tile["id"] == desert["id"]
            unblocked["robber"] = {
                "tile_id": desert["id"],
                "tile_token": desert["token"],
                "coord": desert["coord"],
            }
            assert (dice_production(unblocked, **target) != label) == facts["robber_relevant"]
    for rows, count in (
        ([r for r in train if r["task_type"] == "dice_production"], 128),
        (panels["dice_production"], 64),
    ):
        assert Counter(r["metadata"]["production"]["zero"] for r in rows) == {
            True: count // 2,
            False: count // 2,
        }
        for field in ("city_relevant", "robber_relevant"):
            assert sum(r["metadata"]["production"][field] for r in rows) >= count // 8


def test_real_corpus_launcher_schema_hashes_and_determinism(local_dataset: tuple[Path, DatasetInputs, JsonDict, list[JsonDict], dict[str, list[JsonDict]]]) -> None:
    parent, inputs, metadata, _, _ = local_dataset
    assert set(inputs) == {
        "schema",
        "train_jsonl",
        "image_root",
        "token_inventory",
        "new_panels",
        "metadata",
    }
    assert inputs["schema"] == "catan_spatial_continuation_inputs/v1"
    assert set(inputs["new_panels"]) == set(NEW_PANELS)
    for spec in inputs["new_panels"].values():
        assert set(spec) == {"eval_jsonl", "image_root", "max_new_tokens", "batch_size"}
        assert spec["image_root"] == inputs["image_root"]
        assert spec["max_new_tokens"] == 128 and spec["batch_size"] == 16
        assert Path(spec["eval_jsonl"]).is_absolute()
    for key in ("image_root", "train_jsonl", "token_inventory", "metadata"):
        assert Path(inputs[key]).is_absolute() and Path(inputs[key]).exists()
    assert set(atlas_tokens()) <= set(
        json.loads(Path(inputs["token_inventory"]).read_text())["tokens"]
    )
    for source in metadata["source_hashes"].values():
        assert file_sha256(Path(source["path"])) == source["sha256"]
    for info in metadata["files"].values():
        assert file_sha256(parent / "first" / info["path"]) == info["sha256"]
    for asset in metadata["selected_assets"]:
        for kind, path in asset["paths"].items():
            assert file_sha256(Path(path)) == asset["sha256"][kind]
        assert file_sha256(Path(asset["common_image"])) == asset["sha256"]["image"]
    second = build_dataset(parent / "second")
    second_metadata = json.loads(Path(second["metadata"]).read_text())
    assert metadata == second_metadata
    assert file_sha256(Path(inputs["train_jsonl"])) == file_sha256(Path(second["train_jsonl"]))
    for label in NEW_PANELS:
        assert file_sha256(Path(inputs["new_panels"][label]["eval_jsonl"])) == file_sha256(
            Path(second["new_panels"][label]["eval_jsonl"])
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_dataset(parent / "first")
