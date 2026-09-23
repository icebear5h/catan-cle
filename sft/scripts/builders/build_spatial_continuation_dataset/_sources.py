from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from data_pipeline.board_recognition.full_board_readout import board_answer
from data_pipeline.board_recognition.replay_dataset import (
    dense_labels,
    read_jsonl,
    static_board_facts,
)
from data_pipeline.board_recognition.sources import (
    canonical_sha256,
    file_sha256,
    validate_public_board_contract,
    visible_board_facts,
)
from evals.catan_board_bench.tokens import atlas_tokens
from sft.json_types import JsonDict, as_dict, as_list, as_str, loads_json

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROOT = PROJECT_ROOT / "artifacts/generated/board_recognition/full_board_diverse_v1"
DEFAULT_OUTPUT = DEFAULT_ROOT.parent / "spatial_continuation_v1"
FROZEN_BOARD_EVAL = (
    PROJECT_ROOT / "artifacts/runs/sft/full-board-epoch2-20260907/validation64.jsonl"
)
VERSION = "spatial_continuation_v1"
NEW_PANELS = ("node_tiles", "shortest_node_path", "local_node_tiles", "dice_production")
STEP_QUOTAS = {
    "directions": 16,
    "adjacency_connectivity": 16,
    "node_tiles": 16,
    "shortest_node_path": 16,
    "local_node_tiles": 16,
    "dice_production": 16,
    "full_board_readout": 32,
}
BANK_QUOTAS = {
    "node_direction_yes": 20,
    "node_direction_no": 20,
    "node_direction_token": 24,
    "tile_direction_yes": 20,
    "tile_direction_no": 20,
    "tile_direction_token": 24,
    "node_adjacent_yes": 22,
    "node_adjacent_no": 22,
    "node_connected_yes": 20,
    "node_connected_no": 20,
    "tile_adjacent_yes": 22,
    "tile_adjacent_no": 22,
}
STATIC_GUIDANCE = "Use the fixed Catan atlas, ignoring all buildings, roads, and the robber. "


@dataclass
class _SourceContext:
    root: Path
    readout_root: Path
    image_root: Path
    inventory_path: Path
    manifest_rows: list[JsonDict]
    manifest: dict[str, JsonDict]
    sources: dict[str, list[JsonDict]]
    train_sources: list[JsonDict]
    split_maps: dict[str, set[str]]
    inventory: JsonDict
    source_paths: dict[str, Path]
    source_hashes: JsonDict


def _prepare_sources(root: Path) -> _SourceContext:
    readout_root = root / "full_board_readout_v1"
    image_root = readout_root / "images"
    inventory_path = readout_root / "trainable_tokens.json"
    source_paths = {
        "manifest": root / "manifest.jsonl",
        "token_inventory": inventory_path,
        "frozen_board_eval": FROZEN_BOARD_EVAL,
        "builder": PROJECT_ROOT
        / "sft/scripts/builders/build_spatial_continuation_dataset/__init__.py",
        "builder_sources": PROJECT_ROOT
        / "sft/scripts/builders/build_spatial_continuation_dataset/_sources.py",
        "builder_queries": PROJECT_ROOT
        / "sft/scripts/builders/build_spatial_continuation_dataset/_queries.py",
        "builder_production": PROJECT_ROOT
        / "sft/scripts/builders/build_spatial_continuation_dataset/_production.py",
        "builder_outputs": PROJECT_ROOT
        / "sft/scripts/builders/build_spatial_continuation_dataset/_outputs.py",
        "builder_build": PROJECT_ROOT
        / "sft/scripts/builders/build_spatial_continuation_dataset/_build.py",
        "spatial_tasks": PROJECT_ROOT / "sft/board/spatial_tasks/__init__.py",
        "spatial_tasks_topology": PROJECT_ROOT / "sft/board/spatial_tasks/_topology.py",
        "spatial_tasks_contracts": PROJECT_ROOT / "sft/board/spatial_tasks/_contracts.py",
        "spatial_tasks_scoring": PROJECT_ROOT / "sft/board/spatial_tasks/_scoring.py",
        "spatial_query_bank": PROJECT_ROOT / "data_pipeline/board_recognition/spatial_robber.py",
    }
    for split in ("train", "validation", "test"):
        source_paths[split] = readout_root / "stage1" / f"{split}.jsonl"
    source_hashes: JsonDict = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in source_paths.items()
    }
    inventory = as_dict(loads_json(inventory_path.read_text()))
    if not set(atlas_tokens()) <= set(as_list(inventory["tokens"])):
        raise ValueError("trained token inventory is missing literal atlas tokens")
    manifest_rows = read_jsonl(source_paths["manifest"])
    manifest = {as_str(s["sample_id"]): s for s in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate manifest sample_id")
    sources: dict[str, list[JsonDict]] = {}
    for split in ("train", "validation", "test"):
        rows = read_jsonl(source_paths[split])
        if len({r["state_id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate readout state_id: {split}")
        for row in rows:
            state = manifest[as_str(row["state_id"])]
            if state["split"] != split or row["split"] != split:
                raise ValueError(f"readout/manifest split mismatch: {row['state_id']}")
            if row["task_type"] != "full_board_readout":
                raise ValueError("source row is not a full-board readout")
            if row["images"] != [Path(as_str(state["image_path"])).name]:
                raise ValueError("readout image does not match manifest basename")
            row["board_map_sha256"] = state["board_map_sha256"]
        sources[split] = rows
    frozen = read_jsonl(FROZEN_BOARD_EVAL)
    validation = {row["state_id"]: row for row in sources["validation"]}
    if len(frozen) != 64 or {r["state_id"] for r in frozen} != set(validation):
        raise ValueError("frozen board eval must identify the 64 validation states")
    for row in frozen:
        source = validation[row["state_id"]]
        if row["images"] != source["images"] or row["messages"] != source["messages"]:
            raise ValueError("frozen board eval differs from validation source")
    split_maps = {
        split: {as_str(s["board_map_sha256"]) for s in manifest_rows if s["split"] == split}
        for split in ("train", "validation", "test", "color_diagnostic")
    }
    if split_maps["validation"] & split_maps["test"]:
        raise ValueError("validation/test board-map leakage")
    excluded_maps = set.union(*(maps for split, maps in split_maps.items() if split != "train"))
    train_sources = [r for r in sources["train"] if r["board_map_sha256"] not in excluded_maps]
    if len(train_sources) < 1024:
        raise ValueError("insufficient distinct non-held-out training images")
    return _SourceContext(
        root=root,
        readout_root=readout_root,
        image_root=image_root,
        inventory_path=inventory_path,
        manifest_rows=manifest_rows,
        manifest=manifest,
        sources=sources,
        train_sources=train_sources,
        split_maps=split_maps,
        inventory=inventory,
        source_paths=source_paths,
        source_hashes=source_hashes,
    )


def _make_loader(
    ctx: _SourceContext,
) -> tuple[Callable[[JsonDict], JsonDict], dict[str, JsonDict]]:
    contracts: dict[str, JsonDict] = {}
    verified: dict[str, JsonDict] = {}

    def load_contract(row: JsonDict) -> JsonDict:
        sid = as_str(row["state_id"])
        if sid in contracts:
            return contracts[sid]
        state = ctx.manifest[sid]
        paths = {
            "contract": ctx.root / as_str(state["contract_path"]),
            "image": ctx.root / as_str(state["image_path"]),
            "labels": ctx.root / as_str(state["label_path"]),
        }
        hashes: JsonDict = {kind: file_sha256(path) for kind, path in paths.items()}
        if hashes != state["sha256"]:
            raise ValueError(f"manifest asset hash mismatch: {sid}")
        common_image = ctx.image_root / as_str(as_list(row["images"])[0])
        if file_sha256(common_image) != hashes["image"]:
            raise ValueError(f"common-root image hash mismatch: {sid}")
        contract = as_dict(loads_json(paths["contract"].read_text()))
        validate_public_board_contract(contract)
        if canonical_sha256(static_board_facts(contract)) != state["board_map_sha256"]:
            raise ValueError(f"board-map hash mismatch: {sid}")
        if canonical_sha256(visible_board_facts(contract)) != state["board_fact_sha256"]:
            raise ValueError(f"board-fact hash mismatch: {sid}")
        if dense_labels(contract, sample_id=sid) != loads_json(paths["labels"].read_text()):
            raise ValueError(f"dense labels disagree with contract: {sid}")
        if board_answer(contract) != as_dict(as_list(row["messages"])[-1])["content"]:
            raise ValueError(f"source readout answer disagrees with contract: {sid}")
        contracts[sid] = contract
        verified[sid] = {
            "state_id": sid,
            "split": state["split"],
            "layout_id": row["layout_id"],
            "density_bin": state["density_bin"],
            "board_map_sha256": state["board_map_sha256"],
            "board_fact_sha256": state["board_fact_sha256"],
            "paths": {k: str(v) for k, v in paths.items()},
            "common_image": str(common_image),
            "sha256": hashes,
            "source": state["source"],
        }
        return contract

    return load_contract, verified
