"""Compose the approved mixed 128-step continuation from existing local images."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from sft.board.spatial_tasks import atlas_node_graph
from sft.json_types import JsonDict, JsonList, as_dict

from ._outputs import _write_outputs
from ._production import _row, _select_production
from ._queries import _bank_queries, _path_pairs, _query, _select_paths, _varied_states
from ._sources import (
    DEFAULT_OUTPUT,
    DEFAULT_ROOT,
    NEW_PANELS,
    STEP_QUOTAS,
    VERSION,
    _make_loader,
    _prepare_sources,
)


def build_dataset(output_dir: Path, *, root: Path = DEFAULT_ROOT, seed: int = 45) -> JsonDict:
    """Write a new immutable local dataset; return its launcher input manifest."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    ctx = _prepare_sources(Path(root).resolve())
    load_contract, verified = _make_loader(ctx)

    rng = random.Random(seed)
    ordered = _varied_states(ctx.train_sources, rng)
    validation_order = _varied_states(ctx.sources["validation"], random.Random(seed + 1))
    membership, path_info = _path_pairs(seed)
    nodes = sorted(atlas_node_graph())
    rng.shuffle(nodes)
    queries = _bank_queries(rng)
    queries["node_tiles"] = [{"node": nodes[i % len(nodes)]} for i in range(128)]
    queries["shortest_node_path"] = _select_paths(membership["train"], path_info, 128, rng)
    queries["local_node_tiles"] = [{"node": nodes[i % len(nodes)]} for i in range(128)]
    families: dict[str, list[JsonDict]] = {family: [] for family in STEP_QUOTAS}
    # Retention gets first choice of 256 distinct, density-balanced board maps/images.
    for state in ordered[:256]:
        load_contract(state)
        row = copy.deepcopy(state)
        row.pop("curriculum_stage", None)
        row_id = f"{VERSION}/train/full_board_readout/{state['state_id']}/readout"
        row.update(id=row_id, row_id=row_id, training_family="full_board_readout")
        row["metadata"] = {
            **as_dict(row.get("metadata", {})),
            "task_type": "full_board_readout",
            "training_family": "full_board_readout",
            "state_id": state["state_id"],
            "split": "train",
            "layout_id": state["layout_id"],
            "density_bin": state["density_bin"],
            "board_map_sha256": state["board_map_sha256"],
            "source_readout_row_id": state["row_id"],
        }
        families["full_board_readout"].append(row)
    offset = 256
    for family, targets in queries.items():
        for state, target in zip(ordered[offset : offset + 128], targets, strict=True):
            contract = load_contract(state)
            query = (
                target
                if family in ("directions", "adjacency_connectivity")
                else _query(family, target, contract)
            )
            families[family].append(_row(state, family, query, "train"))
        offset += 128
    for state, query in _select_production(ordered[offset:], 128, load_contract, rng):
        families["dice_production"].append(_row(state, "dice_production", query, "train"))
    panels: dict[str, list[JsonDict]] = {}
    panel_rng = random.Random(seed + 2)
    for task in NEW_PANELS:
        cases: list[tuple[JsonDict, JsonDict]]
        if task == "dice_production":
            cases = _select_production(validation_order, 64, load_contract, panel_rng)
        else:
            targets = (
                _select_paths(membership["validation"], path_info, 64, panel_rng)
                if task == "shortest_node_path"
                else [
                    {"node": nodes[i % len(nodes)]}
                    for i in range(54 if task == "node_tiles" else 64)
                ]
            )
            cases = [
                (state, _query(task, target, load_contract(state)))
                for state, target in zip(validation_order, targets)
            ]
        panels[task] = [_row(state, task, query, "validation") for state, query in cases]

    blocks: list[tuple[int, list[list[JsonDict]]]] = []
    for block in range(16):
        batches: list[list[JsonDict]] = []
        for family, rows in families.items():
            per_block = 2 if family == "full_board_readout" else 1
            for part in range(per_block):
                batch_index = block * per_block + part
                batch = rows[batch_index * 8 : (batch_index + 1) * 8]
                if len(batch) != 8:
                    raise AssertionError(f"incomplete batch: {family}")
                rng.shuffle(batch)
                batches.append(batch)
        rng.shuffle(batches)
        blocks.append((block, batches))
    rng.shuffle(blocks)
    train: list[JsonDict] = []
    step_order: JsonList = []
    for block_id, batches in blocks:
        for batch in batches:
            step = len(step_order)
            step_order.append(
                {
                    "step": step,
                    "block_id": block_id,
                    "training_family": batch[0]["training_family"],
                    "row_ids": [r["row_id"] for r in batch],
                }
            )
            for row in batch:
                as_dict(row["metadata"]).update(optimizer_step=step, block_id=block_id)
            train.extend(batch)
    return _write_outputs(
        output,
        train,
        panels,
        families,
        step_order,
        seed,
        ctx.source_hashes,
        ctx.image_root,
        ctx.inventory_path,
        ctx.split_maps,
        ctx.sources,
        ctx.train_sources,
        membership,
        path_info,
        verified,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output_dir, root=args.root, seed=args.seed), indent=2))
