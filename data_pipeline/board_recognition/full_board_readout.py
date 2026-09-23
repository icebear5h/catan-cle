"""One image -> one complete engine-verified board state, with fixed layout splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from data_pipeline.board_recognition.node_edge_readout import board_pieces, family_tokens
from data_pipeline.board_recognition.replay_dataset import file_sha256, read_jsonl
from data_pipeline.board_recognition.single_piece_localization import _row
from data_pipeline.board_recognition.sources import validate_public_board_contract
from data_pipeline.board_recognition.spatial_localization import (
    _deterministic_shuffle,
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.terrain_readout import layout_id, link_or_copy, terrain_facts
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict, JsonValue
from sft.board_state_readout import PROMPT, TASK, board_keys, score_board_state

SPLITS = ("train", "validation", "test", "color_diagnostic")


def board_answer(contract: JsonDict) -> str:
    validate_public_board_contract(contract)
    tiles, ports = terrain_facts(contract)
    tokens = family_tokens(contract)
    pieces = board_pieces(contract)
    items = [(t["token"], f"{t['resource']} {t['number']}") for t in tiles]
    for family in ("node", "edge"):
        items += [
            (token, as_str(pieces[family].get(token, {}).get("answer", "empty")))
            for token in tokens[family]
        ]
    items += [(port["token"], port["answer"]) for port in ports]
    robber = as_str(as_dict(contract["robber"])["tile_token"])
    tiles_with_robber = [
        as_str(tile["token"])
        for tile in map(as_dict, as_list(contract["tiles"]))
        if tile["has_robber"]
    ]
    if tiles_with_robber != [robber]:
        raise ValueError("robber location disagrees with tile flags")
    items.append(("robber", robber))
    if [key for key, _ in items] != board_keys():
        raise ValueError("board token inventory or order is invalid")
    return "; ".join(f"{key} {value}" for key, value in items)


def row_for_state(state: JsonDict, contract: JsonDict) -> JsonDict:
    pieces = board_pieces(contract)
    answer = board_answer(contract)
    if not score_board_state(answer, answer)["correct"]:
        raise AssertionError("target failed round-trip scoring")
    return _row(
        row_id=as_str(state["sample_id"]) + "_full_board",
        image_name=Path(as_str(state["image_path"])).name,
        prompt=PROMPT, answer=answer, task_type=TASK, category="board.readout", polarity="positive",
        schema="catan_full_board_readout_row/v1", grounding_stage=TASK, task_family=TASK,
        spatial_target=None, metadata={"split": state["split"], "state_id": state["sample_id"],
            "layout_id": layout_id(as_str(state["sample_id"])), "density_bin": state["density_bin"],
            "piece_count": sum(len(p) for p in pieces.values()), "item_count": 155,
            "entity_type": "board", "color_heldout": False},
    )


def export_full_board(dataset_dir: str | Path, output_dir: str | Path) -> JsonDict:
    root, output = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    states = read_jsonl(root / "manifest.jsonl")
    layouts = {
        split: {layout_id(as_str(s["sample_id"])) for s in states if s["split"] == split}
        for split in SPLITS
    }
    for left, right in combinations(SPLITS, 2):
        if layouts[left] & layouts[right]:
            raise ValueError(f"layout leakage: {left}, {right}")
    rows: dict[str, list[JsonDict]] = {split: [] for split in SPLITS}
    (output / "images").mkdir(parents=True)
    for state in states:
        if state["split"] not in rows:
            raise ValueError(f"unknown split: {state['split']}")
        contract_path = root / as_str(state["contract_path"])
        image_path = root / as_str(state["image_path"])
        for kind, path in (("contract", contract_path), ("image", image_path)):
            if file_sha256(path) != as_dict(state["sha256"])[kind]:
                raise ValueError(f"{kind} changed: {path}")
        row = row_for_state(state, as_dict(json.loads(contract_path.read_text())))
        rows[as_str(state["split"])].append(row)
        link_or_copy(image_path, output / "images" / image_path.name)
    files: dict[str, JsonValue] = {}
    for split, items in rows.items():
        items = _deterministic_shuffle(items, "full_board_" + split)
        path = output / "stage1" / (split + ".jsonl")
        _write_jsonl(path, items)
        files[split] = {"rows": len(items), "sha256": file_sha256(path),
            "layouts": len(layouts[split]),
            "by_density": dict(Counter(as_str(r["density_bin"]) for r in items)),
            "max_answer_characters": max(
                (len(as_str(as_dict(as_list(r["messages"])[-1])["content"])) for r in items),
                default=0,
            )}
    metadata: JsonDict = {"schema": "catan_full_board_readout/v1", "source": str(root),
        "manifest_sha256": file_sha256(root / "manifest.jsonl"), "files": files,
        "prompt": PROMPT, "rows_per_image": 1, "items_per_answer": 155,
        "split_unit": "layout", "order": ["tiles", "nodes", "edges", "ports", "robber"]}
    _write_json(output / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(export_full_board(args.dataset_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
