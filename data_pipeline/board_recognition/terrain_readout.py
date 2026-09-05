"""Build the terrain readout curriculum (``terrain_readout_v1``).

Stage 2 of the gaussian ladder asks one question: what is at each tile and
port? Every replay board image already on disk gets the production forward
prompts for all 19 tiles and all 9 ports, duplicates and generic ports
included, plus one complete readout that lists every tile and port keyed by
atlas token in a fixed order so the model also practices answering without
omissions:

- ``tile_resource``: ``<T00> resource?`` -> ``wood`` (``desert`` for the desert)
- ``tile_number``: ``<T00> number?`` -> ``11`` (``none`` for the desert)
- ``port_type``: ``<P00> port?`` -> ``sheep port`` or ``3:1 port``
- ``terrain_readout``: ``Read all tiles and ports.`` ->
  ``<T00> wood 11; <T01> brick 2; ...; <P00> sheep port; ...``

No inverse ("Where is the ...?") rows: the inverse direction is not a
bijection on boards with duplicate tiles and it interferes with the forward
heads. No piece rows. Boards are split by layout, never by image: every
state of one replay shares a layout and lives in one split, so validation
and test layouts are unseen.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.board_recognition.replay_dataset import (
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.single_piece_localization import _row, tile_facts
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _deterministic_shuffle,
    _summarize_rows,
    _write_json,
    _write_jsonl,
)


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_terrain_readout/v1"
ROW_SCHEMA = "catan_terrain_readout_row/v1"
DEFAULT_OUTPUT_NAME = "terrain_readout_v1"
GROUNDING_STAGE = "terrain_readout"
TASK_FAMILY = "terrain_readout"
SPLITS = ("train", "validation", "test")
READOUT_PROMPT = "Read all tiles and ports."
READOUTS_PER_IMAGE = 1


def layout_id(sample_id: str) -> str:
    """The replay a state belongs to; every state of a replay shares one board layout."""

    head, sep, tail = sample_id.rpartition("_s")
    return head if sep and tail.isdigit() else sample_id


def port_answer(port: JsonDict) -> str:
    """Production ``port?`` answer: ``3:1 port`` or ``<resource> port``."""

    if port.get("kind") == "generic" or port.get("resource") is None:
        return "3:1 port"
    return f"{str(port['resource']).lower()} port"


def terrain_facts(contract: JsonDict) -> tuple[list[JsonDict], list[JsonDict]]:
    tiles = sorted(tile_facts(contract), key=lambda tile: tile["token"])
    ports = sorted(({"token": port["token"], "answer": port_answer(port)} for port in contract["ports"]), key=lambda port: port["token"])
    if len(tiles) != 19 or len(ports) != 9:
        raise SpatialLocalizationError(f"expected 19 tiles and 9 ports, got {len(tiles)} and {len(ports)}")
    return tiles, ports


def readout_answer(tiles: Sequence[JsonDict], ports: Sequence[JsonDict]) -> str:
    """Every tile and port in token order, one canonical string."""

    parts = [f"{tile['token']} {tile['resource']} {tile['number']}" for tile in tiles]
    parts += [f"{port['token']} {port['answer']}" for port in ports]
    return "; ".join(parts)


def rows_for_state(state: JsonDict, contract: JsonDict, *, readouts: int = READOUTS_PER_IMAGE) -> list[JsonDict]:
    tiles, ports = terrain_facts(contract)
    image_name = Path(state["image_path"]).name
    base = {
        "split": state["split"],
        "state_id": state["sample_id"],
        "layout_id": layout_id(state["sample_id"]),
        "density_bin": state.get("density_bin"),
        "color_heldout": False,
    }
    common = {"image_name": image_name, "spatial_target": None, "schema": ROW_SCHEMA, "grounding_stage": GROUNDING_STAGE, "task_family": TASK_FAMILY}
    rows: list[JsonDict] = []
    for tile in tiles:
        token = tile["token"]
        metadata = {**base, "entity_type": "tile", "target_token": token, "piece": "TILE", "color": "none"}
        rows.append(_row(row_id=f"{state['sample_id']}_{token[1:-1]}_tile_resource", prompt=f"{token} resource?", answer=tile["resource"], task_type="tile_resource", category="tile.resource", polarity="positive", metadata=metadata, **common))
        rows.append(_row(row_id=f"{state['sample_id']}_{token[1:-1]}_tile_number", prompt=f"{token} number?", answer=tile["number"], task_type="tile_number", category="tile.number", polarity="positive", metadata=metadata, **common))
    for port in ports:
        token = port["token"]
        metadata = {**base, "entity_type": "port", "target_token": token, "piece": "PORT", "color": "none"}
        rows.append(_row(row_id=f"{state['sample_id']}_{token[1:-1]}_port_type", prompt=f"{token} port?", answer=port["answer"], task_type="port_type", category="port.port_type", polarity="positive", metadata=metadata, **common))
    for index in range(readouts):
        metadata = {**base, "entity_type": "board", "target_token": "<T00>", "piece": "BOARD", "color": "none"}
        rows.append(_row(row_id=f"{state['sample_id']}_terrain_readout_{index}", prompt=READOUT_PROMPT, answer=readout_answer(tiles, ports), task_type="terrain_readout", category="terrain.readout", polarity="positive", metadata=metadata, **common))
    return rows


def export_terrain_readout(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    readouts_per_image: int = READOUTS_PER_IMAGE,
    splits: Sequence[str] = SPLITS,
    validate_dataset: bool = True,
) -> JsonDict:
    dataset_root = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    if validate_dataset:
        validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise SpatialLocalizationError("refusing to overwrite output outside the replay dataset")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    states = [row for row in read_jsonl(dataset_root / "manifest.jsonl") if row["split"] in splits]
    layouts_by_split: dict[str, set[str]] = {split: set() for split in splits}
    for state in states:
        layouts_by_split[state["split"]].add(layout_id(state["sample_id"]))
    overlap = set.intersection(*(layouts_by_split[split] for split in splits)) if len(splits) > 1 else set()
    if overlap:
        raise SpatialLocalizationError(f"layouts appear in more than one split: {sorted(overlap)[:5]}")

    files: dict[str, JsonDict] = {}
    for split in splits:
        rows: list[JsonDict] = []
        for state in states:
            if state["split"] != split:
                continue
            contract_path = dataset_root / state["contract_path"]
            if file_sha256(contract_path) != state["sha256"]["contract"]:
                raise SpatialLocalizationError(f"contract changed: {contract_path}")
            if not (dataset_root / state["image_path"]).is_file():
                raise SpatialLocalizationError(f"missing image: {state['image_path']}")
            rows.extend(rows_for_state(state, json.loads(contract_path.read_text()), readouts=readouts_per_image))
        if split == "train":
            rows = _deterministic_shuffle(rows, "terrain_readout")
        path = output / "stage1" / f"{split}.jsonl"
        _write_jsonl(path, rows)
        summary = _summarize_rows(rows)
        summary["dimensions"]["density_bin"] = dict(sorted(Counter(str(row.get("density_bin")) for row in rows).items()))
        summary["unique_images"] = len({row["images"][0] for row in rows})
        summary["layouts"] = len(layouts_by_split[split])
        files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "image_root": str(dataset_root / "images"),
        "rows_per_image": 19 * 2 + 9 + readouts_per_image,
        "readouts_per_image": readouts_per_image,
        "readout_prompt": READOUT_PROMPT,
        "split_unit": "layout (replay); every state of a replay shares one layout and one split",
        "layouts_by_split": {split: len(layouts) for split, layouts in layouts_by_split.items()},
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--readouts-per-image", type=int, default=READOUTS_PER_IMAGE)
    args = parser.parse_args(argv)
    result = export_terrain_readout(args.dataset_dir, output_dir=args.output_dir, overwrite=args.overwrite, readouts_per_image=args.readouts_per_image)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
