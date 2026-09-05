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
- ``terrain_readout``: an explicit instruction naming the 19 tiles, the 9
  ports, the item format, the order, and the separator ->
  ``<T00> wood 11; <T01> brick 2; ...; <P00> sheep port; ...``. The first
  export used the bare "Read all tiles and ports." and half the readouts
  stopped early; the count, order, and schema in the prompt give the model
  a stopping criterion instead of something to infer from one row in 48.

No inverse ("Where is the ...?") rows: the inverse direction is not a
bijection on boards with duplicate tiles and it interferes with the forward
heads. No piece rows. Boards are split by layout, never by image: every
state of one replay shares a layout and lives in one split, so validation
and test layouts are unseen.

The replay corpus only has 77 training layouts, which a 27B model can
memorise. ``--synthetic-train N`` adds N freshly randomised engine boards
(tiles, numbers, and ports shuffled by the game engine from a seed), rendered
empty by the production renderer, each a layout no replay has. Synthetic
validation layouts are drawn from a disjoint seed stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from cle.game_engine.game import GameEngine
from cle.sandbox.palette import balanced_datagen_colors
from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH,
    file_sha256,
    load_render_style,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.single_piece_localization import _row, render_contract, tile_facts
from data_pipeline.board_recognition.sources import validate_public_board_contract
from evals.catan_board_bench.builder import CatanObservationSuite
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
READOUT_PROMPT = (
    "List every tile <T00> to <T18> as \"token resource number\" and every port <P00> to <P08> "
    "as \"token port\", in token order, separated by \"; \"."
)
READOUTS_PER_IMAGE = 1
SYNTHETIC_IMAGE_SIZE = 1024


def synthetic_seed(seed: int, split: str, index: int) -> int:
    digest = hashlib.sha256(f"terrain-synthetic:{seed}:{split}:{index}".encode()).hexdigest()
    return int(digest[:8], 16)


def synthetic_board(seed: int, split: str, index: int) -> tuple[JsonDict, JsonDict]:
    """A freshly randomised empty engine board as a manifest-like state plus its contract."""

    game_seed = synthetic_seed(seed, split, index)
    engine = GameEngine(balanced_datagen_colors(index, seed=seed), seed=game_seed, shuffle_players=False)
    sample_id = f"synth{game_seed:010d}_s000000"
    contract = CatanObservationSuite().public_board_contract(
        engine,
        sample={"id": sample_id, "index": 0},
        source={"kind": "synthetic_layout", "engine_seed": game_seed, "split": split},
    )
    validate_public_board_contract(contract)
    state = {
        "sample_id": sample_id,
        "split": split,
        "image_path": f"images/{split}_{sample_id}.png",
        "density_bin": "empty",
        "image_size": [SYNTHETIC_IMAGE_SIZE, SYNTHETIC_IMAGE_SIZE],
        "source": {"kind": "synthetic_layout", "engine_seed": game_seed},
    }
    return state, contract


def _render_synthetic(seed: int, split: str, index: int, destination: Path, style_path: Path, image_size: int) -> tuple[JsonDict, JsonDict]:
    state, contract = synthetic_board(seed, split, index)
    render_contract(contract, image_size, load_render_style(style_path), destination)
    state["image_size"] = [image_size, image_size]
    return state, contract


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


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
    synthetic: dict[str, int] | None = None,
    synthetic_seed_value: int = 20260904,
    style_path: str | Path = DEFAULT_STYLE_PATH,
    image_size: int = SYNTHETIC_IMAGE_SIZE,
    workers: int | None = None,
) -> JsonDict:
    """Export the terrain rows; ``synthetic`` maps a split to a count of engine layouts to add.

    Synthetic boards are rendered into ``<output>/images`` and the replay
    images are hard-linked there, so one image root serves the whole file.
    """

    dataset_root = Path(dataset_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    synthetic = dict(synthetic or {})
    if validate_dataset:
        validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise SpatialLocalizationError("refusing to overwrite output outside the replay dataset")
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True)

    states = [row for row in read_jsonl(dataset_root / "manifest.jsonl") if row["split"] in splits]
    contracts: dict[str, JsonDict] = {}
    for state in states:
        contract_path = dataset_root / state["contract_path"]
        if file_sha256(contract_path) != state["sha256"]["contract"]:
            raise SpatialLocalizationError(f"contract changed: {contract_path}")
        if not (dataset_root / state["image_path"]).is_file():
            raise SpatialLocalizationError(f"missing image: {state['image_path']}")
        link_or_copy(dataset_root / state["image_path"], images_dir / Path(state["image_path"]).name)
        contracts[state["sample_id"]] = json.loads(contract_path.read_text())
    synthetic_counts = {split: int(synthetic.get(split, 0)) for split in splits}
    worker_count = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(_render_synthetic, synthetic_seed_value, split, index, images_dir / f"{split}_synth{synthetic_seed(synthetic_seed_value, split, index):010d}_s000000.png", Path(style_path), image_size)
            for split in splits
            for index in range(synthetic_counts[split])
        ]
        for future in futures:
            state, contract = future.result()
            states.append(state)
            contracts[state["sample_id"]] = contract
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
            rows.extend(rows_for_state(state, contracts[state["sample_id"]], readouts=readouts_per_image))
        if split == "train":
            rows = _deterministic_shuffle(rows, "terrain_readout")
        path = output / "stage1" / f"{split}.jsonl"
        _write_jsonl(path, rows)
        summary = _summarize_rows(rows)
        summary["dimensions"]["density_bin"] = dict(sorted(Counter(str(row.get("density_bin")) for row in rows).items()))
        summary["unique_images"] = len({row["images"][0] for row in rows})
        summary["layouts"] = len(layouts_by_split[split])
        summary["synthetic_layouts"] = synthetic_counts[split]
        files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "image_root": str(images_dir),
        "synthetic_layouts": synthetic_counts,
        "synthetic_seed": synthetic_seed_value,
        "synthetic_image_size": image_size,
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
    parser.add_argument("--synthetic-train", type=int, default=0, help="Randomised engine layouts to add to train.")
    parser.add_argument("--synthetic-validation", type=int, default=0)
    parser.add_argument("--synthetic-test", type=int, default=0)
    parser.add_argument("--synthetic-seed", type=int, default=20260904)
    parser.add_argument("--image-size", type=int, default=SYNTHETIC_IMAGE_SIZE)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    result = export_terrain_readout(
        args.dataset_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        readouts_per_image=args.readouts_per_image,
        synthetic={"train": args.synthetic_train, "validation": args.synthetic_validation, "test": args.synthetic_test},
        synthetic_seed_value=args.synthetic_seed,
        image_size=args.image_size,
        workers=args.workers,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
