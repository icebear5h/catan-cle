"""Output directory preparation, JSON writers, and candidate writing."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

from data_pipeline.board_recognition.replay_impl._config import (
    SAMPLE_SCHEMA,
    BoardStateCandidate,
)
from data_pipeline.board_recognition.replay_impl._labels import (
    dense_labels,
)
from data_pipeline.board_recognition.sources import (
    file_sha256,
    repository_relative,
)
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict, JsonValue
from evals.catan_board_bench.render import RenderStyle, render_contract_image


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_candidate(
    output_dir: Path,
    candidate: BoardStateCandidate,
    *,
    split: str,
    sample_index: int,
    image_size: int,
    style: RenderStyle,
    style_path: Path,
) -> JsonDict:
    source_prefix = "r" if candidate.source["kind"] == "colonist_replay" else "e"
    source_position = (
        candidate.source["replay_step"]
        if candidate.source["kind"] == "colonist_replay"
        else candidate.source["engine_action_count"]
    )
    trajectory_slug = candidate.trajectory_id.replace(":", "_")
    position = as_int(source_position) if source_position else 0
    sample_id = f"{source_prefix}_{trajectory_slug}_s{position:06d}"
    contract_rel = Path("contracts") / f"{sample_id}.json"
    label_rel = Path("dense_labels") / f"{sample_id}.json"
    image_rel = Path("images") / f"{sample_id}.png"
    contract = copy.deepcopy(candidate.contract)
    contract["sample"] = {
        "id": sample_id,
        "index": sample_index,
        "contract_path": str(contract_rel),
        "image_path": str(image_rel),
        "image_size": [image_size, image_size],
    }
    contract["source"] = {**candidate.source, "split": split}
    labels = dense_labels(contract, sample_id=sample_id)
    image = render_contract_image(contract, image_size=image_size, style=style)
    write_json(output_dir / contract_rel, contract)
    write_json(output_dir / label_rel, labels)
    image.save(output_dir / image_rel)
    nodes = [as_dict(node) for node in as_list(contract["nodes"])]
    edges = [as_dict(edge) for edge in as_list(contract["edges"])]
    colors = sorted(
        {as_str(node["color"]) for node in nodes if node["color"] is not None}
        | {as_str(edge["road_color"]) for edge in edges if edge["road_color"] is not None}
    )
    return {
        "schema": SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "split": split,
        "view": "raw_full_board",
        "contract_path": str(contract_rel),
        "label_path": str(label_rel),
        "image_path": str(image_rel),
        "image_size": [image_size, image_size],
        "source": contract["source"],
        "board_fact_sha256": candidate.board_fact_sha256,
        "board_map_sha256": candidate.board_map_sha256,
        "building_count": candidate.building_count,
        "road_count": candidate.road_count,
        "density_bin": candidate.density_bin,
        "color_palette": [
            as_dict(player)["color"] for player in as_list(contract["players"])
        ],
        "visible_piece_colors": [color for color in colors],
        "render": {
            "renderer": "evals.catan_board_bench.render",
            "style_config": repository_relative(style_path),
            "image_annotation": None,
        },
        "sha256": {
            "contract": file_sha256(output_dir / contract_rel),
            "labels": file_sha256(output_dir / label_rel),
            "image": file_sha256(output_dir / image_rel),
        },
    }

