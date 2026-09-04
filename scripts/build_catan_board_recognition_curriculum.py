#!/usr/bin/env python
"""Build and validate dense, symbolic Catan board-recognition curriculum data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from evals.catan_board_bench.paths import DATASETS_DIR
from evals.catan_board_bench.render import RenderStyle, render_contract_image
from evals.catan_board_bench.tokens import (
    atlas_metadata,
    building_token,
    color_token,
    edge_token,
    node_token,
    port_token,
    resource_token,
    tile_token,
)
from cle.game_engine.models.enums import CITY, SETTLEMENT
from sft.scripts.build_node_factor_dataset import (
    build_contract,
    build_indices,
    choose_distractor_buildings,
    player_summaries,
)


JsonDict = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "curriculum.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "generated" / "board_recognition" / "curriculum"
DEFAULT_LEAKAGE_LEDGER = (
    DATASETS_DIR / "catan_board_bench_100" / "leakage" / "benchmark_game_ids.json"
)
DEFAULT_STYLE_PATH = PROJECT_ROOT / "configs" / "sft" / "renderer_style.json"
DATASET_SCHEMA = "catan_board_recognition_dataset/v1"
SAMPLE_SCHEMA = "catan_board_recognition_sample/v1"
LABEL_SCHEMA = "catan_board_recognition_dense_labels/v1"
ENTITY_TYPES = ("tile", "node", "edge", "port")
RESOURCE_CLASSES = ("DESERT", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
NUMBER_CLASSES = (None, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
PORT_CLASSES = (
    "THREE_TO_ONE",
    "TWO_TO_ONE_WOOD",
    "TWO_TO_ONE_BRICK",
    "TWO_TO_ONE_SHEEP",
    "TWO_TO_ONE_WHEAT",
    "TWO_TO_ONE_ORE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--pairs-per-entity-type", type=int, default=1)
    parser.add_argument("--seed", type=int, default=381_427)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        report = validate_dataset(args.output_dir, spec_path=args.spec)
    else:
        report = build_dataset(
            output_dir=args.output_dir,
            spec_path=args.spec,
            image_size=args.image_size,
            pairs_per_entity_type=args.pairs_per_entity_type,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_dataset(
    *,
    output_dir: Path,
    spec_path: Path = DEFAULT_SPEC_PATH,
    image_size: int | None = None,
    pairs_per_entity_type: int = 1,
    seed: int = 381_427,
    overwrite: bool = False,
) -> JsonDict:
    if pairs_per_entity_type < 1:
        raise ValueError("pairs_per_entity_type must be positive")
    spec = load_spec(spec_path)
    render_config = spec["render"]
    resolved_image_size = image_size or int(render_config["image_size"])
    if resolved_image_size < 64 or resolved_image_size % 16:
        raise ValueError("image_size must be at least 64 and divisible by 16")
    prepare_output_dir(output_dir, overwrite=overwrite)
    contracts_dir = output_dir / "contracts"
    labels_dir = output_dir / "dense_labels"
    images_dir = output_dir / "images"
    splits_dir = output_dir / "splits"
    for directory in (contracts_dir, labels_dir, images_dir, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ledger, _excluded_game_ids = load_leakage_ledger(DEFAULT_LEAKAGE_LEDGER)
    style_path = resolve_project_path(render_config.get("style_config", str(DEFAULT_STYLE_PATH)))
    style = load_render_style(style_path)
    atlas = atlas_metadata()
    indices = build_indices(atlas)
    colors = tuple(spec["class_vocabularies"]["colors"])
    manifest_rows: list[JsonDict] = []
    group_index = 0

    for stage_index, stage in enumerate(spec["stages"]):
        for entity_type_index, entity_type in enumerate(ENTITY_TYPES):
            for pair_index in range(pairs_per_entity_type):
                group_seed = (
                    seed + stage_index * 100_003 + entity_type_index * 10_007 + pair_index * 1_009
                )
                group_id = f"{stage['id']}_{entity_type}_p{pair_index:03d}"
                split = split_for_group(group_index, spec["splits"])
                target_node_id = (stage_index * 13 + entity_type_index * 7 + pair_index * 11) % 54
                base_contract = make_stage_contract(
                    atlas=atlas,
                    indices=indices,
                    stage=stage,
                    sample_id=f"{group_id}_base",
                    sample_index=group_index * 2,
                    target_node_id=target_node_id,
                    colors=colors,
                    seed=group_seed,
                    image_size=resolved_image_size,
                )
                target, before, after = apply_counterfactual(
                    base_contract,
                    entity_type=entity_type,
                    indices=indices,
                    colors=colors,
                    selector=stage_index + entity_type_index + pair_index,
                )
                counterfactual_contract = copy.deepcopy(base_contract)
                apply_declared_value(
                    counterfactual_contract, target=target, value=after, colors=colors
                )
                set_sample_identity(counterfactual_contract, f"{group_id}_counterfactual")
                set_sample_identity(base_contract, f"{group_id}_base")
                pair_contracts = (
                    ("base", base_contract),
                    ("counterfactual", counterfactual_contract),
                )
                pair_descriptor = {
                    "group_id": group_id,
                    "target": target,
                    "before": before,
                    "after": after,
                }
                for role, contract in pair_contracts:
                    sample_id = contract["sample"]["id"]
                    contract_rel = Path("contracts") / f"{sample_id}.json"
                    label_rel = Path("dense_labels") / f"{sample_id}.json"
                    image_rel = Path("images") / f"{sample_id}.png"
                    contract["sample"]["contract_path"] = str(contract_rel)
                    contract["sample"]["image_path"] = str(image_rel)
                    contract["sample"]["image_size"] = [resolved_image_size, resolved_image_size]
                    contract["source"] = {
                        "kind": "synthetic_engine_contract",
                        "generator": "scripts/build_catan_board_recognition_curriculum.py",
                        "seed": group_seed,
                        "curriculum_stage": stage["id"],
                        "counterfactual_group_id": group_id,
                        "benchmark_game_id": None,
                    }
                    labels = dense_labels(contract, stage_id=stage["id"])
                    image = render_contract_image(
                        contract,
                        image_size=resolved_image_size,
                        style=style,
                    )
                    write_json(output_dir / contract_rel, contract)
                    write_json(output_dir / label_rel, labels)
                    image.save(output_dir / image_rel)
                    manifest_rows.append(
                        {
                            "schema": SAMPLE_SCHEMA,
                            "sample_id": sample_id,
                            "counterfactual_group_id": group_id,
                            "counterfactual_role": role,
                            "split": split,
                            "stage": stage["id"],
                            "view": "raw_full_board",
                            "contract_path": str(contract_rel),
                            "label_path": str(label_rel),
                            "image_path": str(image_rel),
                            "image_size": [resolved_image_size, resolved_image_size],
                            "source": contract["source"],
                            "counterfactual": pair_descriptor,
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
                    )
                group_index += 1

    manifest_rows.sort(key=lambda row: row["sample_id"])
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    for split in ("train", "validation", "test"):
        write_jsonl(
            splits_dir / f"{split}.jsonl",
            [row for row in manifest_rows if row["split"] == split],
        )
    metadata = build_metadata(
        spec=spec,
        spec_path=spec_path,
        output_dir=output_dir,
        manifest_rows=manifest_rows,
        image_size=resolved_image_size,
        pairs_per_entity_type=pairs_per_entity_type,
        seed=seed,
        style_path=style_path,
        ledger=ledger,
    )
    write_json(output_dir / "metadata.json", metadata)
    report = validate_dataset(output_dir, spec_path=spec_path)
    report["output_dir"] = str(output_dir)
    return report


def load_spec(path: Path) -> JsonDict:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = json.loads(path.read_text())
    if spec.get("schema") != "catan_board_recognition_curriculum/v1":
        raise ValueError(f"unsupported curriculum schema: {spec.get('schema')}")
    if [stage["id"] for stage in spec.get("stages", [])] != [
        "empty_setup",
        "initial_placements",
        "sparse_midgame",
        "dense_endgame",
    ]:
        raise ValueError("curriculum must define the four ordered board-complexity stages")
    if spec.get("views") != ["raw_full_board"]:
        raise ValueError("pilot supports raw_full_board only")
    if tuple(spec["sampling"]["entity_type_order"]) != ENTITY_TYPES:
        raise ValueError("entity sampling must cover tile, node, edge, and port equally")
    return spec


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_render_style(path: Path) -> RenderStyle:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    values = payload.get("style", payload)
    keys = RenderStyle.__dataclass_fields__.keys()
    return RenderStyle(**{key: values[key] for key in keys if key in values})


def make_stage_contract(
    *,
    atlas: JsonDict,
    indices: JsonDict,
    stage: JsonDict,
    sample_id: str,
    sample_index: int,
    target_node_id: int,
    colors: Sequence[str],
    seed: int,
    image_size: int,
) -> JsonDict:
    contract = build_contract(
        atlas=atlas,
        indices=indices,
        sample_id=sample_id,
        sample_index=sample_index,
        target_node_id=target_node_id,
        occupancy={"name": "empty", "color": None, "building": None},
        colors=list(colors),
        seed=seed,
        image_size=image_size,
        distractor_buildings=0,
        distractor_roads=0,
        assume_rendered_images=True,
    )
    rng = random.Random(seed + 37)
    target_occupancy = stage["target_occupancy"]
    buildings: dict[int, tuple[str, str]] = {}
    if target_occupancy != "EMPTY":
        color, building = target_occupancy.split("_", maxsplit=1)
        buildings[target_node_id] = (color, building)
    buildings.update(
        choose_distractor_buildings(
            indices=indices,
            rng=rng,
            target_node_id=target_node_id,
            colors=list(colors),
            count=int(stage["distractor_buildings"]),
            existing=buildings,
        )
    )
    edge_ids = sorted(indices["edges"])
    rng.shuffle(edge_ids)
    roads = {
        edge: colors[(edge[0] + edge[1] + index + sample_index) % len(colors)]
        for index, edge in enumerate(edge_ids[: int(stage["roads"])])
    }
    set_dynamic_pieces(contract, buildings=buildings, roads=roads, colors=colors)
    return contract


def set_dynamic_pieces(
    contract: JsonDict,
    *,
    buildings: dict[int, tuple[str, str]],
    roads: dict[tuple[int, int], str],
    colors: Sequence[str],
) -> None:
    for node in contract["nodes"]:
        building = buildings.get(int(node["id"]))
        node["color"] = building[0] if building else None
        node["color_token"] = color_token(building[0]) if building else None
        node["building"] = building[1] if building else None
        node["building_token"] = building_token(building[1]) if building else None
    for edge in contract["edges"]:
        edge_id = tuple(int(value) for value in edge["id"])
        owner = roads.get(edge_id)
        edge["road_color"] = owner
        edge["road_color_token"] = color_token(owner) if owner else None
    contract["players"] = player_summaries(
        colors=list(colors),
        buildings=buildings,
        roads=roads,
    )


def apply_counterfactual(
    contract: JsonDict,
    *,
    entity_type: str,
    indices: JsonDict,
    colors: Sequence[str],
    selector: int,
) -> tuple[JsonDict, Any, Any]:
    if entity_type == "tile":
        candidates = [tile for tile in contract["tiles"] if tile["resource"] is not None]
        tile = candidates[selector % len(candidates)]
        before = str(tile["resource"])
        resources = [
            resource for resource in RESOURCE_CLASSES if resource not in {"DESERT", before}
        ]
        after = resources[selector % len(resources)]
        return (
            {"entity_type": "tile", "entity_id": tile_token(tile["id"]), "attribute": "resource"},
            before,
            after,
        )
    if entity_type == "node":
        node = contract["nodes"][selector % len(contract["nodes"])]
        before = node_occupancy(node)
        choices = [
            "EMPTY",
            *[f"{color}_{building}" for color in colors for building in (SETTLEMENT, CITY)],
        ]
        after = choices[(choices.index(before) + 1 + selector) % len(choices)]
        if after == before:
            after = choices[(choices.index(before) + 1) % len(choices)]
        return (
            {"entity_type": "node", "entity_id": node_token(node["id"]), "attribute": "occupancy"},
            before,
            after,
        )
    if entity_type == "edge":
        edge_ids = sorted(indices["edges"])
        edge_id = edge_ids[selector % len(edge_ids)]
        edge = find_edge(contract, edge_id)
        before = edge["road_color"] or "EMPTY"
        choices = ["EMPTY", *colors]
        after = choices[(choices.index(before) + 1 + selector) % len(choices)]
        if after == before:
            after = choices[(choices.index(before) + 1) % len(choices)]
        return (
            {"entity_type": "edge", "entity_id": edge_token(edge_id), "attribute": "owner"},
            before,
            after,
        )
    if entity_type == "port":
        port = contract["ports"][selector % len(contract["ports"])]
        before = port_type(port)
        after = PORT_CLASSES[(PORT_CLASSES.index(before) + 1 + selector) % len(PORT_CLASSES)]
        if after == before:
            after = PORT_CLASSES[(PORT_CLASSES.index(before) + 1) % len(PORT_CLASSES)]
        return (
            {"entity_type": "port", "entity_id": port_token(port["id"]), "attribute": "port_type"},
            before,
            after,
        )
    raise ValueError(f"unknown entity type: {entity_type}")


def apply_declared_value(
    contract: JsonDict,
    *,
    target: JsonDict,
    value: Any,
    colors: Sequence[str],
) -> None:
    entity_type = target["entity_type"]
    entity_id = target["entity_id"]
    if entity_type == "tile":
        tile = find_token(contract["tiles"], entity_id)
        tile["resource"] = value
        tile["resource_token"] = resource_token(value)
        return
    if entity_type == "node":
        node = find_token(contract["nodes"], entity_id)
        if value == "EMPTY":
            node["color"] = None
            node["color_token"] = None
            node["building"] = None
            node["building_token"] = None
        else:
            color, building = value.split("_", maxsplit=1)
            node["color"] = color
            node["color_token"] = color_token(color)
            node["building"] = building
            node["building_token"] = building_token(building)
        refresh_player_summaries(contract, colors=colors)
        return
    if entity_type == "edge":
        edge = find_token(contract["edges"], entity_id)
        owner = None if value == "EMPTY" else value
        edge["road_color"] = owner
        edge["road_color_token"] = color_token(owner) if owner else None
        refresh_player_summaries(contract, colors=colors)
        return
    if entity_type == "port":
        port = find_token(contract["ports"], entity_id)
        if value == "THREE_TO_ONE":
            port.update(
                {"kind": "generic", "ratio": "3:1", "resource": None, "resource_token": None}
            )
        else:
            resource = value.removeprefix("TWO_TO_ONE_")
            port.update(
                {
                    "kind": "resource",
                    "ratio": "2:1",
                    "resource": resource,
                    "resource_token": resource_token(resource),
                }
            )
        return
    raise ValueError(f"unknown entity type: {entity_type}")


def refresh_player_summaries(contract: JsonDict, *, colors: Sequence[str]) -> None:
    buildings = {
        int(node["id"]): (node["color"], node["building"])
        for node in contract["nodes"]
        if node["color"] and node["building"]
    }
    roads = {
        tuple(int(value) for value in edge["id"]): edge["road_color"]
        for edge in contract["edges"]
        if edge["road_color"]
    }
    contract["players"] = player_summaries(
        colors=list(colors),
        buildings=buildings,
        roads=roads,
    )


def set_sample_identity(contract: JsonDict, sample_id: str) -> None:
    contract["sample"]["id"] = sample_id


def dense_labels(contract: JsonDict, *, stage_id: str) -> JsonDict:
    return {
        "schema": LABEL_SCHEMA,
        "sample_id": contract["sample"]["id"],
        "stage": stage_id,
        "entities": {
            "tiles": [
                {
                    "id": tile_token(tile["id"]),
                    "resource": "DESERT" if tile["resource"] is None else tile["resource"],
                    "number": tile["number"],
                    "robber": bool(tile["has_robber"]),
                }
                for tile in sorted(contract["tiles"], key=lambda row: int(row["id"]))
            ],
            "nodes": [
                {"id": node_token(node["id"]), "occupancy": node_occupancy(node)}
                for node in sorted(contract["nodes"], key=lambda row: int(row["id"]))
            ],
            "edges": [
                {
                    "id": edge_token(tuple(edge["id"])),
                    "owner": edge["road_color"] or "EMPTY",
                }
                for edge in sorted(contract["edges"], key=lambda row: tuple(row["id"]))
            ],
            "ports": [
                {"id": port_token(port["id"]), "port_type": port_type(port)}
                for port in sorted(contract["ports"], key=lambda row: int(row["id"]))
            ],
        },
    }


def node_occupancy(node: JsonDict) -> str:
    if not node.get("color") or not node.get("building"):
        return "EMPTY"
    return f"{node['color']}_{node['building']}"


def port_type(port: JsonDict) -> str:
    if port.get("resource") is None:
        return "THREE_TO_ONE"
    return f"TWO_TO_ONE_{port['resource']}"


def find_token(rows: Sequence[JsonDict], token: str) -> JsonDict:
    for row in rows:
        candidate = row.get("token")
        if candidate == token:
            return row
    raise KeyError(token)


def find_edge(contract: JsonDict, edge_id: tuple[int, int]) -> JsonDict:
    for edge in contract["edges"]:
        if tuple(edge["id"]) == edge_id:
            return edge
    raise KeyError(edge_id)


def split_for_group(group_index: int, split_spec: JsonDict) -> str:
    bucket = group_index % int(split_spec["bucket_modulus"])
    for split in ("train", "validation", "test"):
        if bucket in split_spec[split]:
            return split
    raise ValueError(f"split bucket {bucket} is unassigned")


def build_metadata(
    *,
    spec: JsonDict,
    spec_path: Path,
    output_dir: Path,
    manifest_rows: Sequence[JsonDict],
    image_size: int,
    pairs_per_entity_type: int,
    seed: int,
    style_path: Path,
    ledger: JsonDict,
) -> JsonDict:
    stage_counts = Counter(row["stage"] for row in manifest_rows)
    split_counts = Counter(row["split"] for row in manifest_rows)
    target_counts = Counter(
        row["counterfactual"]["target"]["entity_type"]
        for row in manifest_rows
        if row["counterfactual_role"] == "base"
    )
    class_counts = collect_class_counts(output_dir, manifest_rows)
    return {
        "schema": DATASET_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "curriculum_schema": spec["schema"],
        "curriculum_path": repository_relative(spec_path),
        "curriculum_sha256": file_sha256(spec_path),
        "seed": seed,
        "pairs_per_entity_type": pairs_per_entity_type,
        "sample_count": len(manifest_rows),
        "counterfactual_group_count": len(manifest_rows) // 2,
        "image_size": [image_size, image_size],
        "views": ["raw_full_board"],
        "sampling": spec["sampling"],
        "stage_counts": dict(sorted(stage_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "counterfactual_target_counts": dict(sorted(target_counts.items())),
        "class_counts": class_counts,
        "renderer": {
            "implementation": "evals.catan_board_bench.render",
            "style_config": repository_relative(style_path),
            "style_sha256": file_sha256(style_path),
            "image_annotation": None,
        },
        "leakage": {
            "policy": "fail_closed",
            "benchmark": ledger["benchmark"],
            "ledger_path": repository_relative(DEFAULT_LEAKAGE_LEDGER),
            "ledger_sha256": file_sha256(DEFAULT_LEAKAGE_LEDGER),
            "excluded_game_ids": len(ledger["benchmark_game_ids"]),
        },
        "files": {
            "manifest": "manifest.jsonl",
            "contracts_dir": "contracts",
            "labels_dir": "dense_labels",
            "images_dir": "images",
            "splits_dir": "splits",
        },
    }


def collect_class_counts(output_dir: Path, rows: Sequence[JsonDict]) -> JsonDict:
    counts: dict[str, Counter[Any]] = {
        "tile.resource": Counter(),
        "tile.number": Counter(),
        "tile.robber": Counter(),
        "node.occupancy": Counter(),
        "edge.owner": Counter(),
        "port.port_type": Counter(),
    }
    for row in rows:
        labels = json.loads((output_dir / row["label_path"]).read_text())["entities"]
        for tile in labels["tiles"]:
            counts["tile.resource"][str(tile["resource"])] += 1
            counts["tile.number"]["NONE" if tile["number"] is None else str(tile["number"])] += 1
            counts["tile.robber"][str(tile["robber"]).upper()] += 1
        for node in labels["nodes"]:
            counts["node.occupancy"][node["occupancy"]] += 1
        for edge in labels["edges"]:
            counts["edge.owner"][edge["owner"]] += 1
        for port in labels["ports"]:
            counts["port.port_type"][port["port_type"]] += 1
    return {name: dict(sorted(counter.items())) for name, counter in counts.items()}


def validate_dataset(output_dir: Path, *, spec_path: Path = DEFAULT_SPEC_PATH) -> JsonDict:
    spec = load_spec(spec_path)
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "manifest.jsonl"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"dataset is incomplete: {output_dir}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != DATASET_SCHEMA:
        raise ValueError("dataset metadata schema mismatch")
    if metadata.get("curriculum_sha256") != file_sha256(spec_path):
        raise ValueError("curriculum specification changed")
    if metadata["leakage"]["ledger_sha256"] != file_sha256(DEFAULT_LEAKAGE_LEDGER):
        raise ValueError("benchmark leakage ledger changed")
    _ledger, excluded_game_ids = load_leakage_ledger(DEFAULT_LEAKAGE_LEDGER)
    rows = read_jsonl(manifest_path)
    if metadata.get("sample_count") != len(rows):
        raise ValueError("metadata sample count is stale")
    if len(rows) != len({row["sample_id"] for row in rows}):
        raise ValueError("manifest sample IDs are not unique")
    groups: dict[str, list[JsonDict]] = {}
    target_counts: Counter[str] = Counter()
    atlas = atlas_metadata()
    expected_ids = expected_entity_ids(atlas)
    colors = tuple(spec["class_vocabularies"]["colors"])
    style_path = resolve_project_path(spec["render"]["style_config"])
    style = load_render_style(style_path)
    node_classes = {
        "EMPTY",
        *[f"{color}_{building}" for color in colors for building in (SETTLEMENT, CITY)],
    }
    edge_classes = {"EMPTY", *colors}
    stages_by_id = {stage["id"]: stage for stage in spec["stages"]}

    for row in rows:
        if row.get("schema") != SAMPLE_SCHEMA:
            raise ValueError(f"sample schema mismatch: {row.get('sample_id')}")
        if row.get("view") != "raw_full_board" or row["render"].get("image_annotation") is not None:
            raise ValueError(f"non-raw image view in manifest: {row['sample_id']}")
        source = row.get("source", {})
        source_game_id = source.get("benchmark_game_id")
        if source_game_id is not None and str(source_game_id) in excluded_game_ids:
            raise ValueError(f"benchmark game leaked into curriculum: {source_game_id}")
        if source.get("kind") != "synthetic_engine_contract":
            raise ValueError(f"unsupported or unknown source provenance: {row['sample_id']}")
        if source_game_id is not None or not isinstance(source.get("seed"), int):
            raise ValueError(f"synthetic provenance is not fail-closed: {row['sample_id']}")
        if (
            source.get("curriculum_stage") != row["stage"]
            or source.get("counterfactual_group_id") != row["counterfactual_group_id"]
            or row["counterfactual"].get("group_id") != row["counterfactual_group_id"]
        ):
            raise ValueError(f"state provenance/group mismatch: {row['sample_id']}")
        if row["render"] != {
            "renderer": "evals.catan_board_bench.render",
            "style_config": repository_relative(style_path),
            "image_annotation": None,
        }:
            raise ValueError(f"render policy mismatch: {row['sample_id']}")
        for kind, path_key in (
            ("contract", "contract_path"),
            ("labels", "label_path"),
            ("image", "image_path"),
        ):
            artifact_path = output_dir / row[path_key]
            if not artifact_path.is_file():
                raise ValueError(f"missing {kind}: {artifact_path}")
            if row["sha256"][kind] != file_sha256(artifact_path):
                raise ValueError(f"{kind} hash mismatch: {artifact_path}")
        contract = json.loads((output_dir / row["contract_path"]).read_text())
        labels = json.loads((output_dir / row["label_path"]).read_text())
        if contract.get("sample", {}).get("id") != row["sample_id"]:
            raise ValueError(f"contract identity mismatch: {row['sample_id']}")
        if row["counterfactual_role"] == "base":
            stage = stages_by_id[row["stage"]]
            expected_buildings = int(stage["distractor_buildings"]) + int(
                stage["target_occupancy"] != "EMPTY"
            )
            actual_buildings = sum(node["building"] is not None for node in contract["nodes"])
            actual_roads = sum(edge["road_color"] is not None for edge in contract["edges"])
            if (actual_buildings, actual_roads) != (expected_buildings, int(stage["roads"])):
                raise ValueError(f"base state does not match curriculum stage: {row['sample_id']}")
        if labels != dense_labels(contract, stage_id=row["stage"]):
            raise ValueError(f"dense labels disagree with engine contract: {row['sample_id']}")
        with Image.open(output_dir / row["image_path"]) as actual_image:
            actual_rgb = actual_image.convert("RGB")
            expected_image = render_contract_image(
                contract,
                image_size=int(row["image_size"][0]),
                style=style,
            )
            if (
                actual_rgb.size != tuple(row["image_size"])
                or actual_rgb.tobytes() != expected_image.tobytes()
            ):
                raise ValueError(f"image disagrees with engine contract: {row['sample_id']}")
        validate_dense_labels(
            labels,
            expected_ids=expected_ids,
            node_classes=node_classes,
            edge_classes=edge_classes,
        )
        if labels["sample_id"] != row["sample_id"] or labels["stage"] != row["stage"]:
            raise ValueError(f"label identity mismatch: {row['sample_id']}")
        groups.setdefault(row["counterfactual_group_id"], []).append(row)

    for group_id, pair in groups.items():
        if len(pair) != 2 or {row["counterfactual_role"] for row in pair} != {
            "base",
            "counterfactual",
        }:
            raise ValueError(f"counterfactual group must contain one pair: {group_id}")
        if len({row["split"] for row in pair}) != 1 or len({row["stage"] for row in pair}) != 1:
            raise ValueError(f"counterfactual group crossed split or stage: {group_id}")
        descriptors = {canonical_json(row["counterfactual"]) for row in pair}
        if len(descriptors) != 1:
            raise ValueError(f"counterfactual descriptor mismatch: {group_id}")
        base = next(row for row in pair if row["counterfactual_role"] == "base")
        changed = next(row for row in pair if row["counterfactual_role"] == "counterfactual")
        base_labels = json.loads((output_dir / base["label_path"]).read_text())
        changed_labels = json.loads((output_dir / changed["label_path"]).read_text())
        target = base["counterfactual"]["target"]
        differences = dense_label_differences(base_labels["entities"], changed_labels["entities"])
        expected_path = target_label_path(target)
        if differences != [expected_path]:
            raise ValueError(
                f"counterfactual {group_id} changed {differences}, expected {[expected_path]}"
            )
        before = value_at_label_path(base_labels["entities"], expected_path)
        after = value_at_label_path(changed_labels["entities"], expected_path)
        if before != base["counterfactual"]["before"] or after != base["counterfactual"]["after"]:
            raise ValueError(f"counterfactual values disagree with labels: {group_id}")
        if base["sha256"]["image"] == changed["sha256"]["image"]:
            raise ValueError(f"counterfactual has no visible pixel change: {group_id}")
        target_counts[target["entity_type"]] += 1

    if set(target_counts) != set(ENTITY_TYPES) or len(set(target_counts.values())) != 1:
        raise ValueError(f"counterfactual entity types are not balanced: {dict(target_counts)}")
    expected_stage_counts = dict(sorted(Counter(row["stage"] for row in rows).items()))
    expected_split_counts = dict(sorted(Counter(row["split"] for row in rows).items()))
    if metadata.get("stage_counts") != expected_stage_counts:
        raise ValueError("metadata stage counts are stale")
    if metadata.get("split_counts") != expected_split_counts:
        raise ValueError("metadata split counts are stale")
    if metadata.get("counterfactual_target_counts") != dict(sorted(target_counts.items())):
        raise ValueError("metadata counterfactual target counts are stale")
    if metadata.get("class_counts") != collect_class_counts(output_dir, rows):
        raise ValueError("metadata class counts are stale")
    validate_split_files(output_dir, rows)
    return {
        "valid": True,
        "samples": len(rows),
        "counterfactual_groups": len(groups),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "stages": dict(sorted(Counter(row["stage"] for row in rows).items())),
        "targets": dict(sorted(target_counts.items())),
        "entity_counts_per_sample": {"tiles": 19, "nodes": 54, "edges": 72, "ports": 9},
    }


def validate_dense_labels(
    labels: JsonDict,
    *,
    expected_ids: dict[str, set[str]],
    node_classes: set[str],
    edge_classes: set[str],
) -> None:
    if labels.get("schema") != LABEL_SCHEMA:
        raise ValueError(f"dense label schema mismatch: {labels.get('sample_id')}")
    entities = labels.get("entities", {})
    for collection, expected in expected_ids.items():
        rows = entities.get(collection)
        if not isinstance(rows, list) or {row.get("id") for row in rows} != expected:
            raise ValueError(
                f"dense label {collection} coverage mismatch: {labels.get('sample_id')}"
            )
    if any(
        tile["resource"] not in RESOURCE_CLASSES
        or tile["number"] not in NUMBER_CLASSES
        or not isinstance(tile["robber"], bool)
        for tile in entities["tiles"]
    ):
        raise ValueError(f"invalid tile class: {labels.get('sample_id')}")
    if any(
        (tile["resource"] == "DESERT") != (tile["number"] is None) for tile in entities["tiles"]
    ):
        raise ValueError(f"tile resource/number mismatch: {labels.get('sample_id')}")
    if sum(tile["robber"] for tile in entities["tiles"]) != 1:
        raise ValueError(f"dense labels require exactly one robber: {labels.get('sample_id')}")
    if any(node["occupancy"] not in node_classes for node in entities["nodes"]):
        raise ValueError(f"invalid node class: {labels.get('sample_id')}")
    if any(edge["owner"] not in edge_classes for edge in entities["edges"]):
        raise ValueError(f"invalid edge class: {labels.get('sample_id')}")
    if any(port["port_type"] not in PORT_CLASSES for port in entities["ports"]):
        raise ValueError(f"invalid port class: {labels.get('sample_id')}")


def expected_entity_ids(atlas: JsonDict) -> dict[str, set[str]]:
    return {
        "tiles": {tile_token(tile["id"]) for tile in atlas["tiles"]},
        "nodes": {node_token(node["id"]) for node in atlas["nodes"]},
        "edges": {edge_token(tuple(edge["id"])) for edge in atlas["edges"]},
        "ports": {port_token(port["id"]) for port in atlas["ports"]},
    }


def target_label_path(target: JsonDict) -> str:
    collection = {"tile": "tiles", "node": "nodes", "edge": "edges", "port": "ports"}[
        target["entity_type"]
    ]
    return f"{collection}/{target['entity_id']}/{target['attribute']}"


def dense_label_differences(first: JsonDict, second: JsonDict) -> list[str]:
    first_flat = flatten_entities(first)
    second_flat = flatten_entities(second)
    if set(first_flat) != set(second_flat):
        raise ValueError("dense label keys changed across counterfactual pair")
    return sorted(path for path in first_flat if first_flat[path] != second_flat[path])


def flatten_entities(entities: JsonDict) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for collection in ("tiles", "nodes", "edges", "ports"):
        for row in entities[collection]:
            for attribute, value in row.items():
                if attribute != "id":
                    flattened[f"{collection}/{row['id']}/{attribute}"] = value
    return flattened


def value_at_label_path(entities: JsonDict, path: str) -> Any:
    collection, entity_id, attribute = path.split("/")
    for row in entities[collection]:
        if row["id"] == entity_id:
            return row[attribute]
    raise KeyError(path)


def validate_split_files(output_dir: Path, rows: Sequence[JsonDict]) -> None:
    expected = {
        split: sorted(row["sample_id"] for row in rows if row["split"] == split)
        for split in ("train", "validation", "test")
    }
    for split, sample_ids in expected.items():
        split_rows = read_jsonl(output_dir / "splits" / f"{split}.jsonl")
        if sorted(row["sample_id"] for row in split_rows) != sample_ids:
            raise ValueError(f"split manifest is stale: {split}")
        expected_rows = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: row["sample_id"],
        )
        if sorted(split_rows, key=lambda row: row["sample_id"]) != expected_rows:
            raise ValueError(f"split manifest rows disagree with the main manifest: {split}")


def load_leakage_ledger(path: Path) -> tuple[JsonDict, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark leakage ledger is missing: {path}")
    payload = json.loads(path.read_text())
    game_ids = payload.get("benchmark_game_ids")
    if not isinstance(game_ids, list) or not game_ids:
        raise ValueError("benchmark leakage ledger has no game IDs")
    return payload, {str(game_id) for game_id in game_ids}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def repository_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
