"""Build the node and edge readout curriculum (``node_edge_readout_v1``).

Stage 3 of the gaussian ladder asks one question: what is at each node and
edge? Every real replay board already on disk gets the production forward
prompts, token as query only, plus one complete readout per family that
walks every token in a fixed order so the model practices answering a whole
board without omissions:

- ``node_occupancy``: ``<N17> building?`` -> ``red settlement``, ``red city``
  or ``empty``
- ``edge_owner``: ``<E17_18> road?`` -> ``blue road`` or ``empty``
- ``node_readout``: an explicit instruction naming the 54 nodes, the item
  format, the order and the separator ->
  ``<N00> empty; <N01> red settlement; ...; <N53> empty``
- ``edge_readout``: the same for the 72 edges ->
  ``<E00_01> empty; <E00_20> red road; ...; <E52_53> empty``

Readouts list every location, empties included, for the same reason the
terrain readout lists the desert: a fixed count and order is a stopping
criterion, an occupied-only list is not. Train images are capped at
``rows_per_family`` occupied and ``rows_per_family`` empty locations per
family, the empties ranked so the confusable ones come first (touching a
same-type piece, touching the other type, two hops out, far). Eval splits
get every node and edge of every image, a full classification per board.

No inverse rows, no tile, port or robber rows: the rung is scoped to nodes
and edges. Splits are inherited from the replay manifest, whole games at a
time, and re-checked pairwise so no layout leaks between any two splits.
The ``color_diagnostic`` split is exported eval-only; it is the only split
where all eleven piece colours appear.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.board_recognition.replay_dataset import (
    board_density,
    file_sha256,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.single_piece_localization import (
    EDGE_PIECE,
    FORWARD_QUERY,
    NEAR_MAX_HOPS,
    _row,
    forward_answer,
    neighbor_distances,
    neighbor_tokens,
)
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _deterministic_shuffle,
    _stable_rank,
    _summarize_rows,
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.terrain_readout import layout_id, link_or_copy


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_node_edge_readout/v1"
ROW_SCHEMA = "catan_node_edge_readout_row/v1"
DEFAULT_OUTPUT_NAME = "node_edge_readout_v1"
GROUNDING_STAGE = "node_edge_readout"
TASK_FAMILY = "node_edge_readout"
SPLITS = ("train", "validation", "test", "color_diagnostic")
FULL_COVERAGE_SPLITS = ("validation", "test", "color_diagnostic")
FAMILIES = ("node", "edge")
CATEGORY = {"node": "node.occupancy", "edge": "edge.owner"}
TASK_TYPE = {"node": "node_occupancy", "edge": "edge_owner"}
READOUT_CATEGORY = {"node": "node.readout", "edge": "edge.readout"}
READOUT_TASK_TYPE = {"node": "node_readout", "edge": "edge_readout"}
NODE_COUNT = 54
EDGE_COUNT = 72
NODE_READOUT_PROMPT = (
    "List every node <N00> to <N53> as \"token building\", where building is \"empty\", "
    "\"colour settlement\" or \"colour city\", in token order, separated by \"; \"."
)
EDGE_READOUT_PROMPT = (
    "List every edge <E00_01> to <E52_53> as \"token road\", where road is \"empty\" or "
    "\"colour road\", in token order, separated by \"; \"."
)
READOUT_PROMPT = {"node": NODE_READOUT_PROMPT, "edge": EDGE_READOUT_PROMPT}
ROWS_PER_FAMILY = 4
READOUTS_PER_FAMILY = 1
# Empty locations are drawn by kind, hardest first; leftovers fill from the
# ranked pool in kind order when a kind runs short.
EMPTY_KINDS = ("adjacent", "cross_type", "hop2", "hop3", "far")
EMPTY_QUOTA = {"adjacent": 2, "cross_type": 1, "far": 1}
EMPTY_ANSWER = "empty"


def board_pieces(contract: JsonDict) -> dict[str, dict[str, JsonDict]]:
    """Occupied locations per family: token -> {"piece", "color", "answer"}."""

    nodes = {
        node["token"]: {"piece": node["building"], "color": node["color"], "answer": forward_answer(node["color"], node["building"])}
        for node in contract["nodes"]
        if node.get("building") is not None
    }
    edges = {
        edge["token"]: {"piece": EDGE_PIECE, "color": edge["road_color"], "answer": forward_answer(edge["road_color"], EDGE_PIECE)}
        for edge in contract["edges"]
        if edge.get("road_color") is not None
    }
    return {"node": nodes, "edge": edges}


def family_tokens(contract: JsonDict) -> dict[str, list[str]]:
    """Every node and edge token in atlas order (sorted token strings)."""

    tokens = {"node": sorted(node["token"] for node in contract["nodes"]), "edge": sorted(edge["token"] for edge in contract["edges"])}
    if len(tokens["node"]) != NODE_COUNT or len(tokens["edge"]) != EDGE_COUNT:
        raise SpatialLocalizationError(f"expected {NODE_COUNT} nodes and {EDGE_COUNT} edges, got {len(tokens['node'])} and {len(tokens['edge'])}")
    return tokens


def cross_type_tokens(contract: JsonDict, pieces: dict[str, dict[str, JsonDict]]) -> dict[str, set[str]]:
    """Locations touching a piece of the other family: road ends for nodes, building corners for edges."""

    nodes = {node["token"] for node in contract["nodes"] if any(edge in pieces["edge"] for edge in node["adjacent_edge_tokens"])}
    edges = {edge["token"] for edge in contract["edges"] if any(node in pieces["node"] for node in edge["node_tokens"])}
    return {"node": nodes, "edge": edges}


def empty_candidates(contract: JsonDict, family: str, *, pieces: dict[str, dict[str, JsonDict]] | None = None, neighbors: dict[str, list[str]] | None = None) -> list[JsonDict]:
    """Every empty location of ``family`` with its kind and hop distance to the nearest same-type piece."""

    pieces = pieces if pieces is not None else board_pieces(contract)
    neighbors = neighbors if neighbors is not None else neighbor_tokens(contract)
    touching_other = cross_type_tokens(contract, pieces)[family]
    occupied = pieces[family]
    candidates = []
    for token in family_tokens(contract)[family]:
        if token in occupied:
            continue
        distances = neighbor_distances(neighbors, token, NEAR_MAX_HOPS)
        near = [hops for other, hops in distances.items() if other in occupied]
        distance: int | str = min(near) if near else "far"
        if distance == 1:
            kind = "adjacent"
        elif token in touching_other:
            kind = "cross_type"
        elif distance == 2:
            kind = "hop2"
        elif distance == 3:
            kind = "hop3"
        else:
            kind = "far"
        candidates.append({"token": token, "kind": kind, "distance": distance})
    return candidates


def sample_occupied(sample_id: str, family: str, occupied: dict[str, JsonDict], count: int) -> list[str]:
    ordered = sorted(occupied, key=lambda token: (_stable_rank(sample_id, "occupied", family, token), token))
    return ordered[:count]


def sample_empties(sample_id: str, family: str, candidates: Sequence[JsonDict], count: int) -> list[JsonDict]:
    """Draw ``count`` empties by kind quota, hardest kinds first, filling leftovers in kind order."""

    ranked = sorted(candidates, key=lambda item: (EMPTY_KINDS.index(item["kind"]), _stable_rank(sample_id, "empty", family, item["token"]), item["token"]))
    chosen: list[JsonDict] = []
    for kind, quota in EMPTY_QUOTA.items():
        chosen.extend([item for item in ranked if item["kind"] == kind and item not in chosen][:quota])
    for item in ranked:
        if len(chosen) >= count:
            break
        if item not in chosen:
            chosen.append(item)
    return chosen[:count]


def readout_answer(family: str, tokens: Sequence[str], occupied: dict[str, JsonDict]) -> str:
    """Every location of the family in token order, empties explicit, one canonical string."""

    return "; ".join(f"{token} {occupied[token]['answer'] if token in occupied else EMPTY_ANSWER}" for token in tokens)


def rows_for_state(
    state: JsonDict,
    contract: JsonDict,
    *,
    full_coverage: bool,
    rows_per_family: int = ROWS_PER_FAMILY,
    readouts_per_family: int = READOUTS_PER_FAMILY,
) -> list[JsonDict]:
    """Short rows for the sampled (or every) node and edge, then the two readouts."""

    image_name = Path(state["image_path"]).name
    buildings, roads, density = board_density(contract)
    pieces = board_pieces(contract)
    tokens = family_tokens(contract)
    neighbors = neighbor_tokens(contract)
    base = {
        "split": state["split"],
        "state_id": state["sample_id"],
        "layout_id": layout_id(state["sample_id"]),
        "density_bin": state.get("density_bin") or density,
        "piece_count": buildings + roads,
        "color_heldout": False,
    }
    common = {"image_name": image_name, "spatial_target": None, "schema": ROW_SCHEMA, "grounding_stage": GROUNDING_STAGE, "task_family": TASK_FAMILY}
    rows: list[JsonDict] = []
    for family in FAMILIES:
        occupied = pieces[family]
        empties = empty_candidates(contract, family, pieces=pieces, neighbors=neighbors)
        if full_coverage:
            occupied_tokens = list(occupied)
            chosen_empties = empties
        else:
            occupied_tokens = sample_occupied(state["sample_id"], family, occupied, rows_per_family)
            chosen_empties = sample_empties(state["sample_id"], family, empties, rows_per_family)
        for token in tokens[family]:
            if token in occupied_tokens:
                metadata = {**base, "entity_type": family, "target_token": token, "queried_token": token, "piece": occupied[token]["piece"], "color": occupied[token]["color"]}
                rows.append(_row(row_id=f"{state['sample_id']}_{token[1:-1]}_{TASK_TYPE[family]}", prompt=f"{token} {FORWARD_QUERY[family]}", answer=occupied[token]["answer"], task_type=TASK_TYPE[family], category=CATEGORY[family], polarity="positive", metadata=metadata, **common))
        for item in chosen_empties:
            token = item["token"]
            metadata = {**base, "entity_type": family, "target_token": token, "queried_token": token, "piece": "EMPTY", "color": "none", "negative_kind": item["kind"], "negative_distance": item["distance"]}
            rows.append(_row(row_id=f"{state['sample_id']}_{token[1:-1]}_{TASK_TYPE[family]}", prompt=f"{token} {FORWARD_QUERY[family]}", answer=EMPTY_ANSWER, task_type=TASK_TYPE[family], category=CATEGORY[family], polarity="hard_negative", metadata=metadata, **common))
    for family in FAMILIES:
        for index in range(readouts_per_family):
            metadata = {**base, "entity_type": "board", "target_token": tokens[family][0], "piece": "BOARD", "color": "none", "item_count": len(tokens[family]), "occupied_count": len(pieces[family])}
            rows.append(_row(row_id=f"{state['sample_id']}_{family}_readout_{index}", prompt=READOUT_PROMPT[family], answer=readout_answer(family, tokens[family], pieces[family]), task_type=READOUT_TASK_TYPE[family], category=READOUT_CATEGORY[family], polarity="positive", metadata=metadata, **common))
    return rows


def export_node_edge_readout(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    rows_per_family: int = ROWS_PER_FAMILY,
    readouts_per_family: int = READOUTS_PER_FAMILY,
    splits: Sequence[str] = SPLITS,
    validate_dataset: bool = True,
) -> JsonDict:
    """Export the node and edge rows; replay images are hard-linked into ``<output>/images``."""

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
    layouts_by_split: dict[str, set[str]] = {split: set() for split in splits}
    for state in states:
        layouts_by_split[state["split"]].add(layout_id(state["sample_id"]))
    for left, right in combinations(splits, 2):
        overlap = layouts_by_split[left] & layouts_by_split[right]
        if overlap:
            raise SpatialLocalizationError(f"layouts appear in both {left} and {right}: {sorted(overlap)[:5]}")

    files: dict[str, JsonDict] = {}
    for split in splits:
        rows: list[JsonDict] = []
        for state in states:
            if state["split"] != split:
                continue
            rows.extend(rows_for_state(state, contracts[state["sample_id"]], full_coverage=split in FULL_COVERAGE_SPLITS, rows_per_family=rows_per_family, readouts_per_family=readouts_per_family))
        if not rows:
            continue
        if split == "train":
            rows = _deterministic_shuffle(rows, "node_edge_readout")
        path = output / "stage1" / f"{split}.jsonl"
        _write_jsonl(path, rows)
        summary = _summarize_rows(rows)
        summary["dimensions"]["category"] = dict(sorted(Counter(row["category"] for row in rows).items()))
        summary["dimensions"]["density_bin"] = dict(sorted(Counter(str(row.get("density_bin")) for row in rows).items()))
        summary["dimensions"]["negative_kind"] = dict(sorted(Counter(str(row["negative_kind"]) for row in rows if "negative_kind" in row).items()))
        summary["dimensions"]["color"] = dict(sorted(Counter(row["color"] for row in rows if row["polarity"] == "positive" and row["entity_type"] != "board").items()))
        summary["unique_images"] = len({row["images"][0] for row in rows})
        summary["layouts"] = len(layouts_by_split[split])
        summary["full_coverage"] = split in FULL_COVERAGE_SPLITS
        files[f"stage1/{split}.jsonl"] = {**summary, "sha256": file_sha256(path)}

    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "image_root": str(images_dir),
        "rows_per_family": rows_per_family,
        "readouts_per_family": readouts_per_family,
        "empty_quota": dict(EMPTY_QUOTA),
        "empty_kinds": list(EMPTY_KINDS),
        "full_coverage_splits": [split for split in splits if split in FULL_COVERAGE_SPLITS],
        "readout_prompts": dict(READOUT_PROMPT),
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
    parser.add_argument("--rows-per-family", type=int, default=ROWS_PER_FAMILY, help="Occupied and empty locations sampled per family per train image.")
    parser.add_argument("--readouts-per-family", type=int, default=READOUTS_PER_FAMILY)
    args = parser.parse_args(argv)
    result = export_node_edge_readout(
        args.dataset_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        rows_per_family=args.rows_per_family,
        readouts_per_family=args.readouts_per_family,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
