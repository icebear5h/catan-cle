"""Compose the approved mixed 128-step continuation from existing local images."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from data_pipeline.board_recognition.full_board_readout import board_answer
from data_pipeline.board_recognition.replay_dataset import (
    dense_labels,
    read_jsonl,
    static_board_facts,
    write_json,
    write_jsonl,
)
from data_pipeline.board_recognition.sources import (
    canonical_sha256,
    file_sha256,
    validate_public_board_contract,
    visible_board_facts,
)
from data_pipeline.board_recognition.spatial_robber import spatial_query_bank
from evals.catan_board_bench.tokens import atlas_tokens
from sft.board_state_readout import score_board_state
from sft.spatial_tasks import (
    atlas_node_graph,
    dice_production,
    local_node_tiles,
    node_tile_tokens,
    score_spatial_task,
    shortest_node_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _varied_states(rows: list[dict], rng: random.Random) -> list[dict]:
    """Cycle densities, choosing the least-used available board map each time."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: r["state_id"]):
        groups[row["density_bin"]][row["board_map_sha256"]].append(row)
    densities = sorted(groups)
    rng.shuffle(densities)
    maps = sorted({r["board_map_sha256"] for r in rows})
    rng.shuffle(maps)
    rank = {key: index for index, key in enumerate(maps)}
    counts = Counter()
    for buckets in groups.values():
        for bucket in buckets.values():
            rng.shuffle(bucket)
    result = []
    while any(groups.values()):
        for density in densities:
            buckets = groups[density]
            if not buckets:
                continue
            key = min(buckets, key=lambda k: (counts[k], rank[k]))
            result.append(buckets[key].pop())
            counts[key] += 1
            if not buckets[key]:
                del buckets[key]
    return result


def _path_pairs(seed: int) -> tuple[dict, dict]:
    """Partition unordered endpoints, not rendered routes or their reversals."""
    graph = atlas_node_graph()
    nodes = sorted(graph)
    distances = {pair: len(shortest_node_path(*pair)) - 1 for pair in combinations(nodes, 2)}
    info = {}
    for end in nodes:
        distance = {
            node: distances[tuple(sorted((node, end)))] if node != end else 0 for node in nodes
        }
        counts = {end: 1}
        # Count shortest-route multiplicity for coverage, without another path oracle.
        for node in sorted(nodes, key=lambda n: (distance[n], n)):
            if node != end:
                counts[node] = sum(
                    counts[n] for n in graph[node] if distance[n] == distance[node] - 1
                )
            if node < end:
                info[(node, end)] = {
                    "distance": distance[node],
                    "shortest_path_count": counts[node],
                }
    strata = defaultdict(list)
    for pair, facts in sorted(info.items()):
        strata[(facts["distance"], facts["shortest_path_count"] > 1)].append(pair)
    rng = random.Random(seed)
    membership = {split: [] for split in ("train", "validation", "test")}
    for pairs in strata.values():
        rng.shuffle(pairs)
        if len(pairs) < 3:
            raise ValueError("path stratum cannot cover all three splits")
        heldout = max(1, len(pairs) // 5)
        membership["validation"].extend(pairs[:heldout])
        membership["test"].extend(pairs[heldout : 2 * heldout])
        membership["train"].extend(pairs[2 * heldout :])
    return {split: sorted(pairs) for split, pairs in membership.items()}, info


def _select_paths(pairs: list[tuple], info: dict, count: int, rng: random.Random) -> list[dict]:
    strata = defaultdict(list)
    for pair in pairs:
        facts = info[pair]
        strata[(facts["distance"], facts["shortest_path_count"] > 1)].append(pair)
    keys = sorted(strata)
    rng.shuffle(keys)
    for bucket in strata.values():
        rng.shuffle(bucket)
    result = []
    while len(result) < count:
        if not any(strata.values()):
            raise ValueError("insufficient distinct path endpoints")
        for key in keys:
            if strata[key] and len(result) < count:
                start, end = strata[key].pop()
                if rng.randrange(2):
                    start, end = end, start
                result.append({"start": start, "end": end})
    return result


def _query(task: str, target: dict, contract: dict) -> dict:
    if task == "node_tiles":
        prompt = (
            STATIC_GUIDANCE + f"Which land tiles touch node {target['node']}? "
            "Output only the touching tile tokens, separated by spaces, in any order. "
            "Include every touching tile exactly once; no explanation."
        )
        answer = " ".join(node_tile_tokens(target["node"]))
    elif task == "shortest_node_path":
        prompt = (
            STATIC_GUIDANCE + f"Give a shortest node path from {target['start']} "
            f"to {target['end']} along atlas edges. Output only node tokens separated "
            "by spaces, in route order, including both endpoints. Any equally short "
            "valid path is accepted; no explanation."
        )
        answer = " ".join(shortest_node_path(target["start"], target["end"]))
    elif task == "local_node_tiles":
        prompt = (
            f"Read the land tiles touching node {target['node']} in this board image. "
            "Output only a JSON object mapping every touching tile token to an object "
            'with exactly the keys "resource" and "number". Use lowercase resource '
            'names, "desert" for desert, integer dice numbers, and null for the '
            "desert number. Include no other tiles. No explanation."
        )
        answer = json.dumps(local_node_tiles(contract, target["node"]), sort_keys=True)
    elif task == "dice_production":
        color = target["color"].lower().replace("_", " ")
        prompt = (
            f"For the {color} player, what resources are produced on a dice roll of "
            f"{target['roll']} on this board? Count one per settlement and two per city "
            "on matching tiles; the robber blocks its tile. Ignore bank shortages. "
            'Output only a JSON object with exactly the five keys "wood", "brick", '
            '"sheep", "wheat", "ore", and integer counts, including zeros. No explanation.'
        )
        answer = json.dumps(dice_production(contract, target["color"], target["roll"]))
    else:
        raise ValueError(f"unknown new task: {task}")
    return {"task_type": task, "target": target, "prompt": prompt, "answer": answer}


def _bank_queries(rng: random.Random) -> dict[str, list[dict]]:
    bank = spatial_query_bank()
    selected = {"directions": [], "adjacency_connectivity": []}
    for task, quota in BANK_QUOTAS.items():
        pool = bank[task]
        if task.endswith("_token"):
            pools = [
                [q for q in pool if q["answer"] == q["tokens"][position]] for position in (0, 1)
            ]
        else:
            pools = [pool]
        chosen = []
        for bucket in pools:
            bucket = list(bucket)
            rng.shuffle(bucket)
            needed = quota // len(pools)
            if len(bucket) < needed:
                raise ValueError(f"insufficient corrected bank queries: {task}")
            chosen.extend(bucket[:needed])
        family = "directions" if "direction" in task else "adjacency_connectivity"
        for source in chosen:
            guidance = (
                "Output only one of the two listed tokens; no explanation."
                if task.endswith("_token")
                else "Answer only yes or no; no explanation."
            )
            selected[family].append(
                {
                    **source,
                    "task_type": task,
                    "target": {"tokens": source["tokens"], "relationship": source["relationship"]},
                    "prompt": STATIC_GUIDANCE + source["prompt"] + " " + guidance,
                    "bank_prompt": source["prompt"],
                }
            )
    # Distribute each subtype across the 16 batches instead of clustering its quota.
    for family, queries in selected.items():
        buckets = defaultdict(list)
        for query in queries:
            key = query["task_type"]
            if key.endswith("_token"):
                key += str(query["tokens"].index(query["answer"]))
            buckets[key].append(query)
        keys = sorted(buckets)
        rng.shuffle(keys)
        ordered = []
        while any(buckets.values()):
            for key in keys:
                if buckets[key]:
                    ordered.append(buckets[key].pop())
        selected[family] = ordered
    return selected


def _production_options(contract: dict, rng: random.Random) -> list[dict]:
    options = []
    for player in sorted(contract["players"], key=lambda p: p["color"]):
        color = player["color"]
        buildings = [n for n in contract["nodes"] if n["color"] == color and n["building"]]
        for roll in (2, 3, 4, 5, 6, 8, 9, 10, 11, 12):
            target = {"color": color, "roll": roll}
            query = _query("dice_production", target, contract)
            matching = [t for t in contract["tiles"] if t["number"] == roll]
            city = any(
                n["building"] == "CITY"
                and not t["has_robber"]
                and t["token"] in node_tile_tokens(n["token"])
                for n in buildings
                for t in matching
            )
            robber = any(
                t["has_robber"] and t["token"] in node_tile_tokens(n["token"])
                for n in buildings
                for t in matching
            )
            total = sum(json.loads(query["answer"]).values())
            query["production"] = {
                "zero": total == 0,
                "city_relevant": city,
                "robber_relevant": robber,
                "total": total,
            }
            options.append(query)
    rng.shuffle(options)
    return options


def _select_production(states: list[dict], count: int, load_contract, rng: random.Random) -> list:
    selected, used, options = [], set(), {}
    counts = Counter()
    # Reserve genuinely city/robber-sensitive cases before filling zero/nonzero quotas.
    for criterion, quota in (
        ("city_relevant", count // 8),
        ("robber_relevant", count // 8),
        ("nonzero", count // 2),
        ("zero", count // 2),
    ):
        for state in states:
            if counts[criterion] >= quota:
                break
            sid = state["state_id"]
            if sid in used:
                continue
            if sid not in options:
                options[sid] = _production_options(load_contract(state), rng)
            for query in options[sid]:
                facts = query["production"]
                polarity = "zero" if facts["zero"] else "nonzero"
                qualifies = polarity == criterion or facts.get(criterion, False)
                if not qualifies or counts[polarity] >= count // 2:
                    continue
                selected.append((state, query))
                used.add(sid)
                counts[polarity] += 1
                counts["city_relevant"] += facts["city_relevant"]
                counts["robber_relevant"] += facts["robber_relevant"]
                break
        if counts[criterion] < quota:
            raise ValueError(
                f"insufficient real production cases: {criterion} {counts[criterion]}/{quota}"
            )
    if len(selected) != count:
        raise AssertionError("production quotas drifted")
    rng.shuffle(selected)
    return selected


def _row(state: dict, family: str, query: dict, split: str) -> dict:
    task = query["task_type"]
    query_hash = canonical_sha256({"task": task, "target": query["target"]})
    row_id = f"{VERSION}/{split}/{family}/{state['state_id']}/{query_hash}"
    metadata = {
        "task_type": task,
        "training_family": family,
        "target": query["target"],
        "split": split,
        "state_id": state["state_id"],
        "layout_id": state["layout_id"],
        "density_bin": state["density_bin"],
        "board_map_sha256": state["board_map_sha256"],
        "source_readout_row_id": state["row_id"],
        "query_sha256": query_hash,
    }
    for key in ("production", "bank_prompt", "tokens", "relationship", "polarity"):
        if key in query:
            metadata[key] = query[key]
    return {
        "schema": "catan_spatial_continuation_row/v1",
        "id": row_id,
        "row_id": row_id,
        "task_type": task,
        "training_family": family,
        "images": state["images"],
        "messages": [
            {"role": "user", "content": "<image>\n" + query["prompt"]},
            {"role": "assistant", "content": query["answer"]},
        ],
        "metadata": metadata,
    }


def _coverage(rows: list[dict]) -> dict:
    metadata = [row["metadata"] for row in rows]
    return {
        "rows": len(rows),
        "unique_images": len({r["images"][0] for r in rows}),
        "unique_image_hashes": len({m["provenance"]["sha256"]["image"] for m in metadata}),
        "unique_layouts": len({m["layout_id"] for m in metadata}),
        "unique_board_maps": len({m["board_map_sha256"] for m in metadata}),
        "by_density": dict(sorted(Counter(m["density_bin"] for m in metadata).items())),
    }


def build_dataset(output_dir: Path, *, root: Path = DEFAULT_ROOT, seed: int = 45) -> dict:
    """Write a new immutable local dataset; return its launcher input manifest."""
    output, root = Path(output_dir).resolve(), Path(root).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    readout_root = root / "full_board_readout_v1"
    image_root = readout_root / "images"
    inventory_path = readout_root / "trainable_tokens.json"
    source_paths = {
        "manifest": root / "manifest.jsonl",
        "token_inventory": inventory_path,
        "frozen_board_eval": FROZEN_BOARD_EVAL,
        "builder": Path(__file__).resolve(),
        "spatial_tasks": PROJECT_ROOT / "sft/spatial_tasks.py",
        "spatial_query_bank": PROJECT_ROOT / "data_pipeline/board_recognition/spatial_robber.py",
    }
    for split in ("train", "validation", "test"):
        source_paths[split] = readout_root / "stage1" / f"{split}.jsonl"
    source_hashes = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in source_paths.items()
    }
    inventory = json.loads(inventory_path.read_text())
    if not set(atlas_tokens()) <= set(inventory["tokens"]):
        raise ValueError("trained token inventory is missing literal atlas tokens")
    manifest_rows = read_jsonl(source_paths["manifest"])
    manifest = {s["sample_id"]: s for s in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("duplicate manifest sample_id")
    sources = {}
    for split in ("train", "validation", "test"):
        rows = read_jsonl(source_paths[split])
        if len({r["state_id"] for r in rows}) != len(rows):
            raise ValueError(f"duplicate readout state_id: {split}")
        for row in rows:
            state = manifest[row["state_id"]]
            if state["split"] != split or row["split"] != split:
                raise ValueError(f"readout/manifest split mismatch: {row['state_id']}")
            if row["task_type"] != "full_board_readout":
                raise ValueError("source row is not a full-board readout")
            if row["images"] != [Path(state["image_path"]).name]:
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
        split: {s["board_map_sha256"] for s in manifest_rows if s["split"] == split}
        for split in ("train", "validation", "test", "color_diagnostic")
    }
    if split_maps["validation"] & split_maps["test"]:
        raise ValueError("validation/test board-map leakage")
    excluded_maps = set.union(*(maps for split, maps in split_maps.items() if split != "train"))
    train_sources = [r for r in sources["train"] if r["board_map_sha256"] not in excluded_maps]
    if len(train_sources) < 1024:
        raise ValueError("insufficient distinct non-held-out training images")
    contracts, verified = {}, {}

    def load_contract(row: dict) -> dict:
        sid = row["state_id"]
        if sid in contracts:
            return contracts[sid]
        state = manifest[sid]
        paths = {
            "contract": root / state["contract_path"],
            "image": root / state["image_path"],
            "labels": root / state["label_path"],
        }
        hashes = {kind: file_sha256(path) for kind, path in paths.items()}
        if hashes != state["sha256"]:
            raise ValueError(f"manifest asset hash mismatch: {sid}")
        common_image = image_root / row["images"][0]
        if file_sha256(common_image) != hashes["image"]:
            raise ValueError(f"common-root image hash mismatch: {sid}")
        contract = json.loads(paths["contract"].read_text())
        validate_public_board_contract(contract)
        if canonical_sha256(static_board_facts(contract)) != state["board_map_sha256"]:
            raise ValueError(f"board-map hash mismatch: {sid}")
        if canonical_sha256(visible_board_facts(contract)) != state["board_fact_sha256"]:
            raise ValueError(f"board-fact hash mismatch: {sid}")
        if dense_labels(contract, sample_id=sid) != json.loads(paths["labels"].read_text()):
            raise ValueError(f"dense labels disagree with contract: {sid}")
        if board_answer(contract) != row["messages"][-1]["content"]:
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

    rng = random.Random(seed)
    ordered = _varied_states(train_sources, rng)
    validation_order = _varied_states(sources["validation"], random.Random(seed + 1))
    membership, path_info = _path_pairs(seed)
    nodes = sorted(atlas_node_graph())
    rng.shuffle(nodes)
    queries = _bank_queries(rng)
    queries["node_tiles"] = [{"node": nodes[i % len(nodes)]} for i in range(128)]
    queries["shortest_node_path"] = _select_paths(membership["train"], path_info, 128, rng)
    queries["local_node_tiles"] = [{"node": nodes[i % len(nodes)]} for i in range(128)]
    families = {family: [] for family in STEP_QUOTAS}
    # Retention gets first choice of 256 distinct, density-balanced board maps/images.
    for state in ordered[:256]:
        load_contract(state)
        row = copy.deepcopy(state)
        row.pop("curriculum_stage", None)
        row_id = f"{VERSION}/train/full_board_readout/{state['state_id']}/readout"
        row.update(id=row_id, row_id=row_id, training_family="full_board_readout")
        row["metadata"] = {
            **row.get("metadata", {}),
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
    panels = {}
    panel_rng = random.Random(seed + 2)
    for task in NEW_PANELS:
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

    blocks = []
    for block in range(16):
        batches = []
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
    train, step_order = [], []
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
                row["metadata"].update(optimizer_step=step, block_id=block_id)
            train.extend(batch)
    all_rows = train + [row for rows in panels.values() for row in rows]
    if len({r["row_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("duplicate versioned row IDs")
    if any("curriculum_stage" in r for r in all_rows):
        raise AssertionError("curriculum_stage must be uniformly omitted")
    selected_ids = {r["metadata"]["state_id"] for r in all_rows}
    for row in all_rows:
        row["metadata"]["provenance"] = verified[row["metadata"]["state_id"]]
        answer = row["messages"][-1]["content"]
        score = score_spatial_task(answer, answer, row["metadata"])
        if row["task_type"] == "full_board_readout":
            score = score_board_state(answer, answer)
        if score is not None and not score["correct"]:
            raise ValueError(f"label failed scorer: {row['row_id']}")
        if row["task_type"] == "shortest_node_path":
            target = row["metadata"]["target"]
            row["metadata"]["path"] = path_info[tuple(sorted((target["start"], target["end"])))]
    path_metadata = {}
    for split, pairs in membership.items():
        selected = [
            tuple(sorted(r["metadata"]["target"].values()))
            for r in (
                train
                if split == "train"
                else panels["shortest_node_path"] if split == "validation" else []
            )
            if r["task_type"] == "shortest_node_path"
        ]
        path_metadata[split] = {
            "pairs": [list(p) for p in pairs],
            "selected_pairs": [list(p) for p in selected],
            "selected_by_distance": dict(
                sorted(Counter(str(path_info[p]["distance"]) for p in selected).items())
            ),
            "selected_tied": sum(path_info[p]["shortest_path_count"] > 1 for p in selected),
        }
    metadata = {
        "schema": "catan_spatial_continuation/v1",
        "seed": seed,
        "optimizer_steps": 128,
        "batch_size": 8,
        "step_quotas": STEP_QUOTAS,
        "row_quotas": {k: v * 8 for k, v in STEP_QUOTAS.items()},
        "bank_quotas": BANK_QUOTAS,
        "step_order": step_order,
        "global_row_shuffle": False,
        "source_hashes": source_hashes,
        "image_root": str(image_root),
        "token_inventory": str(inventory_path),
        "token_measurement": "deferred_to_launcher_cpu_preflight",
        "splits": {
            "board_map_sha256": {k: sorted(v) for k, v in split_maps.items()},
            "excluded_train_states": len(sources["train"]) - len(train_sources),
            "source_states": {k: len(v) for k, v in sources.items()},
            "path_pairs": path_metadata,
            "path_partition": "canonical unordered pairs stratified by distance and ties; 60/20/20 with minimum one per held-out stratum",
            "novelty": "Only path endpoint pairs are query-disjoint; directions and node_tiles are fixed-atlas recall.",
        },
        "coverage": {
            "train": _coverage(train),
            "training_families": {k: _coverage(v) for k, v in families.items()},
            "panels": {k: _coverage(v) for k, v in panels.items()},
            "all_unique_images": len({r["images"][0] for r in all_rows}),
            "all_unique_image_hashes": len(
                {r["metadata"]["provenance"]["sha256"]["image"] for r in all_rows}
            ),
            "production": {
                split: {
                    key: sum(r["metadata"]["production"][key] for r in rows)
                    for key in ("zero", "city_relevant", "robber_relevant")
                }
                for split, rows in (
                    ("train", families["dice_production"]),
                    ("validation", panels["dice_production"]),
                )
            },
        },
        "selected_assets": [verified[sid] for sid in sorted(selected_ids)],
    }
    # Do all source validation and quota construction before creating any output.
    output.mkdir(parents=True, exist_ok=False)
    files = {"train": (output / "train.jsonl", train)}
    files.update(
        {label: (output / "panels" / f"{label}.jsonl", rows) for label, rows in panels.items()}
    )
    metadata["files"] = {}
    for label, (path, rows) in files.items():
        write_jsonl(path, rows)
        metadata["files"][label] = {
            "path": str(path.relative_to(output)),
            "rows": len(rows),
            "sha256": file_sha256(path),
        }
    write_json(output / "metadata.json", metadata)
    inputs = {
        "schema": "catan_spatial_continuation_inputs/v1",
        "train_jsonl": str(output / "train.jsonl"),
        "image_root": str(image_root),
        "token_inventory": str(inventory_path),
        "new_panels": {
            label: {
                "eval_jsonl": str(path),
                "image_root": str(image_root),
                "max_new_tokens": 128,
                "batch_size": 16,
            }
            for label, (path, _) in files.items()
            if label != "train"
        },
        "metadata": str(output / "metadata.json"),
    }
    write_json(output / "dataset_inputs.json", inputs)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output_dir, root=args.root, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
