"""Prepare an immutable, CPU/offline symbolic atlas pilot from full_board_diverse_v1.

CLI: --dry-run audits sources and constructs/scorers all rows without writing;
--validate verifies a previously built output and its pinned source files.
Default execution creates a new output only after validation. No overwrite mode.
Internal build API: build_dataset(output_dir, root=..., seed=..., dry_run=False,
checkpoint=None). The checkpoint is a proposed paired-experiment input, not loaded
or selected by this builder; final training budget/configuration belongs to integration.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

from data_pipeline.board_recognition.full_board_readout import board_answer
from data_pipeline.board_recognition.replay_dataset import dense_labels, static_board_facts
from data_pipeline.board_recognition.sources import (
    DEFAULT_LEAKAGE_LEDGER, DEFAULT_SOURCE_LOCK, canonical_sha256, file_sha256,
    load_leakage_ledger, source_lock_matches_metadata, validate_replay_source_lock,
    visible_board_facts,
)
from evals.catan_board_bench.tokens import atlas_tokens
from sft.symbolic_board_tasks import (
    DIRECTIONS, OFFSETS, RESOURCES_LOWER, STATIC_TASKS, TRAIN_TASKS,
    TRANSFER_TASKS, PhysicalStateError, atlas_geometry, decode_state, owned_route,
    score_symbolic_task, strict_json, symbolic_answer, symbolic_prompt, symbolic_task_role,
    validate_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "artifacts/generated/board_recognition/full_board_diverse_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/generated/sft/symbolic_board_v2"
VERSION = "symbolic_board_v2"
KNOWN_DIRECTION_TRAINING = (
    PROJECT_ROOT / "artifacts/generated/board_recognition/spatial_continuation_v1/train.jsonl",
)
DENSITIES = ("empty", "setup", "sparse", "dense")
SPLITS = ("train", "validation", "test", "color_diagnostic")
SOURCE_COUNTS = dict(train=5120, validation=64, test=64, color_diagnostic=64)
TRAIN_QUOTAS = {
    "symbolic_direction": 400, "symbolic_direction_choice": 400,
    "symbolic_neighbors": 240, "symbolic_incidence": 400, "symbolic_oriented_step": 160,
    "symbolic_owned_nodes": 160, "symbolic_owned_roads": 160,
    "symbolic_piece_owner": 160, "symbolic_owned_incident_roads": 240,
    "symbolic_reachable": 160, "symbolic_shortest_route": 240,
    "symbolic_near_nodes": 120, "symbolic_near": 120,
    "symbolic_local_constraint": 160, "symbolic_scene_tiles": 80,
}
EVAL_QUOTAS = {task: 32 for task in TRAIN_QUOTAS}
TOKEN_PATTERN = re.compile(r"<[NTEP][0-9_]+>")


def read_json(path: Path) -> dict:
    # Source metadata contains non-integer audit measurements; duplicate keys still fail.
    return json.loads(path.read_text(), object_pairs_hook=_unique_pairs)


def _unique_pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line, object_pairs_hook=_unique_pairs) for line in handle if line.strip()]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _check(path.is_relative_to(root) and path.is_file(), f"invalid source asset path: {relative}")
    return path


def _hash(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _engine_source_digest(source: dict, colors: list[str]) -> str:
    return canonical_sha256({
        "schema": "catan_board_recognition_engine_policy/v2",
        "game_seed": source["engine_seed"], "policy_seed": source["policy_seed"],
        "colors": colors, "legal_actions_only": True, "victory_points_to_win": 13,
        "action_weights": {"BUILD_CITY": 20, "BUILD_SETTLEMENT": 8, "BUILD_ROAD": 5,
                           "BUY_DEVELOPMENT_CARD": 2, "MARITIME_TRADE": 2, "END_TURN": 1},
        "forced_resources": False, "forced_board_mutation": False,
        "trajectory_action_limit": source["trajectory_action_limit"],
    })


def audit_source_state(root: Path, source: dict, readout: dict) -> tuple[dict, dict]:
    """Pin and join contract/dense labels/readout without opening any image asset.

    PhysicalStateError is an explicit exclusion; broken hashes, identity or joins
    are fatal. Returned provenance contains no image keys or model-visible extras.
    """
    sid = source["sample_id"]
    paths = {"contract": _asset(root, source["contract_path"]),
             "labels": _asset(root, source["label_path"])}
    hashes = {kind: file_sha256(path) for kind, path in paths.items()}
    _check(all(h == source["sha256"][kind] for kind, h in hashes.items()), f"source hash mismatch: {sid}")
    contract = read_json(paths["contract"])
    _check(contract["sample"]["id"] == sid and contract["source"] == source["source"],
           f"source contract identity mismatch: {sid}")
    _check(readout["state_id"] == sid and readout["split"] == source["split"]
           and readout["task_type"] == "full_board_readout", f"readout join mismatch: {sid}")
    _check(readout["images"] == [Path(source["image_path"]).name], f"readout asset identity mismatch: {sid}")
    _check(canonical_sha256(static_board_facts(contract)) == source["board_map_sha256"],
           f"map hash mismatch: {sid}")
    _check(canonical_sha256(visible_board_facts(contract)) == source["board_fact_sha256"],
           f"board fact hash mismatch: {sid}")
    _check(dense_labels(contract, sample_id=sid) == read_json(paths["labels"]), f"dense label mismatch: {sid}")
    _check(board_answer(contract) == readout["messages"][-1]["content"], f"readout answer mismatch: {sid}")
    colors = [p["color"] for p in contract["players"]]
    if source["source"]["kind"] == "engine_rollout":
        _check(_engine_source_digest(source["source"], colors) == source["source"]["source_sha256"],
               f"engine policy hash mismatch: {sid}")
    state = validate_contract(contract)
    provenance = {
        "state_id": sid, "split": source["split"], "layout_id": readout["layout_id"],
        "density_bin": source["density_bin"], "board_map_sha256": source["board_map_sha256"],
        "board_fact_sha256": source["board_fact_sha256"], "source": source["source"],
        "source_readout_row_id": readout["row_id"],
        "paths": {k: str(v) for k, v in paths.items()}, "sha256": hashes,
    }
    return state, provenance


def load_sources(root: Path = DEFAULT_ROOT) -> tuple[dict, dict]:
    """Audit all 5,312 real states and original split/benchmark/source-lock joins."""
    root = root.resolve()
    build = read_json(root / "build.json")
    source_paths = {
        "manifest": root / "manifest.jsonl", "build": root / "build.json",
        "historical_code_hashes": root / "source_hashes.json",
        "readout_metadata": root / "full_board_readout_v1/metadata.json",
        "token_inventory": root / "full_board_readout_v1/trainable_tokens.json",
        "benchmark_ledger": DEFAULT_LEAKAGE_LEDGER, "replay_source_lock": DEFAULT_SOURCE_LOCK,
        "builder": Path(__file__), "oracles": PROJECT_ROOT / "sft/symbolic_board_tasks.py",
        "engine_board": PROJECT_ROOT / "cle/game_engine/models/board.py",
        "engine_map": PROJECT_ROOT / "cle/game_engine/models/map.py",
        "atlas": PROJECT_ROOT / "evals/catan_board_bench/tokens.py",
        "board_answer": PROJECT_ROOT / "data_pipeline/board_recognition/full_board_readout.py",
        "dense_labels": PROJECT_ROOT / "data_pipeline/board_recognition/replay_dataset.py",
        "source_validation": PROJECT_ROOT / "data_pipeline/board_recognition/sources.py",
        "terrain_readout": PROJECT_ROOT / "data_pipeline/board_recognition/terrain_readout.py",
        "piece_readout": PROJECT_ROOT / "data_pipeline/board_recognition/node_edge_readout.py",
        "piece_names": PROJECT_ROOT / "data_pipeline/board_recognition/single_piece_localization.py",
        "board_tokens": PROJECT_ROOT / "cle/game_engine/board_tokens.py",
        "engine_enums": PROJECT_ROOT / "cle/game_engine/models/enums.py",
        "engine_colors": PROJECT_ROOT / "cle/game_engine/models/player.py",
    }
    source_paths.update({f"historical_direction_training_{i}": path
                         for i, path in enumerate(KNOWN_DIRECTION_TRAINING)})
    parent = Path(build["parent"]).resolve()
    source_paths.update(parent_manifest=parent / "manifest.jsonl", parent_metadata=parent / "metadata.json")
    hashes = {name: _hash(path) for name, path in source_paths.items()}
    _check(build["manifest_sha256"] == hashes["manifest"]["sha256"], "source manifest changed")
    _check(build["parent_manifest_sha256"] == hashes["parent_manifest"]["sha256"], "parent manifest changed")
    parent_metadata = read_json(source_paths["parent_metadata"])
    lock = read_json(DEFAULT_SOURCE_LOCK)
    validate_replay_source_lock(lock)
    _check(source_lock_matches_metadata(lock, lock_sha256=parent_metadata["source_lock_sha256"],
                                       file_sha256_value=parent_metadata["source_lock_file_sha256"]),
           "parent source lock changed")
    _, benchmark_ids = load_leakage_ledger()
    _check(len(benchmark_ids) == 13 and benchmark_ids == set(lock["leakage"]["excluded_game_ids"]),
           "expected the pinned 13-game benchmark exclusion ledger")
    _check(hashes["benchmark_ledger"]["sha256"] == parent_metadata["leakage_ledger_sha256"],
           "benchmark ledger hash changed")
    inventory = read_json(source_paths["token_inventory"])
    _check(inventory["tokens"] == atlas_tokens() and inventory["atlas_tokens"] == atlas_tokens(),
           "inventory must be the exact canonical 154-token sequence")
    rows = read_jsonl(source_paths["manifest"])
    manifest = {r["sample_id"]: r for r in rows}
    _check(len(manifest) == len(rows), "duplicate manifest sample identity")
    _check(dict(Counter(r["split"] for r in rows)) == SOURCE_COUNTS, "source split counts changed")
    for row in read_jsonl(source_paths["parent_manifest"]):
        _check(manifest.get(row["sample_id"]) == row, "diverse source changed a replay_v1 parent row")
    readouts = {}
    readout_metadata = read_json(source_paths["readout_metadata"])
    _check(readout_metadata["manifest_sha256"] == hashes["manifest"]["sha256"], "readout manifest changed")
    for split in SPLITS:
        path = root / "full_board_readout_v1/stage1" / f"{split}.jsonl"
        hashes["readout_" + split] = _hash(path)
        _check(file_sha256(path) == build["export"]["files"][split]["sha256"]
               == readout_metadata["files"][split]["sha256"], f"readout file hash mismatch: {split}")
        split_rows = read_jsonl(path)
        _check(len(split_rows) == SOURCE_COUNTS[split] and
               len({r["state_id"] for r in split_rows}) == len(split_rows), "readout count/identity error")
        _check({r["state_id"] for r in split_rows} == {r["sample_id"] for r in rows if r["split"] == split},
               "readout source coverage mismatch")
        readouts.update({r["state_id"]: r for r in split_rows})
    groups = {name: defaultdict(set) for name in ("trajectory", "map", "game", "layout")}
    for row in rows:
        source, split = row["source"], row["split"]
        _check(source["split"] == split, "source split disagreement")
        groups["trajectory"][source["trajectory_id"]].add(split)
        groups["map"][row["board_map_sha256"]].add(split)
        groups["layout"][readouts[row["sample_id"]]["layout_id"]].add(split)
        if source["game_id"] is not None:
            groups["game"][source["game_id"]].add(split)
    for name in ("trajectory", "game", "layout"):
        _check(all(len(splits) == 1 for splits in groups[name].values()), f"cross-split {name} leakage")
    overlapping_maps = set()
    for key, splits in groups["map"].items():
        _check(len(splits - {"train"}) <= 1, "cross-heldout-map leakage")
        if len(splits) > 1:
            overlapping_maps.add(key)
    accepted_lock = {r["game_id"]: r for r in lock["accepted"]}
    replay_hashes = {}
    admitted = {s: [] for s in SPLITS}
    exclusions = []
    physical_valid = Counter()
    for source in rows:
        sid, info = source["sample_id"], source["source"]
        reasons = []
        if info["game_id"] in benchmark_ids:
            reasons.append("benchmark_game")
        elif info["kind"] == "colonist_replay":
            _check(info["game_id"] in accepted_lock, f"unlocked replay source: {sid}")
            path = (PROJECT_ROOT / info["replay_path"]).resolve()
            if str(path) not in replay_hashes:
                replay_hashes[str(path)] = _hash(path)
            _check(replay_hashes[str(path)]["sha256"] == info["source_sha256"]
                   == accepted_lock[info["game_id"]]["sha256"], f"raw replay source changed: {sid}")
        else:
            _check(info["kind"] == "engine_rollout", f"unrecognized real source: {sid}")
        try:
            state, provenance = audit_source_state(root, source, readouts[sid])
            physical_valid[source["split"]] += 1
        except PhysicalStateError as exc:
            reasons.append(str(exc))
        if source["split"] == "color_diagnostic":
            reasons.append("diagnostic_reserved")
        if source["split"] == "train" and source["board_map_sha256"] in overlapping_maps:
            reasons.append("heldout_map_overlap")
        if reasons:
            exclusions.append({"state_id": sid, "split": source["split"], "reasons": reasons,
                               "source": info, "board_map_sha256": source["board_map_sha256"],
                               "contract": _hash(root / source["contract_path"]),
                               "labels": _hash(root / source["label_path"])})
            if reasons == ["diagnostic_reserved"]:
                admitted["color_diagnostic"].append({"state": state, "provenance": provenance})
        else:
            admitted[source["split"]].append({"state": state, "provenance": provenance})
    _check(all(admitted[s] for s in SPLITS[:3]), "source admission emptied a split")
    report = {
        "source_root": str(root), "source_hashes": hashes,
        "raw_replay_hashes": list(replay_hashes.values()), "source_counts": SOURCE_COUNTS,
        "physical_valid_counts": dict(physical_valid),
        "admitted_counts": {s: len(admitted[s]) for s in SPLITS[:3]},
        "reserved_diagnostic_valid_count": len(admitted["color_diagnostic"]),
        "exclusions": exclusions, "benchmark_excluded_game_ids": sorted(benchmark_ids),
        "exclusion_reason_counts": dict(Counter(reason.split(":", 1)[0]
                                                for row in exclusions for reason in row["reasons"])),
        "source_split_groups": {k: {v: sorted(s) for v, s in sorted(g.items())} for k, g in groups.items()},
        "certification": "Exact canonical topology/identity, unique ownership, piece supply caps and building distance only; reachable history NOT certified. Historical caches ignored, never repaired.",
        "asset_checks": "Contract, dense-label and full-readout hashes and semantic joins; raw replay hashes and engine-policy digests. Image files are never opened or hashed.",
    }
    return admitted, report


def known_direction_exposure(paths: tuple[Path, ...] = KNOWN_DIRECTION_TRAINING) -> dict:
    """Union explicitly identified historical training corpora; never infer training from v1."""
    pair_sources = defaultdict(list)
    files = []
    vocabulary = set(atlas_tokens())
    for path in paths:
        files.append(_hash(path))
        for row in read_jsonl(path):
            metadata = row.get("metadata", {})
            task = row.get("task_type", metadata.get("task_type", ""))
            if "direction" not in task or task == "port_direction":
                continue
            target = metadata.get("target", {})
            tokens = target.get("tokens", metadata.get("tokens"))
            if tokens is None and "a" in target and "b" in target:
                tokens = [target["a"], target["b"]]
            _check(isinstance(tokens, list) and len(tokens) == 2 and len(set(tokens)) == 2
                   and set(tokens) <= vocabulary and tokens[0][1] == tokens[1][1]
                   and tokens[0][1] in "NT", "unparsed historical direction training query")
            key = " ".join(sorted(tokens))
            if str(path) not in pair_sources[key]:
                pair_sources[key].append(str(path))
    return {"files": files, "pair_sources": dict(sorted(pair_sources.items())),
            "known_pair_count": len(pair_sources),
            "status": "known_sources_audited_history_incomplete",
            "claim": "New-corpus holdout only. Known historical training pairs moved to train/recall; unobserved checkpoint exposure is not certified."}


def directional_pair_manifest(seed: int, exposed_pairs: tuple | list = ()) -> dict:
    """One partition for unordered pairs, irrespective of axis/inverse/choice order."""
    atlas = atlas_geometry()
    strata = defaultdict(list)
    for family in "NT":
        tokens = sorted(t for t in atlas["positions"] if t[1] == family)
        for a, b in combinations(tokens, 2):
            x, y = atlas["positions"][a]
            u, v = atlas["positions"][b]
            strata[(family, x == u, y == v)].append([a, b])
    rng = random.Random(seed)
    splits = {s: [] for s in SPLITS[:3]}
    for _, pairs in sorted(strata.items()):
        rng.shuffle(pairs)
        n = max(1, len(pairs) // 5)
        splits["validation"].extend(pairs[:n])
        splits["test"].extend(pairs[n:2 * n])
        splits["train"].extend(pairs[2 * n:])
    exposed = {tuple(p) for p in exposed_pairs}
    for split in ("validation", "test"):
        splits["train"].extend(p for p in splits[split] if tuple(p) in exposed)
        splits[split] = [p for p in splits[split] if tuple(p) not in exposed]
    return {s: sorted(v) for s, v in splits.items()}


def _balanced(pool: list[dict], count: int, key, rng: random.Random) -> list[dict]:
    buckets = defaultdict(list)
    for item in pool:
        buckets[key(item)].append(item)
    keys = sorted(buckets)
    _check(bool(keys), "empty query pool")
    for bucket in buckets.values():
        rng.shuffle(bucket)
    return [buckets[k][(i // len(keys)) % len(buckets[k])]
            for i in range(count) for k in [keys[i % len(keys)]]]


def static_queries(task: str, pairs: list[list[str]], count: int, rng: random.Random) -> list[dict]:
    atlas = atlas_geometry()
    if task in ("symbolic_direction", "symbolic_direction_choice"):
        pool = []
        for pair in pairs:
            # In a choice prompt the offered order already expresses operand order.
            # Reversing hidden a/b would duplicate an identical model presentation.
            for a, b in ((pair,) if task.endswith("choice") else (pair, pair[::-1])):
                for direction in DIRECTIONS:
                    query = dict(a=a, b=b, direction=direction)
                    if task.endswith("choice"):
                        for choices in (pair, pair[::-1]):
                            candidate = dict(query, choices=choices)
                            try:
                                symbolic_answer(task, {"state": None, "query": candidate})
                            except ValueError:  # equal on this axis: no two-choice question
                                continue
                            pool.append(candidate)
                    else:
                        pool.append(query)

        def group(q: dict) -> tuple:
            answer = symbolic_answer(task, {"state": None, "query": q})
            return (q["a"][1], q["choices"].index(answer), q["direction"]) if task.endswith("choice") else (q["a"][1], answer)

        return _balanced(pool, count, group, rng)
    if task == "symbolic_neighbors":
        pool = [{"token": t} for t in sorted(atlas["positions"])]
        return _balanced(pool, count, lambda q: q["token"][1], rng)
    if task == "symbolic_incidence":
        pool = [{"token": t, "family": f} for t in sorted(atlas["tokens"])
                for f in ({"N": "TEP", "T": "NE", "E": "NT", "P": "N"}[t[1]])]
        # Every edge and port is an active query at least once in training.
        rng.shuffle(pool)
        return [pool[i % len(pool)] for i in range(count)]
    pool = [{"token": t, "direction": d} for t in sorted(atlas["positions"])
            for d in (tuple(OFFSETS) if t[1] == "N" else
                      ("EAST", "WEST", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST"))]
    return _balanced(pool, count, lambda q: (q["token"][1], symbolic_answer(task, {"state": None, "query": q}) == "NONE"), rng)


def component_slots(task: str, count: int, rng: random.Random) -> list[dict]:
    """Factorial roster-position x mode x polarity slots, shuffled independently.

    Every binary mode/position cell has exactly as many positive as negative rows.
    Mode and position marginal sizes may differ by one pair when quotas do not
    divide the full factorial. Selection never chooses the queried color by polarity.
    """
    modes = {
        "symbolic_local_constraint": ("empty", "no_adjacent_building", "has_owned_incident_road"),
        "symbolic_owned_nodes": ("building", "settlement", "city"),
        "symbolic_piece_owner": ("N", "E"),
        "symbolic_near": ("tile", "resource", "port"),
        "symbolic_near_nodes": ("tile", "resource", "port"),
        "symbolic_scene_tiles": tuple(sorted(RESOURCES_LOWER)),
    }.get(task, ("all",))
    color_tasks = {"symbolic_local_constraint", "symbolic_owned_nodes", "symbolic_owned_roads",
                   "symbolic_owned_incident_roads", "symbolic_reachable", "symbolic_shortest_route"}
    positions = range(4) if task in color_tasks else (None,)
    slots = []
    if task in ("symbolic_reachable", "symbolic_shortest_route"):
        _check(count % 8 == 0, "route quota must be divisible by eight")
        for mode, n in (("zero", count // 8), ("no_route", count * 3 // 8), ("route", count // 2)):
            for i in range(n):
                slots.append(dict(mode=mode, position=i % 4, polarity=mode))
    elif task in ("symbolic_near_nodes", "symbolic_scene_tiles"):
        slots = [dict(mode=modes[i % len(modes)], position=None, polarity="nonempty") for i in range(count)]
    else:
        _check(count % 2 == 0, "binary component quotas must be even")
        cells = [(mode, position) for position in positions for mode in modes]
        for i in range(count // 2):
            mode, position = cells[i % len(cells)]
            for polarity in ("positive", "negative"):
                slots.append(dict(mode=mode, position=position, polarity=polarity))
    rng.shuffle(slots)
    return slots


def selector_nodes(selector: dict | None, data: dict, atlas: dict) -> set[str]:
    """Oracle-side selector projection; this node expansion never enters a prompt."""
    if selector is None:
        return set(atlas["graph"])
    if selector["kind"] in ("tile", "port"):
        return {n for n in atlas["touching"][selector["value"]] if n[1] == "N"}
    return {n for n, tiles in atlas["node_tiles"].items()
            if any(data["tiles"][tile][0] == selector["value"] for tile in tiles)}


def _selectors(kind: str, atlas: dict) -> list[dict]:
    values = (sorted(RESOURCES_LOWER) if kind == "resource" else
              sorted(t for t in atlas["tokens"] if t[1] == ("T" if kind == "tile" else "P")))
    return [dict(kind=kind, value=v) for v in values]


def _component_options(task: str, data: dict, slot: dict, atlas: dict) -> list[dict]:
    """Fast real-fact sampling predicates, independently checked by the final scorer."""
    mode, positive = slot["mode"], slot["polarity"] != "negative"
    position = slot["position"]
    color = data["colors"][position] if position is not None else None
    buildings, roads = data["buildings"], data["roads"]
    if task == "symbolic_near_nodes":
        return [{"near": s} for s in _selectors(mode, atlas)]
    if task == "symbolic_scene_tiles":
        return [{"resource": mode}]
    if task == "symbolic_owned_nodes":
        present = any(c == color and (mode == "building" or p == mode) for c, p in buildings.values())
        return [dict(color=color, piece=mode)] if present == positive else []
    if task == "symbolic_owned_roads":
        return [dict(color=color)] if (color in roads.values()) == positive else []
    if task == "symbolic_piece_owner":
        occupied = buildings if mode == "N" else roads
        return [{"token": t} for t in atlas["tokens"] if t[1] == mode and (t in occupied) == positive]
    if task == "symbolic_near":
        # All candidates are source supported. Selection below balances selector kind.
        result = []
        for selector in _selectors(mode, atlas):
            touching = selector_nodes(selector, data, atlas)
            result.extend(dict(node=n, near=selector) for n in atlas["graph"] if (n in touching) == positive)
        return result
    result = []
    for node, neighbors in atlas["graph"].items():
        incident = any(roads.get(e) == color for e in atlas["node_edges"][node])
        if task == "symbolic_owned_incident_roads":
            if incident == positive:
                result.append(dict(color=color, node=node))
        else:
            condition = {"empty": node not in buildings,
                         "no_adjacent_building": not neighbors & buildings.keys(),
                         "has_owned_incident_road": incident}[mode]
            if condition == positive:
                result.append(dict(color=color, node=node, predicate=mode))
    return result


def _route_query(state: dict, slot: dict, rng: random.Random, atlas: dict) -> dict:
    data = decode_state(state)
    color, mode = state["colors"][slot["position"]], slot["mode"]
    nodes = sorted(atlas["graph"])
    if mode == "zero":
        node = rng.choice(nodes)
        return dict(color=color, start=node, end=node)
    road_nodes = sorted({n for e, c in data["roads"].items() if c == color for n in atlas["edges"][e]})
    pool = list(combinations(road_nodes, 2))
    rng.shuffle(pool)
    if mode == "no_route":
        # Prefer disconnected endpoints within an owned graph when supported.
        rest = list(combinations(nodes, 2))
        rng.shuffle(rest)
        pool += rest
    candidates = []
    for start, end in pool:
        route = owned_route(state, color, start, end)
        if mode == "no_route" and route["nodes"] is None:
            candidates = [(start, end)]
            break
        if mode == "route" and route["nodes"] is not None:
            candidates.append((start, end))
            if len(route["edges"]) >= 2:
                candidates = [(start, end)]
                break
    _check(bool(candidates), "admitted route support did not yield a query")
    start, end = rng.choice(candidates)
    if rng.randrange(2):
        start, end = end, start
    return dict(color=color, start=start, end=end)


class ComponentSampler:
    """Shared state/map usage across all dynamic families; no per-family cursor reset."""

    def __init__(self, records: list[dict], rng: random.Random, atlas: dict):
        self.records = sorted(records, key=lambda r: r["provenance"]["state_id"])
        rng.shuffle(self.records)
        self.data = [decode_state(r["state"]) for r in self.records]
        self.rng, self.atlas = rng, atlas
        self.pools, self.cells = {}, {}
        self.state_uses, self.map_uses, self.density_uses = Counter(), Counter(), Counter()

    def choose(self, task: str, slot: dict) -> tuple[dict, dict]:
        key = (task, slot["mode"], slot["position"], slot["polarity"])
        if key not in self.pools:
            pool = []
            for i, data in enumerate(self.data):
                if task in ("symbolic_reachable", "symbolic_shortest_route"):
                    supported = (slot["mode"] != "route" or
                                 data["colors"][slot["position"]] in data["roads"].values())
                elif task in ("symbolic_near", "symbolic_near_nodes", "symbolic_scene_tiles"):
                    supported = True  # Every atlas selector has touching and non-touching nodes.
                else:
                    supported = bool(_component_options(task, data, slot, self.atlas))
                if supported:
                    pool.append(i)
            _check(bool(pool), f"unsupported component cell: {key}")
            counts = Counter(self.records[i]["provenance"]["density_bin"] for i in pool)
            self.cells[key] = dict(task_type=task, **slot,
                                   eligible_states_by_density={d: counts[d] for d in DENSITIES},
                                   selected_by_density=dict.fromkeys(DENSITIES, 0))
            # Occupied-board negatives are the first choice, not empty-board shortcuts.
            nonempty = [i for i in pool if self.records[i]["provenance"]["density_bin"] != "empty"]
            self.pools[key] = nonempty or pool
        pool = self.pools[key]

        def rank(i: int) -> tuple:
            p = self.records[i]["provenance"]
            return (self.state_uses[i], self.density_uses[key, p["density_bin"]],
                    self.map_uses[p["board_map_sha256"]], i)

        index = min(pool, key=rank)
        record, data = self.records[index], self.data[index]
        if task in ("symbolic_reachable", "symbolic_shortest_route"):
            query = _route_query(record["state"], slot, self.rng, self.atlas)
        else:
            query = self.rng.choice(_component_options(task, data, slot, self.atlas))
        p = record["provenance"]
        self.state_uses[index] += 1
        self.map_uses[p["board_map_sha256"]] += 1
        self.density_uses[key, p["density_bin"]] += 1
        self.cells[key]["selected_by_density"][p["density_bin"]] += 1
        return record, query

    def report(self) -> dict:
        return {"conditional_support": list(self.cells.values()),
                "unique_states": len(self.state_uses), "unique_maps": len(self.map_uses),
                "max_presentations_per_state": max(self.state_uses.values(), default=0),
                "policy": "Global least-used state, then conditional density and map usage; nonempty boards preferred where supported. Exact mode/roster/polarity slots. Empty conditional-support cells are genuine population limitations."}


def _ordered_sources(records: list[dict], rng: random.Random) -> list[dict]:
    buckets = defaultdict(list)
    for record in records:
        p = record["provenance"]
        buckets[(p["source"]["kind"], p["density_bin"])].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: r["provenance"]["state_id"])
        rng.shuffle(bucket)
    result = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                result.append(buckets[key].pop())
    return result


def _row(task: str, query: dict, record: dict | None, split: str, index: int) -> dict:
    target = {"state": record["state"] if record else None, "query": query}
    rid = f"{VERSION}/{split}/{task}/{index:05d}"
    role = symbolic_task_role(task, split)
    metadata = {"task_type": task, "training_family": task, "task_role": role,
                "target": target, "split": split,
                "query_sha256": canonical_sha256(query)}
    if record:
        metadata["provenance"] = record["provenance"]
    if task in ("symbolic_direction", "symbolic_direction_choice"):
        metadata["directional_pair"] = sorted([query["a"], query["b"]])
    return {"schema": "catan_symbolic_board_row/v2", "id": rid, "row_id": rid,
            "task_type": task, "training_family": task, "task_role": role, "split": split,
            "messages": [{"role": "user", "content": symbolic_prompt(task, target)},
                         {"role": "assistant", "content": symbolic_answer(task, target)}],
            "metadata": metadata}


def component_rows(records: list[dict], split: str, pairs: list, quotas: dict, seed: int) -> tuple[list[dict], dict]:
    rng, atlas = random.Random(seed), atlas_geometry()
    sampler = ComponentSampler(records, rng, atlas)
    families = {}
    for task, count in quotas.items():
        family = []
        if task in STATIC_TASKS:
            family = [_row(task, query, None, split, i) for i, query in enumerate(static_queries(task, pairs, count, rng))]
        else:
            for i, slot in enumerate(component_slots(task, count, rng)):
                record, query = sampler.choose(task, slot)
                row = _row(task, query, record, split, i)
                row["metadata"]["sampling_cell"] = slot
                family.append(row)
        families[task] = family
    # Deterministic family round-robin presentations, not a claimed optimizer schedule.
    result = []
    for i in range(max(quotas.values())):
        for task in quotas:
            if i < len(families[task]):
                result.append(families[task][i])
    return result, sampler.report()


def transfer_projection(task: str, state: dict, query: dict, *, atlas: dict | None = None,
                        data: dict | None = None) -> str:
    """Hash relevant inputs, never the answer. Ignore robber, irrelevant terrain and pieces.

    Setup ignores queried color, ownership and roads. Normal keeps queried incident
    roads. Near selector kinds remain distinct reasoning modes; within each mode,
    equivalent candidate-node filters and local occupancy deduplicate. Road tasks
    retain participant identities, owned edges and effective transit blockers only.
    """
    atlas = atlas if atlas is not None else atlas_geometry()
    data = data if data is not None else decode_state(state)
    if task == "symbolic_settlement_locations":
        candidates = selector_nodes(query["near"], data, atlas)
        relevant_nodes = candidates | {n for v in candidates for n in atlas["graph"][v]}
        projection = {
            "task": task, "phase": query["phase"],
            "near_kind": query["near"]["kind"] if query["near"] else "all",
            "candidates": sorted(candidates),
            "occupied": sorted(relevant_nodes & data["buildings"].keys()),
        }
        if query["phase"] == "normal":
            projection["owned_incident_edges"] = sorted(
                e for e, c in data["roads"].items() if c == query["color"]
                and candidates.intersection(atlas["edges"][e]))
    else:
        blockers = {}
        for color in state["colors"]:
            degree = Counter(n for e, c in data["roads"].items() if c == color for n in atlas["edges"][e])
            blockers[color] = sorted(n for n, (owner, _) in data["buildings"].items()
                                     if owner != color and degree[n] >= 2)
        projection = {"task": task, "colors": sorted(state["colors"]),
                      "roads": data["roads"], "blockers": blockers}
    return canonical_sha256(projection)


def graph_case_coverage(records: list[dict]) -> dict:
    """Measured graph cases on real, unmodified sources, including absent hard cases."""
    atlas, cases = atlas_geometry(), []
    for record in records:
        state, p = record["state"], record["provenance"]
        data = decode_state(state)
        lengths = strict_json(symbolic_answer("symbolic_longest_lengths", {"state": state, "query": {}}))
        maximum = max(lengths.values())
        flags = dict(cycle=False, effective_blocker=False, branch=False,
                     tied_max_ge5=maximum >= 5 and sum(v == maximum for v in lengths.values()) > 1,
                     nonzero=maximum > 0, award_eligible=maximum >= 5)
        for color in state["colors"]:
            graph = nx.Graph()
            graph.add_edges_from(atlas["edges"][e] for e, c in data["roads"].items() if c == color)
            flags["cycle"] |= bool(nx.cycle_basis(graph))
            flags["branch"] |= any(d >= 3 for _, d in graph.degree)
            flags["effective_blocker"] |= any(
                owner != color and n in graph and graph.degree[n] >= 2
                for n, (owner, _) in data["buildings"].items())
        cases.append({"state_id": p["state_id"], "board_fact_sha256": p["board_fact_sha256"], **flags})
    counts = {key: sum(c[key] for c in cases) for key in
              ("cycle", "effective_blocker", "branch", "tied_max_ge5", "nonzero", "award_eligible")}
    missing = [key for key in ("cycle", "effective_blocker", "tied_max_ge5") if not counts[key]]
    return {"states": len(records), "counts": counts, "cases": cases,
            "missing_coverage": missing,
            "status": "transfer_pilot_limited" if missing else "observed_cases_only_not_exhaustive",
            "definitions": {"cycle": "cycle in one player's owned undirected graph",
                            "effective_blocker": "opponent building with at least two incident roads of a player; blocks interior transit",
                            "tied_max_ge5": "at least two colors tie for maximum engine trail length >=5"}}


def _diverse_candidates(pool: list[dict], count: int, rng: random.Random) -> list[dict]:
    candidates = list(pool)
    rng.shuffle(candidates)
    uses, density = Counter(), Counter()
    selected = []
    while candidates and len(selected) < count:
        index = min(range(len(candidates)), key=lambda i: (
            uses[candidates[i]["record"]["provenance"]["state_id"]],
            density[candidates[i]["record"]["provenance"]["density_bin"]], i))
        candidate = candidates.pop(index)
        selected.append(candidate)
        p = candidate["record"]["provenance"]
        uses[p["state_id"]] += 1
        density[p["density_bin"]] += 1
    return selected


def transfer_weights(rows: list[dict]) -> list[dict]:
    """Per-family mode/polarity macro weights with equal state weight within each cell."""
    states, labels, modes = defaultdict(Counter), defaultdict(set), defaultdict(set)
    for row in rows:
        m = row["metadata"]
        family, mode, label = row["task_type"], m["evaluation_mode"], m["polarity"]
        states[family, mode, label][m["provenance"]["state_id"]] += 1
        labels[family, mode].add(label)
        modes[family].add(mode)
    result = []
    for row in rows:
        m = row["metadata"]
        family, mode, label = row["task_type"], m["evaluation_mode"], m["polarity"]
        counts = states[family, mode, label]
        state_weight = 1 / (len(counts) * counts[m["provenance"]["state_id"]])
        result.append({"state_weight": state_weight,
                       "macro_weight": state_weight / (len(labels[family, mode]) * len(modes[family]))})
    return result


def transfer_rows(records: list[dict], split: str, seed: int) -> tuple[list[dict], dict]:
    """Enumerate real candidates; retain all unique normal positives and positive road cases.

    Setup is a bounded matched control (<=16 positives per selector kind), rather
    than an enormous color/terrain-expanded panel. Negative subsets never exceed
    positive support; a missing positive cell stays explicitly empty, never repaired.
    """
    rng, atlas = random.Random(seed), atlas_geometry()
    unique, population, skipped = {}, Counter(), []

    def admit(task: str, query: dict, record: dict, data: dict, mode: str, positive: bool) -> None:
        label = "positive" if positive else "negative"
        population[task, mode, label] += 1
        projection = transfer_projection(task, record["state"], query, atlas=atlas, data=data)
        if projection not in unique:
            unique[projection] = dict(task=task, query=query, record=record, mode=mode,
                                      polarity=label, projection=projection, equivalent_sources=[])
        candidate = unique[projection]
        _check(candidate["polarity"] == label, "task-relevant projection collapsed different labels")
        sid = record["provenance"]["state_id"]
        if sid not in candidate["equivalent_sources"]:
            candidate["equivalent_sources"].append(sid)

    selectors = [None] + [s for kind in ("tile", "resource", "port") for s in _selectors(kind, atlas)]
    for record in _ordered_sources(records, rng):
        state = record["state"]
        data = decode_state(state)
        for color in state["colors"]:
            for phase in ("setup", "normal"):
                base = symbolic_answer("symbolic_settlement_locations", {
                    "state": state, "query": dict(color=color, phase=phase, near=None)})
                legal = set(base.split()) - {"NONE"}
                for near in selectors:
                    mode = phase + "/" + (near["kind"] if near else "all")
                    admit("symbolic_settlement_locations", dict(color=color, phase=phase, near=near),
                          record, data, mode, bool(legal & selector_nodes(near, data, atlas)))
        lengths = strict_json(symbolic_answer("symbolic_longest_lengths", {"state": state, "query": {}}))
        maximum = max(lengths.values())
        for task in ("symbolic_longest_lengths", "symbolic_longest_leaders", "symbolic_longest_award"):
            if task == "symbolic_longest_award" and maximum >= 5 and sum(v == maximum for v in lengths.values()) > 1:
                skipped.append({"state_id": record["provenance"]["state_id"], "task_type": task,
                                "reason": "unknown_incumbent_tied_maximum"})
                continue
            admit(task, {}, record, data, "all", maximum >= (5 if task == "symbolic_longest_award" else 1))
    buckets = defaultdict(list)
    for candidate in unique.values():
        buckets[candidate["task"], candidate["mode"], candidate["polarity"]].append(candidate)
    selected, group_report = [], []
    groups = sorted({key[:2] for key in population})
    for task, mode in groups:
        pos, neg = buckets[task, mode, "positive"], buckets[task, mode, "negative"]
        budget = len(pos)
        if mode.startswith("setup/"):
            normal_support = len(buckets[task, mode.replace("setup/", "normal/"), "positive"])
            budget = min(budget, normal_support, 16)
        chosen_pos = _diverse_candidates(pos, budget, rng)
        chosen_neg = _diverse_candidates(neg, len(chosen_pos), rng)
        selected.extend(chosen_pos + chosen_neg)
        group_report.append({"task_type": task, "mode": mode,
                             "positive": {"population": population[task, mode, "positive"],
                                          "unique": len(pos), "selected": len(chosen_pos)},
                             "negative": {"population": population[task, mode, "negative"],
                                          "unique": len(neg), "selected": len(chosen_neg)}})
    rng.shuffle(selected)
    result = []
    for candidate in selected:
        row = _row(candidate["task"], candidate["query"], candidate["record"], split, len(result))
        row["metadata"].update(evaluation_mode=candidate["mode"], polarity=candidate["polarity"],
                               task_projection_sha256=candidate["projection"],
                               equivalent_source_state_ids=sorted(candidate["equivalent_sources"]))
        result.append(row)
    for row, weights in zip(result, transfer_weights(result), strict=True):
        row["metadata"].update(weights)
    selected_records = {r["metadata"]["provenance"]["state_id"]: {
        "state": r["metadata"]["target"]["state"], "provenance": r["metadata"]["provenance"]} for r in result}
    report = {
        "status": "transfer_pilot_limited", "population_rows": sum(population.values()),
        "unique_relevant_queries": len(unique), "selected_rows": len(result), "groups": group_report,
        "empty_groups": [{"task_type": g["task_type"], "mode": g["mode"], "polarity": label,
                          "reason": "no_population_support" if not g[label]["unique"] else "matched_selection_has_no_positive_support"}
                         for g in group_report for label in ("positive", "negative") if not g[label]["selected"]],
        "skipped_ambiguous_awards": skipped,
        "population_graph_cases": graph_case_coverage(records),
        "selected_graph_cases": graph_case_coverage(list(selected_records.values())),
        "selection_policy": "All task-relevant unique normal-placement positives and positive road cases retained; equal-or-smaller negative subsets. Setup controls matched to normal positive support with <=16 positives per near kind. No source split changes or fabricated states.",
        "aggregation": {"pooled_micro_headline": False, "report_families_and_modes_separately": True,
                        "weights": "macro_weight sums to one within each task family, balancing supported modes/polarities and source states; state_weight sums to one within each mode/polarity cell",
                        "inference_unit": "whole source state/game; rows are correlated query projections",
                        "missing_groups": "report as unsupported, not zero accuracy or full coverage"},
    }
    return result, report


def validate_row_declarations(row: dict, split: str) -> None:
    """Every task/family/split/role declaration must agree; redundant tags are not authority."""
    _check(set(row) == {"schema", "id", "row_id", "task_type", "training_family", "task_role",
                        "split", "messages", "metadata"}, "unexpected model record keys")
    _check(row["schema"] == "catan_symbolic_board_row/v2", "wrong row schema")
    task, metadata = row["task_type"], row["metadata"]
    _check(isinstance(task, str) and isinstance(metadata, dict), "invalid task declarations")
    role = symbolic_task_role(task, split)
    for key, value in (("task_type", task), ("training_family", task), ("split", split), ("task_role", role)):
        _check(row.get(key) == metadata.get(key) == value, f"task declaration mismatch: {key}")


def component_profile(rows: list[dict]) -> dict:
    """Record model-visible queried roster positions alongside conditional source diversity."""
    cells, families = defaultdict(list), defaultdict(list)
    for row in rows:
        m, task = row["metadata"], row["task_type"]
        if task in STATIC_TASKS:
            continue
        prompt, response = row["messages"][0]["content"], row["messages"][1]["content"]
        roster = re.search(r"Participants: ([A-Z_ ]+)\.\nBoard:", prompt).group(1).split()
        question = prompt.rsplit("\n", 1)[-1]
        color_match = re.search(r"(?:List existing |List all existing |List nodes with a |For |existing )([A-Z_]+)(?: road| incident road| building| settlement| city|,)", question)
        position = roster.index(color_match.group(1)) if color_match else None
        mode = m["sampling_cell"]["mode"]
        if task == "symbolic_local_constraint":
            mode = ("has_owned_incident_road" if "incident road" in question else
                    "no_adjacent_building" if "edge-adjacent" in question else "empty")
        if task in ("symbolic_reachable", "symbolic_shortest_route"):
            polarity = m["sampling_cell"]["polarity"]
        else:
            polarity = "negative" if response in ("NONE", "no") else "positive"
        p = m["provenance"]
        cells[task, mode, position, polarity, p["density_bin"]].append(p)
        families[task].append(p)
    return {
        "rendered_roster_crosstabs": [dict(task_type=t, mode=mode, roster_position=position,
                                           polarity=polarity, density=density, rows=len(ps),
                                           unique_states=len({p["state_id"] for p in ps}))
                                      for (t, mode, position, polarity, density), ps in cells.items()],
        "families": {t: {"rows": len(ps), "unique_states": len({p["state_id"] for p in ps}),
                          "unique_maps": len({p["board_map_sha256"] for p in ps}),
                          "by_density": dict(Counter(p["density_bin"] for p in ps)),
                          "by_source_kind": dict(Counter(p["source"]["kind"] for p in ps))}
                     for t, ps in families.items()},
    }


def validate_component_balance(rows: list[dict]) -> None:
    """Production gate on rendered roster/predicate/label cross-tabs, not cached labels."""
    tables = defaultdict(Counter)
    for cell in component_profile(rows)["rendered_roster_crosstabs"]:
        if cell["task_type"] in ("symbolic_local_constraint", "symbolic_owned_incident_roads",
                                  "symbolic_owned_nodes", "symbolic_owned_roads"):
            key = (cell["task_type"], cell["mode"], cell["roster_position"])
            tables[key][cell["polarity"]] += cell["rows"]
    for key, counts in tables.items():
        _check(counts["positive"] == counts["negative"] > 0, f"rendered roster label shortcut: {key}")


def validate_rows(files: dict[str, list[dict]], pairs: dict, sources: dict | None = None,
                  exposure: dict | None = None) -> dict:
    """Re-render prompts, recompute all answers, check split/pair/source boundaries."""
    ids, pair_sets, coverage, query_coverage = set(), {}, {}, {}
    provenance_groups = {key: {} for key in ("trajectory_id", "board_map_sha256")}
    for split, values in pairs.items():
        pair_sets[split] = {tuple(v) for v in values}
        _check(len(pair_sets[split]) == len(values), "duplicate directional pair")
    for a, b in combinations(pair_sets, 2):
        _check(not pair_sets[a] & pair_sets[b], "directional pair leakage")
    for split, rows in files.items():
        base_split = split.removeprefix("transfer_")
        tasks = TRANSFER_TASKS if split.startswith("transfer_") else TRAIN_TASKS
        tokens, queried, projections = set(), set(), set()
        source_index = ({r["provenance"]["state_id"]: r for r in sources[base_split]} if sources else None)
        for position, row in enumerate(rows):
            validate_row_declarations(row, split)
            _check(row["row_id"] == row["id"] and row["id"] not in ids, "duplicate row id")
            ids.add(row["id"])
            task, metadata = row["task_type"], row["metadata"]
            _check(task in tasks and task == metadata["task_type"], "task split leakage")
            _check(metadata["split"] == split and metadata["row_position"] == position, "row ordering/split error")
            messages = row["messages"]
            _check(len(messages) == 2 and [m["role"] for m in messages] == ["user", "assistant"]
                   and all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in messages),
                   "rows require user/answer text only")
            target = metadata["target"]
            _check(messages[0]["content"] == symbolic_prompt(task, target), "prompt/target mismatch or leaked input")
            score = score_symbolic_task("deliberately untrusted cache", messages[1]["content"], metadata)
            _check(score is not None and score["correct"], "answer failed recomputing scorer")
            _check(metadata["query_sha256"] == canonical_sha256(target["query"]), "query digest mismatch")
            if task in ("symbolic_direction", "symbolic_direction_choice"):
                q = target["query"]
                pair = tuple(sorted((q["a"], q["b"])))
                _check(pair in pair_sets[base_split] and metadata["directional_pair"] == list(pair), "wrong direction partition")
                if exposure is not None:
                    known = exposure["pair_sources"].get(" ".join(pair), [])
                    _check(metadata.get("known_training_exposure") == known, "historical exposure annotation mismatch")
                    _check(base_split == "train" or not known, "known training pair entered heldout")
            if task not in STATIC_TASKS:
                p = metadata["provenance"]
                _check(p["split"] == base_split, "source split leakage")
                for key in provenance_groups:
                    value = p["source"][key] if key == "trajectory_id" else p[key]
                    previous = provenance_groups[key].setdefault(value, base_split)
                    _check(previous == base_split, f"dynamic {key} leakage")
                if source_index is not None:
                    original = source_index[p["state_id"]]
                    _check(original["state"] == target["state"] and original["provenance"] == p,
                           "model state/provenance differs from real source")
                if task in TRAIN_TASKS:
                    slot = metadata["sampling_cell"]
                    q = target["query"]
                    if "color" in q:
                        _check(type(slot["position"]) is int and 0 <= slot["position"] < 4 and
                               q["color"] == target["state"]["colors"][slot["position"]], "sampling roster position mismatch")
                    if slot["polarity"] in ("positive", "negative"):
                        positive = score["expected_normalized"] not in ("no", "NONE")
                        _check(positive == (slot["polarity"] == "positive"), "sampling polarity mismatch")
            if task in TRANSFER_TASKS:
                projection = transfer_projection(task, target["state"], target["query"])
                _check(projection == metadata.get("task_projection_sha256") and projection not in projections,
                       "duplicate or incorrect task-relevant transfer projection")
                projections.add(projection)
                q = target["query"]
                mode = (q["phase"] + "/" + (q["near"]["kind"] if q["near"] else "all")
                        if task == "symbolic_settlement_locations" else "all")
                if task in ("symbolic_longest_leaders", "symbolic_longest_lengths"):
                    lengths = strict_json(symbolic_answer("symbolic_longest_lengths", {"state": target["state"], "query": {}}))
                    positive = max(lengths.values()) > 0
                else:
                    positive = score["expected_normalized"] != "NONE"
                _check(metadata["evaluation_mode"] == mode and
                       metadata["polarity"] == ("positive" if positive else "negative"), "transfer evaluation group mismatch")
            tokens.update(TOKEN_PATTERN.findall(messages[0]["content"] + " " + messages[1]["content"]))
            queried.update(TOKEN_PATTERN.findall(json.dumps(target["query"])))
        coverage[split], query_coverage[split] = sorted(tokens), sorted(queried)
        if split.startswith("transfer_"):
            for row, weights in zip(rows, transfer_weights(rows), strict=True):
                _check(all(row["metadata"].get(k) == v for k, v in weights.items()), "invalid transfer weights")
        else:
            validate_component_balance(rows)
    _check(set(coverage["train"]) == set(atlas_tokens()), "missing or noncanonical training tokens")
    _check(set(query_coverage["train"]) == set(atlas_tokens()), "all 154 tokens must be active training queries")
    return {"tokens": coverage, "active_query_tokens": query_coverage}


def preserved_v1_artifacts() -> dict:
    """Verify historical output bytes only; historical code hashes intentionally stay historical."""
    root = DEFAULT_OUTPUT.with_name("symbolic_board_v1")
    if not root.exists():
        return {"status": "not_present"}
    manifest = read_json(root / "manifest.json")
    for info in manifest["files"].values():
        _check(file_sha256(Path(info["path"])) == info["sha256"], "historical v1 artifact changed")
    return {"status": "preserved", "manifest": _hash(root / "manifest.json"), "files": manifest["files"]}


def _write(path: Path, value: object, *, jsonl: bool = False) -> None:
    # Exclusive file creation; a partially failed build is never silently overwritten.
    with path.open("x") as handle:
        if jsonl:
            for row in value:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        else:
            handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def build_dataset(output_dir: Path = DEFAULT_OUTPUT, *, root: Path = DEFAULT_ROOT,
                  seed: int = 46, dry_run: bool = False, checkpoint: str | None = None) -> dict:
    """Construct/verify a 3,200-presentation pilot; optionally persist immutable files."""
    output, root = output_dir.resolve(), root.resolve()
    if output.exists() and not dry_run:
        raise FileExistsError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {output.parent}")
    preserved = preserved_v1_artifacts()
    sources, audit = load_sources(root)
    exposure = known_direction_exposure()
    exposed_pairs = [key.split() for key in exposure["pair_sources"]]
    pairs = directional_pair_manifest(seed, exposed_pairs)
    files, sampling = {}, {}
    for i, split in enumerate(SPLITS[:3]):
        files[split], sampling[split] = component_rows(
            sources[split], split, pairs[split], TRAIN_QUOTAS if split == "train" else EVAL_QUOTAS, seed + i)
    transfer_reports = {}
    for i, split in enumerate(("validation", "test")):
        name = "transfer_" + split
        files[name], transfer_reports[name] = transfer_rows(sources[split], name, seed + i + 10)
    for rows in files.values():
        for index, row in enumerate(rows):
            row["metadata"]["row_position"] = index
            if "directional_pair" in row["metadata"]:
                key = " ".join(row["metadata"]["directional_pair"])
                row["metadata"]["known_training_exposure"] = exposure["pair_sources"].get(key, [])
    coverage = validate_rows(files, pairs, sources, exposure)
    selected = {r["metadata"]["provenance"]["state_id"]: r["metadata"]["provenance"]
                for rows in files.values() for r in rows if "provenance" in r["metadata"]}
    metadata = {
        "schema": "catan_symbolic_board_dataset/v2", "seed": seed, "source_audit": audit,
        "preserved_v1": preserved,
        "train_quotas": TRAIN_QUOTAS, "component_eval_quotas": EVAL_QUOTAS,
        "static_training_presentations": sum(TRAIN_QUOTAS[t] for t in STATIC_TASKS),
        "state_training_presentations": sum(TRAIN_QUOTAS[t] for t in TRAIN_TASKS - STATIC_TASKS),
        "row_order": "family round-robin, insertion-order quotas; explicit row_position; presentation counts, not optimizer steps",
        "row_ids": {s: [r["id"] for r in rows] for s, rows in files.items()},
        "counts": {s: len(rows) for s, rows in files.items()},
        "task_counts": {s: dict(Counter(r["task_type"] for r in rows)) for s, rows in files.items()},
        "unique_prompt_counts": {s: len({r["messages"][0]["content"] for r in rows})
                                 for s, rows in files.items()},
        "repetition_policy": "Fixed-atlas neighbor/incidence/oriented-step questions may repeat to meet presentation quotas; report unique prompts separately.",
        "source_kind_counts": {s: dict(Counter(r["metadata"]["provenance"]["source"]["kind"]
                                               for r in rows if "provenance" in r["metadata"])) for s, rows in files.items()},
        "answer_counts": {s: {t: dict(Counter(r["messages"][1]["content"] for r in rows
                                              if r["task_type"] == t)) for t in ("symbolic_direction", "symbolic_reachable", "symbolic_near", "symbolic_local_constraint")}
                          for s, rows in files.items() if not s.startswith("transfer_")},
        "coverage": coverage, "selected_sources": [selected[s] for s in sorted(selected)],
        "component_sampling": sampling,
        "component_profiles": {s: component_profile(files[s]) for s in SPLITS[:3]},
        "transfer_selection": transfer_reports,
        "reserved_color_diagnostic_coverage": {
            "included_in_transfer": False, "graph_cases": graph_case_coverage(sources["color_diagnostic"]),
            "provenance": [r["provenance"] for r in sources["color_diagnostic"]],
        },
        "historical_exposure_audit": exposure,
        "transfer_semantics": "Compositional transfer: empty, adjacent-building and owned-incident-road atomic predicates overlap settlement components; their complete setup/normal conjunction is held out. Owned shortest routes overlap graph traversal but never longest edge-simple lengths, leaders or awards. Dynamic source states/maps/trajectories stay split-disjoint.",
        "training_budget": "dataset preparation only; no final SFT budget approved",
    }
    inputs = {
        "schema": "catan_symbolic_board_inputs/v2", "output_dir": str(output),
        "train_jsonl": str(output / "train.jsonl"),
        "validation_jsonl": str(output / "validation.jsonl"), "test_jsonl": str(output / "test.jsonl"),
        "transfer_validation_jsonl": str(output / "transfer_validation.jsonl"),
        "transfer_test_jsonl": str(output / "transfer_test.jsonl"),
        "token_inventory": str(output / "token_inventory.json"), "metadata": str(output / "metadata.json"),
        "manifest": str(output / "manifest.json"), "source_root": str(root),
        "source_hashes": audit["source_hashes"],
        "proposed_paired_experiment_inputs": {
            "initial_bundle": checkpoint, "reuse_existing_atlas_checkpoint": True, "token_init": "keep",
            "modality": "text_only", "conditions": ["before_sft", "after_component_sft"],
            "same_evaluation_files": True, "final_sft_budget_approved": False,
            "checkpoint_hash_and_token_id_audit": "required in integration; weights not loaded here",
            "max_new_tokens_proposal": 1024,
            "transfer_status": "transfer_pilot_limited",
            "evaluation_aggregation": "Report each family/mode with stored per-state macro weights; no pooled micro headline. Missing graph cases remain unsupported.",
        },
    }
    if dry_run:
        return {"dry_run": True, "counts": metadata["counts"], "source_admitted": audit["admitted_counts"],
                "source_exclusions": len(audit["exclusions"]), "inputs": inputs,
                "component_unique_states": {s: sampling[s]["unique_states"] for s in sampling},
                "transfer_selection": transfer_reports}
    # Parent existence was checked before any directory creation; no parents=True.
    output.mkdir(exist_ok=False)
    assets = {}
    for name, rows in files.items():
        path = output / f"{name}.jsonl"
        _write(path, rows, jsonl=True)
        assets[name] = {**_hash(path), "rows": len(rows)}
    inventory = {"schema": "catan_symbolic_token_inventory/v1", "tokens": atlas_tokens(),
                 "atlas_tokens": atlas_tokens(), "token_type": "regular_added_tokens",
                 "counts": {"atlas": 154, "total": 154}, "reuse_existing_token_ids": True}
    _write(output / "token_inventory.json", inventory)
    _write(output / "directional_pairs.json", {"seed": seed, "splits": pairs,
                                               "known_training_pairs": exposed_pairs,
                                               "group_unit": "unordered_same_family_pair_across_axes_operands_inverses_choices"})
    metadata["files"] = assets
    _write(output / "metadata.json", metadata)
    _write(output / "dataset_inputs.json", inputs)
    for name in ("token_inventory", "directional_pairs", "metadata", "dataset_inputs"):
        assets[name] = _hash(output / f"{name}.json")
    _write(output / "manifest.json", {"schema": "catan_symbolic_board_manifest/v2", "files": assets,
                                       "counts": metadata["counts"], "source_hashes": audit["source_hashes"]})
    _check(preserved_v1_artifacts() == preserved, "v1 artifacts changed during build")
    return {"counts": metadata["counts"], "source_admitted": audit["admitted_counts"],
            "source_exclusions": len(audit["exclusions"]), "inputs": inputs,
            "component_unique_states": {s: sampling[s]["unique_states"] for s in sampling},
            "transfer_graph_coverage": {s: r["population_graph_cases"]["counts"] for s, r in transfer_reports.items()}}


def validate_dataset(output_dir: Path = DEFAULT_OUTPUT) -> dict:
    """Validate bytes, current pinned sources, real-state joins, prompts and all labels."""
    output = output_dir.resolve()
    manifest = read_json(output / "manifest.json")
    _check(manifest["schema"] == "catan_symbolic_board_manifest/v2", "this validator requires v2 output")
    for info in manifest["files"].values():
        path = Path(info["path"])
        _check(path.parent == output and path.is_file(), "manifest output path mismatch")
        _check(file_sha256(path) == info["sha256"], f"generated file changed: {path}")
    metadata = read_json(output / "metadata.json")
    _check(metadata["preserved_v1"] == preserved_v1_artifacts(), "historical v1 receipt changed")
    for info in manifest["source_hashes"].values():
        _check(file_sha256(Path(info["path"])) == info["sha256"], f"pinned source changed: {info['path']}")
    sources, audit = load_sources(Path(metadata["source_audit"]["source_root"]))
    _check(audit == metadata["source_audit"], "source admission audit changed")
    files = {name: read_jsonl(Path(info["path"])) for name, info in manifest["files"].items() if "rows" in info}
    _check({s: len(rows) for s, rows in files.items()} == manifest["counts"] == metadata["counts"], "output count mismatch")
    pairs = read_json(output / "directional_pairs.json")
    exposure = known_direction_exposure()
    _check(exposure == metadata["historical_exposure_audit"], "known historical exposure changed")
    exposed_pairs = [key.split() for key in exposure["pair_sources"]]
    _check(pairs["known_training_pairs"] == exposed_pairs and
           pairs["splits"] == directional_pair_manifest(metadata["seed"], exposed_pairs), "pair assignment changed")
    _check(validate_rows(files, pairs["splits"], sources, exposure) == metadata["coverage"], "coverage changed")
    for split, rows in files.items():
        _check([r["id"] for r in rows] == metadata["row_ids"][split], "row order changed")
        if split in SPLITS[:3]:
            _check(dict(Counter(r["task_type"] for r in rows)) == (TRAIN_QUOTAS if split == "train" else EVAL_QUOTAS),
                   "family quota drift")
            _check(component_profile(rows) == metadata["component_profiles"][split], "conditional coverage changed")
    for i, split in enumerate(("validation", "test")):
        name = "transfer_" + split
        regenerated, report = transfer_rows(sources[split], name, metadata["seed"] + i + 10)
        for index, row in enumerate(regenerated):
            row["metadata"]["row_position"] = index
        _check(regenerated == files[name] and report == metadata["transfer_selection"][name],
               "transfer population/selection/weights changed")
    _check(graph_case_coverage(sources["color_diagnostic"]) ==
           metadata["reserved_color_diagnostic_coverage"]["graph_cases"], "reserved diagnostics changed")
    return {"valid": True, "counts": metadata["counts"], "source_exclusions": len(audit["exclusions"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--checkpoint", help="Proposed existing atlas bundle path; not loaded")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    result = (validate_dataset(args.output_dir) if args.validate else
              build_dataset(args.output_dir, root=args.root, seed=args.seed,
                            dry_run=args.dry_run, checkpoint=args.checkpoint))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
