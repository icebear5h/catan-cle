from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

from data_pipeline.board_recognition.full_board_readout import board_answer
from data_pipeline.board_recognition.replay_dataset import dense_labels, static_board_facts
from data_pipeline.board_recognition.sources import (
    DEFAULT_LEAKAGE_LEDGER,
    DEFAULT_SOURCE_LOCK,
    canonical_sha256,
    file_sha256,
    load_leakage_ledger,
    source_lock_matches_metadata,
    validate_replay_source_lock,
    visible_board_facts,
)
from evals.catan_board_bench.tokens import atlas_tokens
from sft.board.symbolic_board_tasks import PhysicalStateError, validate_contract
from sft.board.symbolic_board_tasks._types import StatePayload
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_list, as_str

from ._io import _asset as _asset
from ._io import _check as _check
from ._io import _hash as _hash
from ._io import _unique_pairs as _unique_pairs
from ._io import read_json as read_json
from ._io import read_jsonl as read_jsonl
from ._types import SourceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[4]
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


def _engine_source_digest(source: JsonDict, colors: list[str]) -> str:
    return canonical_sha256({
        "schema": "catan_board_recognition_engine_policy/v2",
        "game_seed": source["engine_seed"], "policy_seed": source["policy_seed"],
        "colors": colors, "legal_actions_only": True, "victory_points_to_win": 13,
        "action_weights": {"BUILD_CITY": 20, "BUILD_SETTLEMENT": 8, "BUILD_ROAD": 5,
                           "BUY_DEVELOPMENT_CARD": 2, "MARITIME_TRADE": 2, "END_TURN": 1},
        "forced_resources": False, "forced_board_mutation": False,
        "trajectory_action_limit": source["trajectory_action_limit"],
    })


def audit_source_state(root: Path, source: JsonDict,
                       readout: JsonDict) -> tuple[StatePayload, JsonDict]:
    """Pin and join contract/dense labels/readout without opening any image asset.

    PhysicalStateError is an explicit exclusion; broken hashes, identity or joins
    are fatal. Returned provenance contains no image keys or model-visible extras.
    """
    sid = as_str(source["sample_id"])
    paths = {"contract": _asset(root, as_str(source["contract_path"])),
             "labels": _asset(root, as_str(source["label_path"]))}
    hashes = {kind: file_sha256(path) for kind, path in paths.items()}
    declared = as_dict(source["sha256"])
    _check(all(h == declared[kind] for kind, h in hashes.items()), f"source hash mismatch: {sid}")
    contract = read_json(paths["contract"])
    _check(as_dict(contract["sample"])["id"] == sid and contract["source"] == source["source"],
           f"source contract identity mismatch: {sid}")
    _check(readout["state_id"] == sid and readout["split"] == source["split"]
           and readout["task_type"] == "full_board_readout", f"readout join mismatch: {sid}")
    _check(readout["images"] == [Path(as_str(source["image_path"])).name],
           f"readout asset identity mismatch: {sid}")
    _check(canonical_sha256(static_board_facts(contract)) == source["board_map_sha256"],
           f"map hash mismatch: {sid}")
    _check(canonical_sha256(visible_board_facts(contract)) == source["board_fact_sha256"],
           f"board fact hash mismatch: {sid}")
    _check(dense_labels(contract, sample_id=sid) == read_json(paths["labels"]), f"dense label mismatch: {sid}")
    _check(board_answer(contract) == as_dict(as_list(readout["messages"])[-1])["content"],
           f"readout answer mismatch: {sid}")
    colors = [as_str(as_dict(p)["color"]) for p in as_list(contract["players"])]
    info = as_dict(source["source"])
    if info["kind"] == "engine_rollout":
        _check(_engine_source_digest(info, colors) == info["source_sha256"],
               f"engine policy hash mismatch: {sid}")
    state = validate_contract(contract)
    provenance: JsonDict = {
        "state_id": sid, "split": source["split"], "layout_id": readout["layout_id"],
        "density_bin": source["density_bin"], "board_map_sha256": source["board_map_sha256"],
        "board_fact_sha256": source["board_fact_sha256"], "source": source["source"],
        "source_readout_row_id": readout["row_id"],
        "paths": cast("JsonDict", {k: str(v) for k, v in paths.items()}),
        "sha256": cast("JsonDict", hashes),
    }
    return state, provenance


def load_sources(root: Path = DEFAULT_ROOT) -> tuple[dict[str, list[SourceRecord]], JsonLikeDict]:
    """Audit all 5,312 real states and original split/benchmark/source-lock joins."""
    root = root.resolve()
    build = read_json(root / "build.json")
    source_paths = {
        "manifest": root / "manifest.jsonl", "build": root / "build.json",
        "historical_code_hashes": root / "source_hashes.json",
        "readout_metadata": root / "full_board_readout_v1/metadata.json",
        "token_inventory": root / "full_board_readout_v1/trainable_tokens.json",
        "benchmark_ledger": DEFAULT_LEAKAGE_LEDGER, "replay_source_lock": DEFAULT_SOURCE_LOCK,
        "builder": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/__init__.py",
        "builder_sources": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/_sources.py",
        "builder_components": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/_components.py",
        "builder_sampling": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/_sampling.py",
        "builder_transfer": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/_transfer.py",
        "builder_io": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/_io.py",
        "builder_types": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/_types.py",
        "builder_validate": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/_validate.py",
        "builder_build": PROJECT_ROOT
        / "sft/scripts/builders/build_symbolic_board_dataset/_build.py",
        "oracles": PROJECT_ROOT / "sft/board/symbolic_board_tasks/__init__.py",
        "oracles_constants": PROJECT_ROOT / "sft/board/symbolic_board_tasks/_constants.py",
        "oracles_geometry": PROJECT_ROOT / "sft/board/symbolic_board_tasks/_geometry.py",
        "oracles_contracts": PROJECT_ROOT / "sft/board/symbolic_board_tasks/_contracts.py",
        "oracles_solve": PROJECT_ROOT / "sft/board/symbolic_board_tasks/_solve.py",
        "oracles_scoring": PROJECT_ROOT / "sft/board/symbolic_board_tasks/_scoring.py",
        # The engine board is a package: the Board class plus its static graph
        # and longest-road helpers, each fingerprinted separately for provenance.
        "engine_board": PROJECT_ROOT / "cle/game_engine/models/board/core.py",
        "engine_board_graph": PROJECT_ROOT / "cle/game_engine/models/board/graph.py",
        "engine_board_roads": PROJECT_ROOT / "cle/game_engine/models/board/roads.py",
        "engine_board_package": PROJECT_ROOT / "cle/game_engine/models/board/__init__.py",
        "engine_map": PROJECT_ROOT / "cle/game_engine/models/map.py",
        # The token atlas is a package: the atlas tables, the vocabulary and the
        # manifest, each fingerprinted separately for provenance.
        "atlas": PROJECT_ROOT / "evals/catan_board_bench/tokens/atlas.py",
        "atlas_manifest": PROJECT_ROOT / "evals/catan_board_bench/tokens/manifest.py",
        "atlas_package": PROJECT_ROOT / "evals/catan_board_bench/tokens/__init__.py",
        "atlas_vocabulary": PROJECT_ROOT / "evals/catan_board_bench/tokens/vocabulary.py",
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
    parent = Path(as_str(build["parent"])).resolve()
    source_paths.update(parent_manifest=parent / "manifest.jsonl", parent_metadata=parent / "metadata.json")
    hashes = {name: _hash(path) for name, path in source_paths.items()}
    _check(build["manifest_sha256"] == hashes["manifest"]["sha256"], "source manifest changed")
    _check(build["parent_manifest_sha256"] == hashes["parent_manifest"]["sha256"], "parent manifest changed")
    parent_metadata = read_json(source_paths["parent_metadata"])
    lock = read_json(DEFAULT_SOURCE_LOCK)
    validate_replay_source_lock(lock)
    _check(source_lock_matches_metadata(
        lock, lock_sha256=as_str(parent_metadata["source_lock_sha256"]),
        file_sha256_value=as_str(parent_metadata["source_lock_file_sha256"])),
           "parent source lock changed")
    _, benchmark_ids = load_leakage_ledger()
    _check(len(benchmark_ids) == 13
           and benchmark_ids == set(as_list(as_dict(lock["leakage"])["excluded_game_ids"])),
           "expected the pinned 13-game benchmark exclusion ledger")
    _check(hashes["benchmark_ledger"]["sha256"] == parent_metadata["leakage_ledger_sha256"],
           "benchmark ledger hash changed")
    inventory = read_json(source_paths["token_inventory"])
    _check(inventory["tokens"] == atlas_tokens() and inventory["atlas_tokens"] == atlas_tokens(),
           "inventory must be the exact canonical 154-token sequence")
    rows = read_jsonl(source_paths["manifest"])
    manifest = {as_str(r["sample_id"]): r for r in rows}
    _check(len(manifest) == len(rows), "duplicate manifest sample identity")
    _check(dict(Counter(r["split"] for r in rows)) == SOURCE_COUNTS, "source split counts changed")
    for row in read_jsonl(source_paths["parent_manifest"]):
        _check(manifest.get(as_str(row["sample_id"])) == row,
               "diverse source changed a replay_v1 parent row")
    readouts: dict[str, JsonDict] = {}
    readout_metadata = read_json(source_paths["readout_metadata"])
    _check(readout_metadata["manifest_sha256"] == hashes["manifest"]["sha256"], "readout manifest changed")
    for split in SPLITS:
        path = root / "full_board_readout_v1/stage1" / f"{split}.jsonl"
        hashes["readout_" + split] = _hash(path)
        _check(file_sha256(path)
               == as_dict(as_dict(as_dict(build["export"])["files"])[split])["sha256"]
               == as_dict(as_dict(readout_metadata["files"])[split])["sha256"],
               f"readout file hash mismatch: {split}")
        split_rows = read_jsonl(path)
        _check(len(split_rows) == SOURCE_COUNTS[split] and
               len({r["state_id"] for r in split_rows}) == len(split_rows), "readout count/identity error")
        _check({r["state_id"] for r in split_rows} == {r["sample_id"] for r in rows if r["split"] == split},
               "readout source coverage mismatch")
        readouts.update({as_str(r["state_id"]): r for r in split_rows})
    groups: dict[str, defaultdict[str, set[str]]] = {
        name: defaultdict(set) for name in ("trajectory", "map", "game", "layout")}
    for row in rows:
        source, split = as_dict(row["source"]), as_str(row["split"])
        _check(source["split"] == split, "source split disagreement")
        groups["trajectory"][cast("str", source["trajectory_id"])].add(split)
        groups["map"][cast("str", row["board_map_sha256"])].add(split)
        groups["layout"][cast("str", readouts[as_str(row["sample_id"])]["layout_id"])].add(split)
        if source["game_id"] is not None:
            groups["game"][cast("str", source["game_id"])].add(split)
    for name in ("trajectory", "game", "layout"):
        _check(all(len(splits) == 1 for splits in groups[name].values()), f"cross-split {name} leakage")
    overlapping_maps: set[object] = set()
    for key, splits in groups["map"].items():
        _check(len(splits - {"train"}) <= 1, "cross-heldout-map leakage")
        if len(splits) > 1:
            overlapping_maps.add(key)
    accepted_lock = {as_dict(r)["game_id"]: as_dict(r) for r in as_list(lock["accepted"])}
    replay_hashes: dict[str, dict[str, str]] = {}
    admitted: dict[str, list[SourceRecord]] = {s: [] for s in SPLITS}
    exclusions: list[JsonLikeDict] = []
    physical_valid: Counter[str] = Counter()
    for source in rows:
        sid, info = as_str(source["sample_id"]), as_dict(source["source"])
        reasons: list[str] = []
        if info["game_id"] in benchmark_ids:
            reasons.append("benchmark_game")
        elif info["kind"] == "colonist_replay":
            _check(info["game_id"] in accepted_lock, f"unlocked replay source: {sid}")
            path = (PROJECT_ROOT / as_str(info["replay_path"])).resolve()
            if str(path) not in replay_hashes:
                replay_hashes[str(path)] = _hash(path)
            _check(replay_hashes[str(path)]["sha256"] == info["source_sha256"]
                   == accepted_lock[info["game_id"]]["sha256"], f"raw replay source changed: {sid}")
        else:
            _check(info["kind"] == "engine_rollout", f"unrecognized real source: {sid}")
        try:
            state, provenance = audit_source_state(root, source, readouts[sid])
            physical_valid[as_str(source["split"])] += 1
        except PhysicalStateError as exc:
            reasons.append(str(exc))
        if source["split"] == "color_diagnostic":
            reasons.append("diagnostic_reserved")
        if source["split"] == "train" and source["board_map_sha256"] in overlapping_maps:
            reasons.append("heldout_map_overlap")
        if reasons:
            exclusions.append({"state_id": sid, "split": source["split"], "reasons": reasons,
                               "source": info, "board_map_sha256": source["board_map_sha256"],
                               "contract": _hash(root / as_str(source["contract_path"])),
                               "labels": _hash(root / as_str(source["label_path"]))})
            if reasons == ["diagnostic_reserved"]:
                admitted["color_diagnostic"].append(
                    SourceRecord(state=state, provenance=provenance))
        else:
            admitted[as_str(source["split"])].append(
                SourceRecord(state=state, provenance=provenance))
    _check(all(admitted[s] for s in SPLITS[:3]), "source admission emptied a split")
    report: JsonLikeDict = {
        "source_root": str(root), "source_hashes": hashes,
        "raw_replay_hashes": list(replay_hashes.values()), "source_counts": SOURCE_COUNTS,
        "physical_valid_counts": dict(physical_valid),
        "admitted_counts": {s: len(admitted[s]) for s in SPLITS[:3]},
        "reserved_diagnostic_valid_count": len(admitted["color_diagnostic"]),
        "exclusions": exclusions, "benchmark_excluded_game_ids": sorted(benchmark_ids),
        "exclusion_reason_counts": dict(Counter(
            reason.split(":", 1)[0] for row in exclusions
            for reason in cast("list[str]", row["reasons"]))),
        "source_split_groups": {k: {v: sorted(s) for v, s in sorted(g.items())} for k, g in groups.items()},
        "certification": "Exact canonical topology/identity, unique ownership, piece supply caps and building distance only; reachable history NOT certified. Historical caches ignored, never repaired.",
        "asset_checks": "Contract, dense-label and full-readout hashes and semantic joins; raw replay hashes and engine-policy digests. Image files are never opened or hashed.",
    }
    return admitted, report
