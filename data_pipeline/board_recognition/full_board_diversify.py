"""Immutable full-board training shard from new legal engine trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data_pipeline.board_recognition.full_board_readout import export_full_board
from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH, BoardStateCandidate, _write_candidate, canonical_sha256,
    file_sha256, generate_engine_trajectory_candidates, load_render_style, read_jsonl,
    stable_seed, write_json, write_jsonl,
)
from data_pipeline.board_recognition.terrain_readout import layout_id, link_or_copy

QUOTAS = {"empty": 1, "setup": 3, "sparse": 5, "dense": 7}
DEFAULT_PARENT = Path("artifacts/generated/board_recognition/replay_v1")
DEFAULT_OUTPUT = Path("artifacts/generated/board_recognition/full_board_diverse_v1")


def placement_hash(candidate: BoardStateCandidate) -> str:
    """Ignore robber moves and other trajectory bookkeeping when deduplicating pieces."""
    contract = candidate.contract
    return canonical_sha256({
        "nodes": [(n["token"], n["color"], n["building"]) for n in contract["nodes"]],
        "edges": [(e["token"], e["road_color"]) for e in contract["edges"]],
    })


def select_positions(candidates: list[BoardStateCandidate], seed: int) -> list[BoardStateCandidate]:
    """Take 16 distinct piece configurations, spread across density and robber position."""
    unique = {}
    # A seeded representative prevents always preferring the first robber location.
    for candidate in sorted(candidates, key=lambda c: stable_seed(seed, c.board_fact_sha256)):
        unique.setdefault(placement_hash(candidate), candidate)
    selected = []
    robber_counts = Counter()
    for density, count in QUOTAS.items():
        pool = [c for c in unique.values() if c.density_bin == density]
        if len(pool) < count:
            return []
        for _ in range(count):
            choice = min(pool, key=lambda c: (
                robber_counts[c.contract["robber"]["tile_token"]],
                stable_seed(seed, "select", c.board_fact_sha256),
            ))
            selected.append(choice)
            robber_counts[choice.contract["robber"]["tile_token"]] += 1
            pool.remove(choice)
    return selected


def _generate_one(job: tuple) -> dict:
    index, seed, output_string, excluded_maps = job
    candidates = generate_engine_trajectory_candidates(
        trajectory_index=index, seed=seed, split="train", max_actions=1500,
    )
    if not candidates or candidates[0].board_map_sha256 in excluded_maps:
        return {"index": index, "reason": "layout_overlap", "rows": []}
    selected = select_positions(candidates, stable_seed(seed, index))
    if not selected:
        return {"index": index, "reason": "insufficient_unique_density_support", "rows": []}
    output = Path(output_string)
    style = load_render_style()
    rows = [_write_candidate(output, c, split="train", sample_index=index * 16 + i,
                             image_size=1024, style=style, style_path=DEFAULT_STYLE_PATH)
            for i, c in enumerate(selected)]
    return {"index": index, "rows": rows}


def build_shard(parent: Path, output: Path, *, layouts: int = 256, seed: int = 9072026,
                workers: int = 4) -> dict:
    if layouts < 1 or workers < 1:
        raise ValueError("layouts and workers must be positive")
    parent, output = parent.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    old = read_jsonl(parent / "manifest.jsonl")
    excluded_maps = {r["board_map_sha256"] for r in old}
    excluded_facts = {r["board_fact_sha256"] for r in old}
    reference = parent / "full_board_readout_v1"
    fixed_hashes = {s: file_sha256(reference / "stage1" / f"{s}.jsonl")
                    for s in ("validation", "test", "color_diagnostic")}
    for folder in ("contracts", "dense_labels", "images"):
        (output / folder).mkdir(parents=True)
    report = {"status": "building", "seed": seed, "requested_new_layouts": layouts,
              "positions_per_layout": QUOTAS, "max_actions_per_trajectory": 1500,
              "parent": str(parent), "parent_manifest_sha256": file_sha256(parent / "manifest.jsonl"),
              "fixed_evaluation_sha256": fixed_hashes, "skipped": {}, "completed_layouts": 0}
    write_json(output / "build.json", report)
    new_rows = []
    new_maps = set()
    attempted = 0
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            while len(new_maps) < layouts and attempted < layouts * 8:
                batch = min(workers, layouts - len(new_maps))
                jobs = [(i, seed, str(output), excluded_maps) for i in range(attempted, attempted + batch)]
                for result in pool.map(_generate_one, jobs):
                    rows = result["rows"]
                    if not rows:
                        reason = result["reason"]
                        report["skipped"][reason] = report["skipped"].get(reason, 0) + 1
                        continue
                    board_map = rows[0]["board_map_sha256"]
                    if board_map in new_maps or any(r["board_fact_sha256"] in excluded_facts for r in rows):
                        raise ValueError("duplicate generated board/layout")
                    new_maps.add(board_map)
                    new_rows.extend(rows)
                attempted += batch
                report.update(completed_layouts=len(new_maps), new_boards=len(new_rows), attempted_trajectories=attempted)
                write_json(output / "build.json", report)
                print(json.dumps({"layouts": len(new_maps), "target": layouts, "boards": len(new_rows),
                                  "attempted": attempted}), flush=True)
            if len(new_maps) != layouts:
                raise ValueError("could not fill the requested distinct-layout quota")
        # Original train examples and fixed evaluation assets remain byte-identical.
        for row in old:
            for field, kind in (("contract_path", "contract"), ("label_path", "labels"), ("image_path", "image")):
                source = parent / row[field]
                if file_sha256(source) != row["sha256"][kind]:
                    raise ValueError(f"parent asset changed: {source}")
                destination = output / row[field]
                if destination.exists():
                    raise ValueError(f"asset name collision: {destination}")
                link_or_copy(source, destination)
        manifest = sorted([*old, *new_rows], key=lambda r: r["sample_id"])
        if len({r["sample_id"] for r in manifest}) != len(manifest):
            raise ValueError("duplicate sample IDs")
        if len({r["board_fact_sha256"] for r in manifest}) != len(manifest):
            raise ValueError("duplicate board facts")
        write_jsonl(output / "manifest.jsonl", manifest)
        write_jsonl(output / "new_train_manifest.jsonl", new_rows)
        export = output / "full_board_readout_v1"
        metadata = export_full_board(output, export)
        for split, digest in fixed_hashes.items():
            if file_sha256(export / "stage1" / f"{split}.jsonl") != digest:
                raise ValueError(f"fixed {split} changed")
        colors, pieces, robber, locations, per_layout = Counter(), Counter(), Counter(), Counter(), defaultdict(Counter)
        for row in new_rows:
            contract = json.loads((output / row["contract_path"]).read_text())
            per_layout[layout_id(row["sample_id"])][row["density_bin"]] += 1
            robber[contract["robber"]["tile_token"]] += 1
            for node in contract["nodes"]:
                if node["building"]:
                    colors[node["color"]] += 1
                    pieces[f"{node['color']} {node['building']}"] += 1
                    locations[node["token"]] += 1
            for edge in contract["edges"]:
                if edge["road_color"]:
                    colors[edge["road_color"]] += 1
                    pieces[f"{edge['road_color']} ROAD"] += 1
                    locations[edge["token"]] += 1
        if any(dict(counts) != QUOTAS for counts in per_layout.values()):
            raise ValueError("per-layout density quota changed")
        report.update(status="completed", export=metadata, new_color_counts=dict(colors),
                      new_piece_counts=dict(pieces), new_robber_counts=dict(robber),
                      occupied_location_coverage=len(locations), new_occupied_counts_by_location=dict(locations),
                      train_only_new_layouts_disjoint_from_parent=True,
                      fixed_evaluation_bytes_unchanged=True,
                      manifest_sha256=file_sha256(output / "manifest.jsonl"),
                      new_manifest_sha256=file_sha256(output / "new_train_manifest.jsonl"),
                      generator_sha256=file_sha256(Path(__file__)),
                      engine_generator_sha256=file_sha256(Path(__file__).with_name("replay_dataset.py")))
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "build.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layouts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=9072026)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = build_shard(args.parent, args.output, layouts=args.layouts, seed=args.seed, workers=args.workers)
    print(json.dumps({"status": result["status"], "new_boards": result["new_boards"],
                      "completed_layouts": result["completed_layouts"]}))


if __name__ == "__main__":
    main()
