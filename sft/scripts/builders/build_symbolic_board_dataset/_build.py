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
from collections import Counter
from pathlib import Path
from typing import cast

from evals.catan_board_bench.tokens import atlas_tokens
from sft.board.symbolic_board_tasks import STATIC_TASKS, TRAIN_TASKS
from sft.json_types import JsonDict, JsonLike, JsonLikeDict, as_dict, as_list, as_str

from ._components import directional_pair_manifest, known_direction_exposure
from ._sampling import component_rows
from ._sources import (
    DEFAULT_OUTPUT,
    DEFAULT_ROOT,
    EVAL_QUOTAS,
    SPLITS,
    TRAIN_QUOTAS,
    _check,
    _hash,
    load_sources,
)
from ._transfer import graph_case_coverage, transfer_rows
from ._validate import component_profile, preserved_v1_artifacts, validate_dataset, validate_rows


def _write(path: Path, value: JsonLike, *, jsonl: bool = False) -> None:
    # Exclusive file creation; a partially failed build is never silently overwritten.
    with path.open("x") as handle:
        if jsonl:
            for row in cast("list[JsonDict]", value):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        else:
            handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")


def build_dataset(output_dir: Path = DEFAULT_OUTPUT, *, root: Path = DEFAULT_ROOT,
                  seed: int = 46, dry_run: bool = False,
                  checkpoint: str | None = None) -> JsonLikeDict:
    """Construct/verify a 3,200-presentation pilot; optionally persist immutable files."""
    output, root = output_dir.resolve(), root.resolve()
    if output.exists() and not dry_run:
        raise FileExistsError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent must already exist: {output.parent}")
    preserved = preserved_v1_artifacts()
    sources, audit = load_sources(root)
    exposure = known_direction_exposure()
    pair_sources = as_dict(exposure["pair_sources"])
    exposed_pairs = [key.split() for key in pair_sources]
    pairs = directional_pair_manifest(seed, exposed_pairs)
    files: dict[str, list[JsonDict]] = {}
    sampling: dict[str, JsonLikeDict] = {}
    for i, split in enumerate(SPLITS[:3]):
        files[split], sampling[split] = component_rows(
            sources[split], split, pairs[split], TRAIN_QUOTAS if split == "train" else EVAL_QUOTAS, seed + i)
    transfer_reports: dict[str, JsonLikeDict] = {}
    for i, split in enumerate(("validation", "test")):
        name = "transfer_" + split
        files[name], transfer_reports[name] = transfer_rows(sources[split], name, seed + i + 10)
    for rows in files.values():
        for index, row in enumerate(rows):
            metadata_row = as_dict(row["metadata"])
            metadata_row["row_position"] = index
            if "directional_pair" in metadata_row:
                key = " ".join(as_str(t) for t in as_list(metadata_row["directional_pair"]))
                metadata_row["known_training_exposure"] = pair_sources.get(key, [])
    coverage = validate_rows(files, pairs, sources, exposure)
    selected = {as_str(as_dict(as_dict(r["metadata"])["provenance"])["state_id"]):
                as_dict(as_dict(r["metadata"])["provenance"])
                for rows in files.values() for r in rows
                if "provenance" in as_dict(r["metadata"])}
    metadata: JsonLikeDict = {
        "schema": "catan_symbolic_board_dataset/v2", "seed": seed, "source_audit": audit,
        "preserved_v1": preserved,
        "train_quotas": TRAIN_QUOTAS, "component_eval_quotas": EVAL_QUOTAS,
        "static_training_presentations": sum(TRAIN_QUOTAS[t] for t in STATIC_TASKS),
        "state_training_presentations": sum(TRAIN_QUOTAS[t] for t in TRAIN_TASKS - STATIC_TASKS),
        "row_order": "family round-robin, insertion-order quotas; explicit row_position; presentation counts, not optimizer steps",
        "row_ids": {s: [r["id"] for r in rows] for s, rows in files.items()},
        "counts": {s: len(rows) for s, rows in files.items()},
        "task_counts": {s: dict(Counter(as_str(r["task_type"]) for r in rows))
                        for s, rows in files.items()},
        "unique_prompt_counts": {
            s: len({as_str(as_dict(as_list(r["messages"])[0])["content"]) for r in rows})
            for s, rows in files.items()},
        "repetition_policy": "Fixed-atlas neighbor/incidence/oriented-step questions may repeat to meet presentation quotas; report unique prompts separately.",
        "source_kind_counts": {
            s: dict(Counter(
                as_str(as_dict(as_dict(as_dict(r["metadata"])["provenance"])["source"])["kind"])
                for r in rows if "provenance" in as_dict(r["metadata"])))
            for s, rows in files.items()},
        "answer_counts": {
            s: {t: dict(Counter(as_str(as_dict(as_list(r["messages"])[1])["content"])
                                for r in rows if r["task_type"] == t))
                for t in ("symbolic_direction", "symbolic_reachable", "symbolic_near",
                          "symbolic_local_constraint")}
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
        return {"dry_run": True, "counts": metadata["counts"],
                "source_admitted": audit["admitted_counts"],
                "source_exclusions": len(cast("list[object]", audit["exclusions"])),
                "inputs": inputs,
                "component_unique_states": {s: sampling[s]["unique_states"] for s in sampling},
                "transfer_selection": transfer_reports}
    # Parent existence was checked before any directory creation; no parents=True.
    output.mkdir(exist_ok=False)
    assets: dict[str, JsonLike] = {}
    for name, rows in files.items():
        path = output / f"{name}.jsonl"
        _write(path, cast("JsonLike", rows), jsonl=True)
        assets[name] = {**_hash(path), "rows": len(rows)}
    inventory: JsonLikeDict = {"schema": "catan_symbolic_token_inventory/v1", "tokens": atlas_tokens(),
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
            "source_exclusions": len(cast("list[object]", audit["exclusions"])),
            "inputs": inputs,
            "component_unique_states": {s: sampling[s]["unique_states"] for s in sampling},
            "transfer_graph_coverage": {
                s: as_dict(as_dict(r["population_graph_cases"])["counts"])
                for s, r in transfer_reports.items()}}


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
