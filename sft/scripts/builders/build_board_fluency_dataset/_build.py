"""Build the approved, admitted symbolic board-fluency SFT corpus locally.

Reuse source admission and the reviewed candidates/questions/independent oracles.
No source generation, model calls, or historical artifact writes. --dry-run checks
the complete proposed corpus without writing; --validate is read-only admission
for launchers (the messages-only trainer cannot enforce these declarations).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.board_recognition.sources import file_sha256
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_list, as_str, dict_items
from sft.scripts.builders import build_board_fluency_review as build_board_fluency_review
from sft.scripts.builders.build_symbolic_board_dataset import (
    DEFAULT_ROOT,
    load_sources,
    read_json,
    read_jsonl,
)

from ._rows import make_rows, validation_subset
from ._select import select
from ._selection import eligible_sources, review_exclusions
from ._sources import (
    COUNTS,
    INVENTORY,
    OUTPUT,
    ROOT,
    SCHEMA,
    SEED,
    SPLITS,
    BuildSummary,
    Summary,
    ValidateSummary,
    pin,
    require,
)
from ._validate import validate_rows

review = build_board_fluency_review


def build_dataset(output: Path = OUTPUT, *, seed: int = SEED, dry_run: bool = False) -> BuildSummary:
    output = output.resolve()
    require(output.parent.is_dir(), "output parent must already exist")
    require(not output.exists(), "output already exists; validate it instead of overwriting")
    exclusions = review_exclusions()
    sources, source_audit = load_sources(DEFAULT_ROOT)
    donors, eligibility = eligible_sources(sources, exclusions)
    files: dict[str, list[JsonDict]] = {}
    selection: dict[str, JsonLikeDict] = {}
    for i, split in enumerate(SPLITS):
        selected, selection[split] = select(donors[split], split, seed + i)
        files[split] = make_rows(selected, split)
    files["validation_eval"] = validation_subset(files["validation"])
    validation = validate_rows(files, sources, exclusions)
    code = [ROOT / "sft/scripts/builders/build_board_fluency_dataset/__init__.py",
            ROOT / "sft/scripts/builders/build_board_fluency_dataset/_sources.py",
            ROOT / "sft/scripts/builders/build_board_fluency_dataset/_selection.py",
            ROOT / "sft/scripts/builders/build_board_fluency_dataset/_select.py",
            ROOT / "sft/scripts/builders/build_board_fluency_dataset/_rows.py",
            ROOT / "sft/scripts/builders/build_board_fluency_dataset/_validate.py",
            ROOT / "sft/scripts/builders/build_board_fluency_dataset/_build.py",
            ROOT / "sft/scripts/builders/build_board_fluency_review/__init__.py",
            ROOT / "sft/scripts/builders/build_board_fluency_review/_sources.py",
            ROOT / "sft/scripts/builders/build_board_fluency_review/_facts.py",
            ROOT / "sft/scripts/builders/build_board_fluency_review/_prompt.py",
            ROOT / "sft/scripts/builders/build_board_fluency_review/_candidates.py",
            ROOT / "sft/scripts/builders/build_board_fluency_review/_audit.py",
            ROOT / "sft/scripts/builders/build_board_fluency_review/_build.py", ROOT / "sft/board/board_fluency_scoring.py",
            ROOT / "sft/board/spatial_tasks/__init__.py",
            ROOT / "sft/board/spatial_tasks/_topology.py",
            ROOT / "sft/board/spatial_tasks/_contracts.py",
            ROOT / "sft/board/spatial_tasks/_scoring.py"]
    metadata = {
        "schema": SCHEMA, "corpus_status": "admitted_sft", "review_only": False,
        "admitted_for_training": True, "training_split": "train", "seed": seed, "counts": COUNTS,
        "families": review.FAMILIES, "source_audit": source_audit, "review_exclusions": exclusions,
        "eligibility": eligibility, "selection": selection, "validation": validation,
        "token_inventory_source": pin(INVENTORY), "build_code": [pin(path) for path in code],
        "training_plan": {
            "source_checkpoint": "/runs/catan-vision-sft/spatial-continuation-20260912-r01/checkpoints/checkpoint-128",
            "approved_language_lora_rank": 16, "optimizer_steps": 128, "effective_batch_size": 8,
            "planned_presentations": 1024, "corpus_presentations": 3200,
            "full_corpus_epoch_steps_at_batch_8": 400, "planned_fraction_of_corpus": 0.32,
            "order": "20-operation round-robin; deterministic shuffle within each operation. "
                     "First 1024 coverage assumes sequential single-pass consumption; launcher must enforce it.",
            "token_exposure": "See validation.first_1024 and validation.all_train. Atlas token occurrences "
                              "are exact; full native token exposure is measured by checkpoint-tokenizer preflight.",
        },
        "scope": [
            "Only the 20 approved reviewed operations; original nonempty admitted states and original splits.",
            "All 200 reviewed state IDs, canonical state hashes, and visible-board hashes excluded.",
            "Board-wide pip questions deduplicate on terrain, ignoring pieces, robber and ports.",
            "Full settlement conjunction and Longest Road remain transfer-only; strategy deferred to RL.",
            "Reserved color diagnostics contribute zero rows. No static atlas rehearsal.",
            "Heldout cycles/alternate routes and effective blockers have no source support; see absent strata.",
            "Admission certifies material state and source provenance, not independently reconstructed histories.",
            "Rows on a state/game/map are correlated; row count is not an independent sample count.",
        ],
    }
    summary: BuildSummary = {"valid": True, "dry_run": dry_run, "corpus_status": "admitted_sft", "counts": COUNTS,
               "output_dir": str(output), "eligibility": eligibility,
               "profiles": validation["profiles"], "checks": validation["checks"],
               "first_1024": validation["first_1024"]}
    if dry_run:
        return summary
    output.mkdir()
    for name, rows in files.items():
        (output / f"{name}.jsonl").write_text("".join(review.compact(row) + "\n" for row in rows))
    # A byte-for-byte copy of the successful trainer inventory, including its schema.
    (output / "trainable_tokens.json").write_bytes(INVENTORY.read_bytes())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    names = [f"{name}.jsonl" for name in files] + ["trainable_tokens.json", "metadata.json"]
    manifest = {"schema": SCHEMA, "corpus_status": "admitted_sft", "counts": COUNTS,
                "files": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size,
                                  **({"rows": COUNTS[Path(name).stem]} if name.endswith(".jsonl") else {})}
                          for name in names}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return summary


def validate_dataset(path: Path) -> ValidateSummary:
    """Read-only launcher gate: hashes/counts/splits/exclusions/source joins/all golds.

    Accept a corpus directory or its train.jsonl. Raises on any discrepancy; a
    successful return explicitly admits only train.jsonl for training.
    """
    output = path.resolve()
    if output.is_file():
        require(output.name == "train.jsonl", "only train.jsonl is an admitted training input")
        output = output.parent
    manifest = read_json(output / "manifest.json")
    require(manifest["schema"] == SCHEMA and manifest["corpus_status"] == "admitted_sft", "not an admitted corpus")
    names = {f"{name}.jsonl" for name in COUNTS} | {"trainable_tokens.json", "metadata.json"}
    require(set(as_dict(manifest["files"])) == names and manifest["counts"] == COUNTS,
            "manifest files/counts mismatch")
    for name, info in dict_items(manifest["files"]):
        asset = output / name
        require(asset.stat().st_size == info["bytes"] and file_sha256(asset) == info["sha256"],
                f"artifact bytes changed: {name}")
        if name.endswith(".jsonl"):
            require(info["rows"] == COUNTS[Path(name).stem], f"manifest row count: {name}")
    metadata = read_json(output / "metadata.json")
    require(metadata["schema"] == SCHEMA and metadata["corpus_status"] == "admitted_sft"
            and metadata["admitted_for_training"] is True and metadata["review_only"] is False
            and metadata["training_split"] == "train" and metadata["counts"] == COUNTS, "corpus admission changed")
    for pinned in as_list(metadata["build_code"]) + [metadata["token_inventory_source"]]:
        info = as_dict(pinned)
        require(file_sha256(Path(as_str(info["path"]))) == info["sha256"], f"pinned input changed: {info['path']}")
    require((output / "trainable_tokens.json").read_bytes() == INVENTORY.read_bytes(), "inventory must be exact copy")
    exclusions = review_exclusions()
    require(exclusions == metadata["review_exclusions"], "review exclusions changed")
    sources, source_audit = load_sources(DEFAULT_ROOT)
    require(source_audit == metadata["source_audit"], "source admission changed")
    files = {name: read_jsonl(output / f"{name}.jsonl") for name in COUNTS}
    validation = validate_rows(files, sources, exclusions)
    require(json.loads(json.dumps(validation)) == metadata["validation"], "coverage/exposure/gold audit changed")
    return {"valid": True, "schema": SCHEMA, "corpus_status": "admitted_sft", "counts": COUNTS,
            "admitted_for_training": True, "train_jsonl": str(output / "train.jsonl"),
            "manifest_sha256": file_sha256(output / "manifest.json"),
            "files": manifest["files"], "review_overlap": 0, "checks": validation["checks"],
            "profiles": validation["profiles"], "first_1024": validation["first_1024"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    result: Summary = (validate_dataset(args.output_dir) if args.validate else
              build_dataset(args.output_dir, seed=args.seed, dry_run=args.dry_run))
    # Aggregate facts only; source records never enter terminal/chat output.
    print(json.dumps({"valid": result["valid"], "counts": result["counts"], "checks": result["checks"]},
                     indent=2, sort_keys=True))
    print(json.dumps({split: {key: p[key] for key in
                             ("unique_states", "unique_maps", "max_presentations_per_state", "by_stratum")}
                      for split, p in result["profiles"].items()}, indent=2, sort_keys=True))
