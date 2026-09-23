"""Prepare a small held-out reload panel through the established Miles evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sft.json_types import as_dict, load_json_dict, loads_json
from sft.miles_eval.run import prepare

from .data.contracts import STATIC_OPERATIONS
from .preflight import file_hash
from .run import write_json


def prepare_reload_panel(source: Path, output: Path, limit: int = 32) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if limit <= 0:
        raise ValueError("positive panel limit required")
    selected = []
    for line in source.read_text().splitlines():
        row = as_dict(loads_json(line))
        metadata = as_dict(row.get("metadata"))
        if metadata.get("split") not in ("validation", "test"):
            raise ValueError("reload source must contain held-out rows only")
        if metadata.get("task_type") in STATIC_OPERATIONS:
            selected.append(row)
    if len(selected) < limit:
        raise ValueError("not enough held-out primitive rows")
    output.mkdir(parents=True, exist_ok=False)
    subset = output / "source.jsonl"
    subset.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected[:limit]))
    prepared = prepare([("topology_reload", subset)], output / "panels")
    receipt: dict[str, object] = {
        "source": str(source.resolve()), "source_sha256": file_hash(source),
        "selected_rows": limit, "selection": "first admitted static rows in original source order",
        "purpose": "export reload smoke, not a matched training-quality comparison", "panels": prepared,
    }
    write_json(output / "selection.json", receipt)
    return receipt


def final_export(training: Path) -> Path:
    """Resolve only a completed runtime receipt, never the parent or raw Bridge model."""
    run = load_json_dict(training / "receipts/run.json")
    if run.get("complete") is not True:
        raise ValueError("training has no complete final receipt")
    checkpoint = as_dict(run["checkpoint"])
    path = Path(str(checkpoint["hf_checkpoint_dir"]))
    if path.name != "model" or not path.resolve().is_relative_to(training.resolve()):
        raise ValueError("final model must be inside the completed training output")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(prepare_reload_panel(args.source, args.output, args.limit), indent=2))


if __name__ == "__main__":
    main()
