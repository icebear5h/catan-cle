"""Build behavior accuracy and forgetting matrices from checkpoint eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sft.behavior_diagnostics import (
    build_behavior_history,
    history_markdown,
    load_checkpoint_results,
)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label.strip(), Path(raw_path).expanduser()


def build_report(checkpoint_args: list[tuple[str, Path]]) -> dict:
    grouped: dict[str, list[Path]] = {}
    order: list[str] = []
    for label, path in checkpoint_args:
        if label not in grouped:
            grouped[label] = []
            order.append(label)
        grouped[label].append(path)
    checkpoints = [load_checkpoint_results(label, grouped[label]) for label in order]
    return build_behavior_history(checkpoints)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=parse_checkpoint,
        help="Chronological LABEL=PATH; repeat a label to merge targeted result roots.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    history = build_report(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "behavior_matrix.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n"
    )
    markdown = history_markdown(history)
    (args.output_dir / "behavior_matrix.md").write_text(markdown)
    print(markdown)
    return 1 if history["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
