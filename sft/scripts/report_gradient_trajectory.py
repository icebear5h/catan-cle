"""Combine per-adapter gradient probes into a checkpoint trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sft.gradient_diagnostics import (
    build_gradient_trajectory,
    gradient_trajectory_markdown,
)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label.strip(), Path(raw_path).expanduser()


def load_report(path: Path) -> dict:
    report_path = path / "gradient_conflicts.json" if path.is_dir() else path
    report = json.loads(report_path.read_text())
    if report.get("schema") != "catan_gradient_conflict_probe/v1":
        raise ValueError(f"not a gradient-conflict report: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    trajectory = build_gradient_trajectory(
        [(label, load_report(path)) for label, path in args.checkpoint]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gradient_trajectory.json").write_text(
        json.dumps(trajectory, indent=2, sort_keys=True) + "\n"
    )
    markdown = gradient_trajectory_markdown(trajectory)
    (args.output_dir / "gradient_trajectory.md").write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
