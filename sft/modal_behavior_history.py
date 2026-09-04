"""Build behavior and forgetting matrices directly on the Modal runs volume."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from sft.behavior_diagnostics import history_markdown
from sft.scripts.report_behavior_history import build_report, parse_checkpoint


APP_NAME = "catan-behavior-history"
REMOTE_RUNS = "/runs"

app = modal.App(APP_NAME)
sft_runs = modal.Volume.from_name("catan-sft-runs", create_if_missing=True)
report_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("sft")


def parse_checkpoints(value: str) -> list[tuple[str, Path]]:
    items = [item.strip() for item in value.split(";") if item.strip()]
    if not items:
        raise ValueError("at least one LABEL=PATH checkpoint is required")
    parsed = [parse_checkpoint(item) for item in items]
    for _, path in parsed:
        if not str(path).startswith(f"{REMOTE_RUNS}/"):
            raise ValueError("checkpoint result roots must be container paths under /runs/")
    return parsed


@app.function(
    image=report_image,
    volumes={REMOTE_RUNS: sft_runs},
    timeout=60 * 30,
)
def report_remote(checkpoints: str, output_dir: str) -> dict:
    parsed = parse_checkpoints(checkpoints)
    report = build_report(parsed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "behavior_matrix.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output / "behavior_matrix.md").write_text(history_markdown(report))
    sft_runs.commit()
    return report


@app.local_entrypoint()
def main(
    checkpoints: str,
    label: str = "v3-pairs-v1-pairs-v2",
    dry_run: bool = True,
) -> None:
    """Use semicolons between LABEL=PATH items; repeat labels to merge backfills."""

    parsed = parse_checkpoints(checkpoints)
    if not label or "/" in label:
        raise ValueError("label must be one non-empty path component")
    output_dir = f"{REMOTE_RUNS}/qwen-series-eval/behavior-history/{label}"
    plan = {
        "schema": "catan_behavior_history_launch/v1",
        "checkpoints": [{"label": name, "path": str(path)} for name, path in parsed],
        "output_dir": output_dir,
        "dry_run": dry_run,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if dry_run:
        return
    result = report_remote.remote(checkpoints, output_dir)
    print(history_markdown(result))
