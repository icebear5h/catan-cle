"""Run the fixed regression eval panel against one adapter checkpoint.

Every prior eval set is scored in one Modal container with a single model
load, so a checkpoint can be compared against all earlier checkpoints on the
same rows. Free-generation exact match and the location-only candidate score
are both recorded per set and per variant.

Usage:

    uv run python -m sft.scripts.eval_regression_panel \
        --adapter-dir /runs/catan-vision-sft/<run>/<identity>/checkpoints/checkpoint-256 \
        --label single-piece-v2-ck256

Add ``--dry-run`` to print the launch command without running it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPLAY_ROOT = Path("artifacts/generated/board_recognition/replay_v1")
TOKEN_INVENTORY = REPLAY_ROOT / "ms_swift_bidirectional_v1" / "trainable_tokens.json"
PANEL = (
    # (eval jsonl, image root, note)
    (REPLAY_ROOT / "spatial_localization_v1/stage1/validation.jsonl", REPLAY_ROOT / "spatial_localization_v1/images", "marker validation, diamond style"),
    (REPLAY_ROOT / "spatial_localization_v1/probes/validation.jsonl", REPLAY_ROOT / "spatial_localization_v1/images", "gray-dot probes, both sizes"),
    (REPLAY_ROOT / "spatial_localization_v2/stage1/validation.jsonl", REPLAY_ROOT / "spatial_localization_v2/images", "single-piece validation"),
    (REPLAY_ROOT / "spatial_localization_v3/stage1/validation.jsonl", REPLAY_ROOT / "spatial_localization_v3/images", "single-piece plus tile validation"),
    (REPLAY_ROOT / "spatial_localization_v7/stage1/validation.jsonl", REPLAY_ROOT / "spatial_localization_v7/images", "single-piece plus tile validation, one touching and one far negative per image"),
    (REPLAY_ROOT / "spatial_localization_pairs_v2/stage1/validation.jsonl", REPLAY_ROOT / "spatial_localization_pairs_v2/images", "adjacent-pair validation, road-heavy kind mix"),
    (REPLAY_ROOT / "spatial_localization_pairs_control_v1/stage1/validation.jsonl", REPLAY_ROOT / "spatial_localization_pairs_control_v1/images", "far-pair control: two pieces beyond three hops"),
    (REPLAY_ROOT / "evals/validation_v1.jsonl", REPLAY_ROOT / "images", "replay production heads and inverse rows"),
)
VARIANTS = "original,blank"


def build_command(adapter_dir: str, label: str, *, gpu: str, batch_size: int, limit: int | None) -> list[str]:
    if not adapter_dir.startswith("/runs/"):
        raise ValueError(
            "adapter_dir must be the container mount path under /runs/, not the "
            f"volume-relative path shown by `modal volume ls`: {adapter_dir}"
        )
    missing = [str(path) for path, root, _ in PANEL if not path.is_file() or not root.is_dir()]
    if missing:
        raise FileNotFoundError(f"regression panel inputs missing: {missing}")
    command = [
        # --detach keeps the ephemeral app alive after the entrypoint exits;
        # without it Modal cancels the spawned call.
        "uv", "run", "modal", "run", "--detach", "sft/modal_qwen_series_eval.py",
        "--eval-jsonl", ",".join(str(path) for path, _, _ in PANEL),
        "--image-root", ",".join(str(root) for _, root, _ in PANEL),
        "--token-inventory", str(TOKEN_INVENTORY),
        "--remote-dir", "catan-qwen-series-eval/regression-panel",
        "--output-dir", f"/runs/qwen-series-eval/regression-panel/{label}",
        "--adapter-dir", adapter_dir,
        "--model-id", "Qwen/Qwen3.8-27B",
        "--gpu", gpu,
        "--bits", "16",
        "--batch-size", str(batch_size),
        "--max-new-tokens", "16",
        "--image-variant", VARIANTS,
        # Spawn so a client-side disconnect or a stopped local app cannot
        # cancel the scoring; results land under --output-dir on the volume.
        "--spawn-eval",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, help="Remote checkpoint or final bundle directory")
    parser.add_argument("--label", required=True, help="Output directory name under regression-panel/")
    parser.add_argument("--gpu", default="h200", choices=("h200", "l40s"))
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--limit", type=int, help="Rows per set, for smoke runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    command = build_command(args.adapter_dir, args.label, gpu=args.gpu, batch_size=args.batch_size, limit=args.limit)
    print(json.dumps({"panel": [{"eval_jsonl": str(p), "note": n} for p, _, n in PANEL], "variants": VARIANTS, "command": command}, indent=2))
    if args.dry_run:
        return 0
    log_dir = Path("artifacts/runs/sft/launches")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"regression-panel-{args.label}.log"
    with log_path.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    print(f"log: {log_path}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
