"""Prepare named Catan panels, then launch Miles' built-in evaluation-only flow."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from sft.miles_eval.contracts import JsonObject, compact, json_object, parse_json, require
from sft.miles_eval.data import prepare_panel, read_panel
from sft.miles_eval.preflight import file_hash, preflight
from sft.paths import PROJECT_ROOT

MILES_REVISION = "12754e9507e64d5e537288da17793246e913c525"
CONTEXT = 4096
NEW_TOKENS = 512


def panel_spec(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or not path:
        raise argparse.ArgumentTypeError("panel must be NAME=/path/to/source.jsonl")
    return name, Path(path).expanduser().resolve()


def prepare(panels: Sequence[tuple[str, Path]], output: Path) -> JsonObject:
    require(bool(panels) and len({name for name, _ in panels}) == len(panels),
            "at least one panel is required; panel names must be unique")
    output.mkdir(parents=True, exist_ok=False)
    manifest: JsonObject = {
        "schema": "catan_miles_eval/v1", "miles_revision": MILES_REVISION,
        "panels": {name: prepare_panel(source, output / f"{name}.jsonl") for name, source in panels},
    }
    (output / "manifest.json").write_text(compact(manifest) + "\n")
    return manifest


def prepared_panels(directory: Path) -> dict[str, Path]:
    manifest = parse_json((directory / "manifest.json").read_text())
    require(manifest.get("schema") == "catan_miles_eval/v1"
            and manifest.get("miles_revision") == MILES_REVISION, "unsupported panel manifest")
    result: dict[str, Path] = {}
    for name, value in json_object(manifest["panels"]).items():
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is not None, "unsafe panel name")
        receipt = json_object(value)
        path = directory / f"{name}.jsonl"
        require(file_hash(path) == receipt["destination_sha256"], f"prepared panel changed: {name}")
        rows = read_panel(path)
        require(len(rows) == receipt["rows"], f"prepared row count changed: {name}")
        result[name] = path
    require(bool(result), "no evaluation panels")
    return result


def miles_arguments(checkpoint: Path, panels: dict[str, Path], output: Path) -> list[str]:
    """The HF-aware backend needs no Megatron conversion; debug mode skips training weights."""
    return [
        "--train-backend", "fsdp", "--debug-rollout-only", "--num-rollout", "0",
        "--hf-checkpoint", str(checkpoint),
        "--num-gpus-per-node", "1", "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "1",
        "--rollout-num-gpus", "1", "--rollout-num-gpus-per-engine", "1", "--eval-num-gpus", "0",
        "--rollout-batch-size", "1", "--global-batch-size", "1", "--n-samples-per-prompt", "1",
        "--disable-rollout-global-dataset", "--eval-interval", "1",
        "--eval-prompt-data", *(part for name, path in panels.items() for part in (name, str(path))),
        "--eval-input-key", "prompt", "--eval-label-key", "label", "--metadata-key", "metadata",
        "--apply-chat-template", "--apply-chat-template-kwargs", '{"enable_thinking":false}',
        "--n-samples-per-eval-prompt", "1", "--eval-temperature", "0", "--eval-top-p", "1",
        "--eval-top-k", "1", "--eval-max-response-len", str(NEW_TOKENS),
        "--rollout-max-response-len", str(NEW_TOKENS),
        "--custom-rm-path", "sft.miles_eval.hooks.reward",
        "--custom-eval-rollout-log-function-path", "sft.miles_eval.hooks.log_results",
        "--save-debug-rollout-data", str(output / "debug" / "{rollout_id}.pt"),
        "--sglang-context-length", str(CONTEXT), "--sglang-dtype", "bfloat16",
        "--sglang-mem-fraction-static", "0.8", "--sglang-max-running-requests", "16",
        "--sglang-disable-cuda-graph",
    ]


def launch(miles_root: Path, checkpoint: Path, data_dir: Path, output: Path, *, execute: bool) -> None:
    panels = prepared_panels(data_dir)
    command = [sys.executable, "-m", "sft.miles_eval.worker", *miles_arguments(checkpoint, panels, output)]
    print(shlex.join([
        sys.executable, "-m", "sft.miles_eval.run", "run", "--miles-root", str(miles_root),
        "--hf-checkpoint", str(checkpoint), "--data-dir", str(data_dir),
        "--output", str(output), "--execute",
    ]), flush=True)
    if not execute:
        return
    require((miles_root / "train.py").is_file(), "--miles-root must contain Miles train.py")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=miles_root, text=True).strip()
    require(revision == MILES_REVISION, f"Miles must be checked out at {MILES_REVISION}")
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=miles_root, text=True)
    require(not dirty.strip(), "Miles tracked source differs from the pinned revision")
    require(not output.exists(), "choose a new output directory")
    audit = preflight(checkpoint, panels, context=CONTEXT, new_tokens=NEW_TOKENS)
    output.mkdir(parents=True, exist_ok=False)
    (output / "debug").mkdir()
    receipt: JsonObject = {
        "miles_revision": revision, "command": list(command), "checkpoint": str(checkpoint),
        "panels": {name: {"path": str(path), "sha256": file_hash(path)} for name, path in panels.items()},
        "preflight": audit, "status": "prepared",
    }
    receipt_path = output / "launch.json"
    receipt_path.write_text(compact(receipt) + "\n")
    environment = {**os.environ, "CATAN_MILES_ROOT": str(miles_root),
                   "PYTHONPATH": os.pathsep.join((str(PROJECT_ROOT), str(miles_root), os.environ.get("PYTHONPATH", ""))),
                   "TOKENIZERS_PARALLELISM": "false"}
    try:
        subprocess.run(command, env=environment, check=True)
        summary = parse_json((output / "results/eval-0/summary.json").read_text())
        require(set(json_object(summary["panels"])) == set(panels), "incomplete evaluation output")
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        receipt_path.write_text(compact(receipt) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prep = commands.add_parser("prepare", help="validate/project local panels without Miles or a GPU")
    prep.add_argument("--panel", type=panel_spec, action="append", required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="print the eval command; --execute starts one local GPU job")
    run.add_argument("--miles-root", type=Path, required=True)
    run.add_argument("--hf-checkpoint", type=Path, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "prepare":
        destination = args.output.expanduser().resolve()
        manifest = prepare(args.panel, destination)
        print(compact({"output": str(destination), "panels": {
            name: json_object(value)["rows"] for name, value in json_object(manifest["panels"]).items()
        }}))
    else:
        launch(args.miles_root.expanduser().resolve(), args.hf_checkpoint.expanduser().resolve(),
               args.data_dir.expanduser().resolve(), args.output.expanduser().resolve(), execute=args.execute)


if __name__ == "__main__":
    main()
