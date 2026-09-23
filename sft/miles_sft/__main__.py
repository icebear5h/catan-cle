"""CLI for the explicit merge → primitive data → Miles SFT workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .config import BASE_REVISION, TrainPlan
from .data import inspect_corpus, prepare_dataset
from .merge import merge_checkpoint, validate_merged_export
from .merge._contracts import MANIFEST
from .preflight import file_hash, inspect_plan, tokenizer_for
from .run import execute


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("inspect", help="count primitive rows without loading a model")
    inspect.add_argument("source", type=Path)
    merge = commands.add_parser("merge", help="merge existing r04 adapter/rows/vision into a fresh HF base")
    merge.add_argument("--base", type=Path, required=True)
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="encode training-only topology primitives")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--max-tokens", type=int, default=4096)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--steps", type=int, default=2)
        command.add_argument("--batch-size", type=int, default=2)
        command.add_argument("--max-tokens", type=int, default=4096)
        command.add_argument("--learning-rate", type=float, default=5e-5)
        command.add_argument("--timeout-seconds", type=int, default=1740)
        if name == "run":
            command.add_argument("--miles-root", type=Path, default=Path("/opt/catan-miles"))
            command.add_argument("--megatron-root", type=Path, default=Path("/opt/catan-megatron"))
            command.add_argument("--bridge-root", type=Path, default=Path("/opt/catan-bridge"))
            command.add_argument("--execute", action="store_true")
    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    result: object
    if args.action == "inspect":
        result = inspect_corpus(args.source)
    elif args.action == "merge":
        result = {"manifest": str(merge_checkpoint(
            args.base, args.adapter, args.output, base_revision=BASE_REVISION,
        ))}
    elif args.action == "prepare":
        validate_merged_export(args.checkpoint)
        result = prepare_dataset(args.source, args.output, tokenizer_for(args.checkpoint),
                                 args.max_tokens, args.limit,
                                 tokenizer_identity=file_hash(args.checkpoint / MANIFEST))
    else:
        plan = TrainPlan(args.checkpoint, args.data, args.output, args.steps, args.batch_size,
                         args.max_tokens, args.learning_rate, args.timeout_seconds)
        result = inspect_plan(plan) if args.action == "preflight" else execute(
            plan, args.miles_root, args.megatron_root, args.bridge_root, dry_run=not args.execute,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
