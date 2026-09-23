"""CLI: python -m sft.miles_sft.merge {merge,validate,manifest} ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import load_manifest, merge_checkpoint, validate_merged_export
from ._contracts import object_map, read_json, valid_token_ids


def _inventory(path: Path) -> tuple[int, ...]:
    saved = read_json(path)
    if "semantic_tokens" in saved:
        saved = object_map(saved["semantic_tokens"])
    return valid_token_ids(saved.get("token_ids"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CPU streaming r04 merge for fresh Miles Megatron LoRA")
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge", help="export to a new directory; fail if it exists")
    merge.add_argument("--base-dir", type=Path, required=True)
    merge.add_argument("--adapter-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--base-revision", help="declared pinned HF base revision; content hashes are also recorded")
    ids = merge.add_mutually_exclusive_group()
    ids.add_argument("--token-ids", type=int, nargs="+", help="exact ordered 154 token IDs")
    ids.add_argument("--token-inventory", type=Path, help="JSON with token_ids or semantic_tokens.token_ids")
    for command in ("validate", "manifest"):
        sub = commands.add_parser(command)
        sub.add_argument("output", type=Path)
        sub.add_argument("--expected-sha256", help="externally pinned manifest SHA256")
    args = parser.parse_args(argv)
    if args.command == "merge":
        token_ids = _inventory(args.token_inventory) if args.token_inventory else (
            valid_token_ids(args.token_ids) if args.token_ids is not None else ())
        path = merge_checkpoint(args.base_dir, args.adapter_dir, args.output, token_ids,
                                base_revision=args.base_revision)
        print(path)
    else:
        reader = validate_merged_export if args.command == "validate" else load_manifest
        print(json.dumps(reader(args.output, expected_sha256=args.expected_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
