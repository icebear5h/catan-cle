"""Report the geometry of the trainable atlas-token rows in a PEFT adapter.

Training gates use this to catch entangled rows (near-duplicate directions, "twins") and
rows that point away from their own family (node, edge, tile, port) or away from the shared
mean direction. Both the input embedding rows and the lm_head rows are inspected. Every
cosine is computed over L2-normalized float32 rows; a family centroid is the normalized mean
of that family's raw rows, and the mean direction is the normalized mean of all rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._base import DEFAULT_FAMILY_FLOOR, DEFAULT_TOKEN_INVENTORY, DEFAULT_TWIN_THRESHOLD
from ._report import build_report, format_table


def parse_adapter_spec(spec: str) -> tuple[str, Path]:
    label, separator, raw_path = spec.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {spec!r}")
    return label.strip(), Path(raw_path.strip()).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--adapter",
        action="append",
        type=parse_adapter_spec,
        required=True,
        metavar="LABEL=PATH",
        help="adapter_model.safetensors to inspect; repeat for several adapters",
    )
    parser.add_argument(
        "--token-inventory",
        type=Path,
        default=DEFAULT_TOKEN_INVENTORY,
        help="trainable_tokens.json whose 'tokens' list orders the adapter rows",
    )
    parser.add_argument(
        "--twin-threshold",
        type=float,
        default=DEFAULT_TWIN_THRESHOLD,
        help="pairwise cosine above which two rows count as twins",
    )
    parser.add_argument(
        "--family-floor",
        type=float,
        default=DEFAULT_FAMILY_FLOOR,
        help="own-family loading below which a row is reported",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the JSON report here instead of printing it to stdout",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="skip the human-readable table on stderr"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        args.adapter,
        args.token_inventory,
        twin_threshold=args.twin_threshold,
        family_floor=args.family_floor,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    if not args.quiet:
        print(format_table(report), file=sys.stderr)
    return 0
