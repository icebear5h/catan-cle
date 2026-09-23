"""Command line entry point for the board-parts failure probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.probes.probe_catan_board_parts.report import (
    build_payload,
    render_text_report,
)
from scripts.probes.probe_catan_board_parts.stats import (
    JsonDict,
    analyze_records,
    top_failures_by_part,
)

# The pre-split module path stays the advertised program name and description.
PROG = "probe_catan_board_parts.py"
DESCRIPTION = """Probe Catan eval outputs for part-level failure diagnostics.

This script ingests an OpenRouter eval JSONL (or any similarly shaped run log)
and reports:
- part-level exact/component accuracy (tile/node/edge/port/robber/player)
- common failure modes (wrong color, wrong occupancy, missed EMPTY/NONE, etc.)
- top failed examples for each part to speed up data / prompt iteration.
"""

__all__ = ["DESCRIPTION", "PROG", "build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    parser.add_argument("--responses", type=Path, required=True, help="Eval JSONL from eval runner.")
    parser.add_argument("--model", type=str, default=None, help="Filter to single model_key (optional).")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output report path.")
    parser.add_argument("--top-failures", type=int, default=10, help="Examples per part in report.")
    return parser


def _load_records(path: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"{path} contains a non-object row")
        records.append(record)
    return records


def main() -> int:
    args = build_parser().parse_args()
    records = _load_records(args.responses)

    result = analyze_records(records, model_filter=args.model)
    fail_examples = {
        part: ex[: args.top_failures]
        for part, ex in top_failures_by_part(result.parts, args.top_failures).items()
    }

    print(render_text_report(result, top_n=max(5, min(args.top_failures, 15))))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as handle:
            payload = build_payload(
                result,
                fail_examples,
                responses=args.responses,
                model_filter=args.model,
                top_failures=args.top_failures,
            )
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    return 0
