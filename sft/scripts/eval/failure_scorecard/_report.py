"""report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sft.json_types import JsonValue, as_dict, load_json_dict, opt_dict
from sft.scripts.eval.analyze_occupancy_misses import (
    read_jsonl,
)
from sft.scripts.eval.eval_regression_panel import TOKEN_INVENTORY

from ._base import DEFAULT_CONTRACTS_DIR, SCHEMA, TABLE_KEYS, JsonDict
from ._scorer import Scorer
from ._sets import discover_sets, eval_jsonl_for, row_entanglement


def compute_deltas(report: JsonDict, baseline: JsonDict) -> JsonDict:
    """Per-set ``current - baseline`` for every numeric count present in both scorecards."""

    deltas: JsonDict = {}
    for name, entry in as_dict(report["sets"]).items():
        before = opt_dict(as_dict(as_dict(baseline.get("sets", {})).get(name, {})).get("counts"))
        if before is None:
            continue
        changes: JsonDict = {
            key: round(value - prior, 4)
            for key, value in as_dict(as_dict(entry)["counts"]).items()
            if isinstance(value, (int, float)) and isinstance(prior := before.get(key), (int, float))
        }
        deltas[name] = changes
    return deltas


def format_number(value: float | int | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return format(value, ("+d" if signed else "d") if isinstance(value, int) else ("+.3f" if signed else ".3f"))


def as_number(value: JsonValue) -> float | int | None:
    """Narrow a scorecard count, which is a number or ``None``."""

    if value is None or isinstance(value, (int, float)):
        return value
    raise TypeError(f"expected a number or null, got {type(value).__name__}")


def print_table(report: JsonDict) -> None:
    lines = []
    for name, entry in as_dict(report["sets"]).items():
        counts = as_dict(as_dict(entry)["counts"])
        delta = as_dict(as_dict(report.get("deltas", {})).get(name, {}))
        lines.append(f"{name}  rows {counts['rows']}  errors {counts['errors']}")
        for key in TABLE_KEYS:
            change = f"  ({format_number(as_number(delta[key]), signed=True)})" if key in delta else ""
            lines.append(f"  {key:<28}{format_number(as_number(counts[key])):>8}{change}")
    rows = opt_dict(report.get("row_entanglement"))
    if rows is not None and rows["available"]:
        lines.append(f"row_entanglement (input rows)  tokens_with_twin {rows['tokens_with_twin_count']}  rows_below_family_floor {rows['rows_below_family_floor_count']}")
    elif rows is not None:
        lines.append(f"row_entanglement unavailable: {rows['reason']}")
    print("\n".join(lines), file=sys.stderr)


def build_scorecard(panel_dir: Path, contracts_dir: Path, overrides: dict[str, Path], adapter: Path | None, inventory: Path) -> JsonDict:
    sets = discover_sets(panel_dir, overrides)
    if not sets:
        raise FileNotFoundError(f"no original-variant records.jsonl under {panel_dir}")
    scorer = Scorer(contracts_dir)
    generated_at = datetime.now(timezone.utc).isoformat()
    scored: JsonDict = {}
    report: JsonDict = {"schema": SCHEMA, "panel_dir": str(panel_dir), "generated_at": generated_at, "sets": scored}
    for set_id, records_path in sets:
        eval_path = eval_jsonl_for(set_id, overrides)
        entry = scorer.score_set(read_jsonl(records_path), read_jsonl(eval_path))
        set_entry: JsonDict = {"eval_jsonl": str(eval_path), "records": str(records_path), **entry}
        scored[set_id] = set_entry
    report["row_entanglement"] = row_entanglement(adapter, inventory)
    return report


def parse_override(spec: str) -> tuple[str, Path]:
    name, separator, path = spec.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(f"--set expects NAME=EVAL_JSONL, got {spec!r}")
    return name, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel-dir", type=Path, required=True, help="downloaded panel, or a single- or two-set eval dir")
    parser.add_argument("--contracts-dir", type=Path, default=DEFAULT_CONTRACTS_DIR)
    parser.add_argument("--adapter", type=Path, default=None, help="adapter directory for the optional row check")
    parser.add_argument("--token-inventory", type=Path, default=TOKEN_INVENTORY)
    parser.add_argument("--set", action="append", default=[], type=parse_override, metavar="NAME=EVAL_JSONL")
    parser.add_argument("--baseline", type=Path, default=None, help="previous scorecard JSON; adds per-mode deltas")
    parser.add_argument("--output", type=Path, default=None, help="write the scorecard here instead of stdout")
    parser.add_argument("--quiet", action="store_true", help="skip the human-readable table on stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_scorecard(args.panel_dir, args.contracts_dir, dict(args.set), args.adapter, args.token_inventory)
    if args.baseline is not None:
        report["baseline"] = str(args.baseline)
        report["deltas"] = compute_deltas(report, load_json_dict(args.baseline))
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    if not args.quiet:
        print_table(report)
    return 0
