"""One JSON scorecard of named failure modes for a trained board-reader checkpoint.

Every ``original``-variant ``records.jsonl`` under ``--panel-dir`` (a downloaded
regression panel, or a single- or two-set eval directory) is joined to its eval
jsonl and, for occupancy prompts, to a board: the state's contract for replay
rows, or only the placed target and partner pieces for synthetic rows. Misses
are bucketed into modes a training gate can compare directly instead of loss:
``blindness`` (occupied spot answered ``empty``), ``neighbor_confusion`` (a piece
within two same-type hops or touching across types, or the pair partner, named
instead of the truth), ``far_false_positive`` (a piece from elsewhere on the
board, or one that is not on the board), ``head_flip`` (the answer belongs to
another prompt head), ``token_glitch`` (one atlas token expected, response is not
exactly one token), ``orientation`` (edge-occupancy error rate, vertical versus
slanted edges), ``colour_dropout`` (occupied recall per colour), and, with
``--adapter``, ``row_entanglement`` from ``inspect_token_rows`` when importable.
A per-token table covers every atlas token a row is about: the prompt token, or
the expected token on inverse and localization rows.

Usage:

    uv run python -m sft.scripts.failure_scorecard --panel-dir <panel> \
        --output artifacts/runs/sft/scorecards/<label>.json --baseline <previous>.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_pipeline.board_recognition.replay_dataset import DEFAULT_STYLE_PATH, load_render_style
from data_pipeline.board_recognition.single_piece_localization import color_words
from data_pipeline.board_recognition.spatial_localization import atlas_regions
from sft.scripts.analyze_occupancy_misses import (
    CLASS_ORDER,
    OCCUPANCY_CATEGORIES,
    Board,
    answer_words,
    classify,
    eval_row_for,
    finalize_recall,
    queried_token,
    read_jsonl,
)
from sft.scripts.eval_regression_panel import REPLAY_ROOT, TOKEN_INVENTORY

try:
    from sft.scripts.inspect_token_rows import inspect_adapter
except ImportError:  # the row inspector is optional; the scorecard runs without it
    inspect_adapter = None

JsonDict = dict[str, Any]

SCHEMA = "catan_failure_scorecard/v1"
DEFAULT_CONTRACTS_DIR = REPLAY_ROOT / "contracts"
ATLAS_TOKEN_RE = re.compile(r"<[NETP][0-9_]+>")
ONE_TOKEN_RE = re.compile(r"^<[NETP][0-9_]+>$")
HEAD_RE = re.compile(r"^<image>\s*<[NETP][0-9_]+> (number|resource|building|road|port)\?$")
HEAD_TYPE = {"number": "number", "resource": "resource", "building": "occupancy", "road": "occupancy", "port": "port"}
RESOURCE_WORDS = frozenset({"wood", "brick", "sheep", "wheat", "ore", "desert"})
NEIGHBOR_CLASSES = (
    "neighbor_false_positive_hop1", "neighbor_false_positive_hop2", "cross_type_false_positive",
    "wrong_piece_neighbor_hop1", "wrong_piece_neighbor_hop2", "wrong_piece_cross_type",
)
FAR_CLASSES = ("false_positive_elsewhere", "false_positive_absent")
OTHER_CLASSES = ("right_color_wrong_type", "right_type_wrong_color", "wrong_piece_other")
COUNT_MODES = ("blindness", "neighbor_confusion", "far_false_positive", "other_occupancy_miss", "head_flip", "token_glitch")
TABLE_KEYS = COUNT_MODES + ("orientation_ratio", "colour_dropout_min_recall")
IMAGE_SIZE = 1024
VERTICAL_MAX_DEGREES = 15.0


def user_prompt(row: JsonDict | None) -> str:
    messages = row.get("messages", []) if row else []
    return next((str(m.get("content", "")) for m in messages if m.get("role") == "user"), "").strip()


def subject_token(prompt: str, expected: str, record: JsonDict, row: JsonDict | None) -> str | None:
    """The token a row is about: the prompt token, else a one-token answer, else row metadata."""

    if match := ATLAS_TOKEN_RE.search(prompt):
        return match.group(0)
    return expected if ONE_TOKEN_RE.match(expected) else queried_token(record, row)


def response_type(answer: str) -> str | None:
    if answer.isdigit() or answer == "none":
        return "number"
    if answer in RESOURCE_WORDS:
        return "resource"
    if answer == "empty" or answer.endswith((" settlement", " city", " road")):
        return "occupancy"
    return "port" if answer.endswith(" port") else None


def glitch_kind(response: str) -> str:
    found = ATLAS_TOKEN_RE.findall(response)
    return "multiple_tokens" if len(found) > 1 else "extra_text" if found else "no_token"


def is_synthetic(row: JsonDict) -> bool:
    return all(key in row for key in ("target_token", "piece", "color"))


def synthetic_board(base: Board, row: JsonDict) -> Board:
    """The single-piece or pair board: only the placed target (and partner) piece exists."""

    placements = {row["target_token"]: (row["color"], row["piece"])}
    if row.get("partner_token"):
        placements[row["partner_token"]] = (row["partner_color"], row["partner_piece"])
    board = copy.copy(base)
    board.piece = placements
    board.answers = {token: answer_words(color, piece) for token, (color, piece) in placements.items()}
    return board


def edge_orientations(contract: JsonDict) -> dict[str, str]:
    """Label each edge vertical or slanted from its rendered endpoint centres."""

    style = load_render_style(DEFAULT_STYLE_PATH)
    regions = atlas_regions(contract, image_size=IMAGE_SIZE, view_padding_factor=style.view_padding_factor)
    labels = {}
    for edge in contract["edges"]:
        (ax, ay), (bx, by) = (regions[token]["center_pixels"] for token in edge["node_tokens"])
        degrees = math.degrees(math.atan2(abs(bx - ax), abs(by - ay)))
        labels[edge["token"]] = "vertical" if degrees <= VERTICAL_MAX_DEGREES else "slanted"
    return labels


def rate_entry(errors: int, total: int) -> JsonDict:
    return {"errors": errors, "n": total, "error_rate": round(errors / total, 4) if total else None}


def class_group(classes: Counter[str], names: tuple[str, ...]) -> JsonDict:
    return {"count": sum(classes[n] for n in names), "by_class": {n: classes[n] for n in names if classes[n]}}


class Scorer:
    """Scores one set at a time; contracts and edge geometry are cached across sets."""

    def __init__(self, contracts_dir: Path) -> None:
        self.contracts_dir = contracts_dir
        self.boards: dict[str, Board] = {}
        self.orientation: dict[str, str] = {}

    def board(self, state_id: str) -> Board:
        if state_id not in self.boards:
            self.boards[state_id] = Board(json.loads((self.contracts_dir / f"{state_id}.json").read_text()))
        return self.boards[state_id]

    def orientation_of(self, token: str) -> str:
        if not self.orientation:  # the atlas geometry is the same on every contract
            contract = next(iter(sorted(self.contracts_dir.glob("*.json"))), None)
            if contract is None:
                raise FileNotFoundError(f"no contracts under {self.contracts_dir}")
            self.orientation = edge_orientations(json.loads(contract.read_text()))
        return self.orientation[token]

    def token_entry(self, token: str, stats: JsonDict) -> JsonDict:
        entry = {"n": stats["n"], "errors": stats["errors"], "error_rate": round(stats["errors"] / stats["n"], 4)}
        entry["dominant_wrong"] = stats["wrong"].most_common(1)[0][0] if stats["wrong"] else None
        entry.update({"head_flips": stats["head_flips"], "glitches": stats["glitches"]})
        if token.startswith("<E"):
            entry["orientation"] = self.orientation_of(token)
        return entry

    def score_set(self, records: list[JsonDict], eval_rows: list[JsonDict]) -> JsonDict:
        by_id = {str(row.get("id", row.get("row_id"))): row for row in eval_rows}
        tokens: dict[str, JsonDict] = defaultdict(lambda: {"n": 0, "errors": 0, "wrong": Counter(), "head_flips": 0, "glitches": 0})
        classes, heads, head_tokens, glitches, glitch_tokens, blind_piece, blind_color = (Counter() for _ in range(7))
        recall_color, recall_piece = (defaultdict(lambda: [0, 0]) for _ in range(2))
        orientation = {"vertical": [0, 0], "slanted": [0, 0]}
        errors = unjoined = 0
        for record in records:
            row = eval_row_for(record, by_id, eval_rows)
            score = record["score"]
            correct = bool(score["correct"])
            expected = str(score["expected_normalized"]).strip()
            response = str(score["response_normalized"]).strip()
            truth, answer = expected.lower(), response.lower()
            errors += not correct
            prompt = user_prompt(row)
            token = subject_token(prompt, expected, record, row)
            if token is not None:
                tokens[token]["n"] += 1
                if not correct:
                    tokens[token]["errors"] += 1
                    tokens[token]["wrong"][response] += 1
            if (match := HEAD_RE.match(prompt)) and (kind := response_type(answer)) not in (None, HEAD_TYPE[match.group(1)]):
                heads[f"{match.group(1)}->{kind}"] += 1
                head_tokens[token] += 1
                tokens[token]["head_flips"] += 1
            if ONE_TOKEN_RE.match(expected) and not ONE_TOKEN_RE.match(response):
                glitches[glitch_kind(response)] += 1
                glitch_tokens[token] += 1
                tokens[token]["glitches"] += 1
            metadata = record.get("metadata", {})
            if metadata.get("category") not in OCCUPANCY_CATEGORIES or token is None:
                continue
            if row is None:
                unjoined += 1
                continue
            base = self.board(row.get("state_id") or row.get("metadata", {}).get("state_id") or metadata["state_id"])
            board = synthetic_board(base, row) if is_synthetic(row) else base
            if token in board.piece:
                color, piece = board.piece[token]
                for bucket, key in ((recall_color, color_words(color)), (recall_piece, piece.lower())):
                    bucket[key][0] += correct
                    bucket[key][1] += 1
            if token.startswith("<E"):
                bucket = orientation[self.orientation_of(token)]
                bucket[0] += not correct
                bucket[1] += 1
            if correct:
                continue
            label = classify(board, token, truth, answer)
            classes[label] += 1
            if label == "false_negative":
                color, _, piece = truth.rpartition(" ")
                blind_color[color] += 1
                blind_piece[piece] += 1
        vertical, slanted = (rate_entry(*orientation[key]) for key in ("vertical", "slanted"))
        ratio = None
        if vertical["error_rate"] is not None and slanted["error_rate"]:
            ratio = round(vertical["error_rate"] / slanted["error_rate"], 4)
        recall = finalize_recall(recall_color)
        weakest = min(recall, key=lambda key: recall[key]["recall"]) if recall else None
        modes = {
            "blindness": {"count": classes["false_negative"], "by_piece": dict(sorted(blind_piece.items())), "by_color": dict(sorted(blind_color.items()))},
            "neighbor_confusion": class_group(classes, NEIGHBOR_CLASSES),
            "far_false_positive": class_group(classes, FAR_CLASSES),
            "other_occupancy_miss": class_group(classes, OTHER_CLASSES),
            "head_flip": {"count": sum(heads.values()), "by_head": dict(sorted(heads.items())), "tokens_with_3plus": {t: n for t, n in sorted(head_tokens.items()) if n >= 3}},
            "token_glitch": {"count": sum(glitches.values()), "by_kind": dict(sorted(glitches.items())), "tokens": dict(sorted(glitch_tokens.items()))},
            "orientation": {"vertical": vertical, "slanted": slanted, "ratio": ratio},
            "colour_dropout": {"recall_by_color": recall, "min_color": weakest, "min_recall": recall[weakest]["recall"] if weakest else None},
        }
        counts = {"rows": len(records), "errors": errors, **{key: modes[key]["count"] for key in COUNT_MODES}}
        counts.update({"orientation_ratio": ratio, "colour_dropout_min_recall": modes["colour_dropout"]["min_recall"]})
        return {
            "counts": counts,
            "modes": modes,
            "occupied_recall_by_piece": finalize_recall(recall_piece),
            "occupancy_classes": {label: classes[label] for label in CLASS_ORDER if classes[label]},
            "unjoined_records": unjoined,
            "per_token": {token: self.token_entry(token, stats) for token, stats in sorted(tokens.items())},
        }


FIRST_PANEL_SET = "spatial_localization_v1-stage1-validation"


def set_id_for(records_path: Path, info: JsonDict, metadata: JsonDict, overrides: dict[str, Path]) -> str | None:
    """Identify the eval set of one records file.

    Newer summaries carry ``eval_set_id``. Older ones only carry the remote
    ``eval_jsonl`` path, whose parent directory is the set name inside a
    multi-set panel and the panel root for the panel's first set. Failing
    both, a ``regression-panel-<set>-eval`` directory name is used, and a
    flat single-set directory is resolved by a lone ``--set`` override.
    """

    set_id = info.get("eval_set_id") or metadata.get("eval_set_id")
    if set_id:
        return str(set_id)
    remote = info.get("eval_jsonl")
    if remote:
        parent = Path(str(remote)).parent.name
        if parent == "regression-panel":
            return FIRST_PANEL_SET
        if parent.count("-") >= 2 or parent in overrides:
            return parent
    for ancestor in records_path.parents:
        name = ancestor.name
        if name == "catan-qwen-series-eval-regression-panel-eval":
            return FIRST_PANEL_SET
        if name.startswith("regression-panel-") and name.endswith("-eval"):
            return name[len("regression-panel-") : -len("-eval")]
    if len(overrides) == 1:
        return next(iter(overrides))
    return None


def discover_sets(panel_dir: Path, overrides: dict[str, Path] | None = None) -> list[tuple[str, Path]]:
    """Every original-variant records.jsonl under ``panel_dir`` with its eval set id."""

    overrides = overrides or {}
    found = []
    for records_path in sorted(panel_dir.rglob("records.jsonl")):
        summary_path = records_path.with_name("summary.json")
        info = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        with records_path.open() as handle:
            metadata = json.loads(next(handle, "{}")).get("metadata", {})
        if (info.get("image_variant") or metadata.get("eval_variant") or "original") != "original":
            continue
        set_id = set_id_for(records_path, info, metadata, overrides)
        if not set_id:
            raise ValueError(f"cannot tell which eval set {records_path} belongs to; pass --set NAME=PATH")
        found.append((set_id, records_path))
    return found


def eval_jsonl_for(set_id: str, overrides: dict[str, Path]) -> Path:
    """Invert an eval set id (its last three path parts joined by dashes) under the replay root."""

    if set_id in overrides:
        return overrides[set_id]
    parts = set_id.rsplit("-", 2)
    for root in (REPLAY_ROOT, REPLAY_ROOT.parent):
        candidate = root.joinpath(*parts).with_suffix(".jsonl")
        if len(parts) == 3 and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no eval jsonl found for set {set_id!r}; pass --set {set_id}=PATH")


def count_of(value: Any) -> int | None:
    return len(value) if isinstance(value, (list, tuple, dict, set)) else value if isinstance(value, int) else None


def row_entanglement(adapter: Path | None, inventory: Path) -> JsonDict | None:
    if adapter is None:
        return None
    if inspect_adapter is None:
        return {"available": False, "reason": "sft.scripts.inspect_token_rows is not importable"}
    rows = inspect_adapter(adapter, json.loads(inventory.read_text())["atlas_tokens"])["input"]
    twins, floor = rows.get("tokens_with_twin"), rows.get("rows_below_family_floor")
    return {
        "available": True, "adapter": str(adapter), "twins": rows.get("twins"),
        "tokens_with_twin": twins, "tokens_with_twin_count": count_of(twins),
        "rows_below_family_floor": floor, "rows_below_family_floor_count": count_of(floor),
    }


def compute_deltas(report: JsonDict, baseline: JsonDict) -> JsonDict:
    """Per-set ``current - baseline`` for every numeric count present in both scorecards."""

    deltas = {}
    for name, entry in report["sets"].items():
        before = baseline.get("sets", {}).get(name, {}).get("counts")
        if before is None:
            continue
        deltas[name] = {
            key: round(value - before[key], 4)
            for key, value in entry["counts"].items()
            if isinstance(value, (int, float)) and isinstance(before.get(key), (int, float))
        }
    return deltas


def format_number(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return format(value, ("+d" if signed else "d") if isinstance(value, int) else ("+.3f" if signed else ".3f"))


def print_table(report: JsonDict) -> None:
    lines = []
    for name, entry in report["sets"].items():
        counts, delta = entry["counts"], report.get("deltas", {}).get(name, {})
        lines.append(f"{name}  rows {counts['rows']}  errors {counts['errors']}")
        for key in TABLE_KEYS:
            change = f"  ({format_number(delta[key], signed=True)})" if key in delta else ""
            lines.append(f"  {key:<28}{format_number(counts[key]):>8}{change}")
    rows = report.get("row_entanglement")
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
    report: JsonDict = {"schema": SCHEMA, "panel_dir": str(panel_dir), "generated_at": generated_at, "sets": {}}
    for set_id, records_path in sets:
        eval_path = eval_jsonl_for(set_id, overrides)
        entry = scorer.score_set(read_jsonl(records_path), read_jsonl(eval_path))
        report["sets"][set_id] = {"eval_jsonl": str(eval_path), "records": str(records_path), **entry}
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
        report["deltas"] = compute_deltas(report, json.loads(args.baseline.read_text()))
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    if not args.quiet:
        print_table(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
