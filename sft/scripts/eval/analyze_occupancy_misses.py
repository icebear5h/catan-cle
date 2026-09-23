"""Classify every wrong ``node.occupancy`` / ``edge.owner`` answer in a panel eval.

Each miss is joined to its eval prompt (for the queried token) and the state's
contract (for the truth), then bucketed: the model saw nothing
(``false_negative``), named a piece that sits a hop or two away
(``neighbor_false_positive_*``, ``wrong_piece_neighbor_*``), named the touching
piece of the other type (``*_cross_type``), named a piece from elsewhere on the
board, invented one, or got only the color or only the type right. Recall on
occupied spots is also split by color, piece, and how many pieces touch the spot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data_pipeline.board_recognition.single_piece_localization import (
    color_words,
    neighbor_distances,
    neighbor_tokens,
)
from sft.json_types import (
    JsonDict,
    JsonLikeDict,
    JsonValue,
    as_dict,
    as_float,
    as_int,
    as_list,
    as_str,
    load_json_dict,
    loads_json,
)

OCCUPANCY_CATEGORIES = ("node.occupancy", "edge.owner")
TOKEN_RE = re.compile(r"<[NE][^>]+>")
DEFAULT_CONTRACTS_DIR = Path("artifacts/generated/board_recognition/replay_v1/contracts")
MAX_HOPS = 2
EXAMPLES_PER_CLASS = 3
CLASS_ORDER = (
    "false_negative",
    "neighbor_false_positive_hop1",
    "neighbor_false_positive_hop2",
    "cross_type_false_positive",
    "false_positive_elsewhere",
    "false_positive_absent",
    "wrong_piece_neighbor_hop1",
    "wrong_piece_neighbor_hop2",
    "wrong_piece_cross_type",
    "right_color_wrong_type",
    "right_type_wrong_color",
    "wrong_piece_other",
)


def read_jsonl(path: Path) -> list[JsonDict]:
    with path.open() as handle:
        return [as_dict(loads_json(line)) for line in handle if line.strip()]


def eval_row_for(record: JsonDict, by_id: dict[str, JsonDict], rows: list[JsonDict]) -> JsonDict | None:
    """Join a record to its eval row by ``id``, then ``row_id``, then ``index`` order."""

    row = by_id.get(str(record.get("id")))
    if row is None and (index := record.get("index")) is not None and 1 <= as_int(index) <= len(rows):
        row = rows[as_int(index) - 1]
    return row


def queried_token(record: JsonDict, row: JsonDict | None) -> str | None:
    """Parse the queried token from the eval prompt, falling back to explicit token fields."""

    metadata = as_dict(record.get("metadata", {}))
    sources: list[JsonValue] = []
    if row is not None:
        messages = [as_dict(message) for message in as_list(row.get("messages", []))]
        sources += [message.get("content") for message in messages if message.get("role") == "user"]
        sources += [row.get("queried_token"), row.get("target_token")]
    sources += [metadata.get("queried_token"), metadata.get("target_token"), record.get("prompt")]
    for text in sources:
        if text and (match := TOKEN_RE.search(str(text))):
            return match.group(0)
    return None


class Board:
    """Piece placement plus same-type and cross-type adjacency for one contract."""

    def __init__(self, contract: JsonDict) -> None:
        self.piece: dict[str, tuple[str, str]] = {}
        nodes = [as_dict(node) for node in as_list(contract["nodes"])]
        edges = [as_dict(edge) for edge in as_list(contract["edges"])]
        for node in nodes:
            if node.get("building"):
                self.piece[as_str(node["token"])] = (as_str(node["color"]), as_str(node["building"]))
        for edge in edges:
            if edge.get("road_color"):
                self.piece[as_str(edge["token"])] = (as_str(edge["road_color"]), "ROAD")
        self.neighbors = neighbor_tokens(contract)
        self.touching = {as_str(node["token"]): _str_list(node["adjacent_edge_tokens"]) for node in nodes}
        self.touching.update({as_str(edge["token"]): _str_list(edge["node_tokens"]) for edge in edges})
        self.answers = {token: answer_words(color, piece) for token, (color, piece) in self.piece.items()}

    def truth(self, token: str) -> str:
        return self.answers.get(token, "empty")

    def named_hop(self, token: str, answer: str) -> int | None:
        """Smallest same-type hop distance (1..MAX_HOPS) to a piece answering ``answer``."""

        distances = neighbor_distances(self.neighbors, token, MAX_HOPS)
        hops = [hop for other, hop in distances.items() if hop > 0 and self.answers.get(other) == answer]
        return min(hops) if hops else None

    def named_cross_type(self, token: str, answer: str) -> bool:
        return any(self.answers.get(other) == answer for other in self.touching[token])

    def touching_pieces(self, token: str) -> int:
        same = sum(other in self.piece for other in self.neighbors[token])
        return same + sum(other in self.piece for other in self.touching[token])


def _str_list(value: JsonValue) -> list[str]:
    return [as_str(item) for item in as_list(value)]


def _pair_counts() -> defaultdict[str, list[int]]:
    return defaultdict(lambda: [0, 0])


def answer_words(color: str, piece: str) -> str:
    return f"{color_words(color)} {piece.lower()}"


def classify(board: Board, token: str, expected: str, response: str) -> str:
    if expected != "empty" and response == "empty":
        return "false_negative"
    hop = board.named_hop(token, response)
    cross = board.named_cross_type(token, response)
    if expected == "empty":
        if hop is not None:
            return f"neighbor_false_positive_hop{hop}"
        if cross:
            return "cross_type_false_positive"
        return "false_positive_elsewhere" if response in board.answers.values() else "false_positive_absent"
    if hop is not None:
        return f"wrong_piece_neighbor_hop{hop}"
    if cross:
        return "wrong_piece_cross_type"
    expected_color, _, expected_piece = expected.rpartition(" ")
    response_color, _, response_piece = response.rpartition(" ")
    if expected_color == response_color:
        return "right_color_wrong_type"
    if expected_piece == response_piece:
        return "right_type_wrong_color"
    return "wrong_piece_other"


def touching_bin(count: int) -> str:
    return "3+" if count >= 3 else str(count)


def finalize_recall(buckets: dict[str, list[int]]) -> dict[str, dict[str, int | float]]:
    return {
        key: {"correct": correct, "total": total, "recall": round(correct / total, 4)}
        for key, (correct, total) in sorted(buckets.items())
    }


def load_board(contracts_dir: Path, state_id: str, cache: dict[str, Board]) -> Board:
    if state_id not in cache:
        cache[state_id] = Board(load_json_dict(contracts_dir / f"{state_id}.json"))
    return cache[state_id]


def analyze(records: list[JsonDict], eval_rows: list[JsonDict], contracts_dir: Path) -> JsonLikeDict:
    by_id = {str(row.get("id", row.get("row_id"))): row for row in eval_rows}
    boards: dict[str, Board] = {}
    by_class: Counter[str] = Counter()
    by_class_by_density: dict[str, Counter[str]] = defaultdict(Counter)
    recall_color, recall_piece, recall_touching, accuracy = (_pair_counts() for _ in range(4))
    examples: dict[str, list[JsonDict]] = defaultdict(list)
    total = misses = 0
    for record in records:
        metadata = as_dict(record.get("metadata", {}))
        if metadata.get("category") not in OCCUPANCY_CATEGORIES:
            continue
        row = eval_row_for(record, by_id, eval_rows)
        token = queried_token(record, row)
        if token is None:
            raise ValueError(f"no queried token for record {record.get('id')!r}")
        board = load_board(contracts_dir, as_str(metadata["state_id"]), boards)
        density = str(metadata.get("density_bin", "unknown"))
        score = as_dict(record["score"])
        correct = bool(score["correct"])
        expected = board.truth(token)
        response = str(score.get("response_normalized", "")).strip()
        total += 1
        accuracy[f"{metadata['category']}/{density}"][1] += 1
        accuracy[f"{metadata['category']}/{density}"][0] += correct
        if token in board.piece:
            color, piece = board.piece[token]
            for bucket, key in (
                (recall_color, color),
                (recall_piece, piece),
                (recall_touching, touching_bin(board.touching_pieces(token))),
            ):
                bucket[key][1] += 1
                bucket[key][0] += correct
        if correct:
            continue
        misses += 1
        label = classify(board, token, expected, response)
        by_class[label] += 1
        by_class_by_density[label][density] += 1
        if len(examples[label]) < EXAMPLES_PER_CLASS:
            examples[label].append(
                {"id": record.get("id"), "density": density, "token": token, "expected": expected, "response": response}
            )
    ordered = [label for label in CLASS_ORDER if label in by_class]
    return {
        "total": total,
        "misses": misses,
        "by_class": {label: by_class[label] for label in ordered},
        "by_class_by_density": {label: dict(sorted(by_class_by_density[label].items())) for label in ordered},
        "occupied_recall_by_color": finalize_recall(recall_color),
        "occupied_recall_by_piece": finalize_recall(recall_piece),
        "occupied_recall_by_touching": finalize_recall(recall_touching),
        "accuracy_by_category_density": {
            key: {"correct": correct, "total": count, "accuracy": round(correct / count, 4)}
            for key, (correct, count) in sorted(accuracy.items())
        },
        "examples": {label: examples[label] for label in ordered},
    }


def _recall_line(entry: JsonDict, rate_key: str) -> str:
    return f"{entry['correct']:>4}/{entry['total']:<4}{as_float(entry[rate_key]):.3f}"


def print_summary(report: JsonLikeDict) -> None:
    lines = [f"occupancy rows: {report['total']}  misses: {report['misses']}", "miss classes:"]
    lines += [f"  {label:<32}{count:>5}" for label, count in as_dict(report["by_class"]).items()]
    for title, key in (
        ("occupied recall by color", "occupied_recall_by_color"),
        ("occupied recall by piece", "occupied_recall_by_piece"),
        ("occupied recall by touching pieces", "occupied_recall_by_touching"),
    ):
        lines.append(f"{title}:")
        lines += [f"  {key_:<16}{_recall_line(as_dict(v), 'recall')}" for key_, v in as_dict(report[key]).items()]
    lines.append("accuracy by category/density:")
    accuracy = as_dict(report["accuracy_by_category_density"])
    lines += [f"  {k:<32}{_recall_line(as_dict(v), 'accuracy')}" for k, v in accuracy.items()]
    print("\n".join(lines), file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True, help="panel records.jsonl")
    parser.add_argument("--eval-jsonl", type=Path, required=True, help="eval jsonl the records were scored against")
    parser.add_argument("--contracts-dir", type=Path, default=DEFAULT_CONTRACTS_DIR)
    parser.add_argument("--output", type=Path, default=None, help="also write the JSON report here")
    parser.add_argument("--quiet", action="store_true", help="skip the human-readable table on stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(read_jsonl(args.records), read_jsonl(args.eval_jsonl), args.contracts_dir)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if not args.quiet:
        print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
