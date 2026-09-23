"""scorer."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from data_pipeline.board_recognition.single_piece_localization import color_words
from sft.json_types import as_dict, as_int, as_str, json_dict, load_json_dict
from sft.scripts.eval.analyze_occupancy_misses import (
    CLASS_ORDER,
    OCCUPANCY_CATEGORIES,
    Board,
    classify,
    eval_row_for,
    finalize_recall,
)

from ._base import (
    COUNT_MODES,
    FAR_CLASSES,
    HEAD_RE,
    HEAD_TYPE,
    NEIGHBOR_CLASSES,
    ONE_TOKEN_RE,
    OTHER_CLASSES,
    TERRAIN_RECALL,
    JsonDict,
)
from ._rows import (
    class_group,
    edge_orientations,
    error_rate,
    glitch_kind,
    is_synthetic,
    rate_entry,
    readout_items,
    readout_skips,
    response_type,
    subject_token,
    synthetic_board,
    user_prompt,
)

READOUT_FIELDS = (
    "count", "exact", "items_correct", "items_total", "items_extra", "occupied_items_correct",
    "occupied_items_total", "missing_tokens", "shifted_values", "sequence_skips",
)

SCORED_ITEM_FIELDS = ("items_correct", "items_total", "items_extra", "occupied_items_correct", "occupied_items_total")


@dataclass
class TokenStats:
    """Per-token tallies accumulated while scoring one set."""

    n: int = 0
    errors: int = 0
    wrong: Counter[str] = field(default_factory=Counter)
    head_flips: int = 0
    glitches: int = 0


def recall_json(recall: dict[str, dict[str, int | float]]) -> JsonDict:
    return {key: json_dict(entry) for key, entry in recall.items()}


def readout_json(entry: dict[str, int]) -> JsonDict:
    return {
        **entry,
        "exact_rate": round(entry["exact"] / entry["count"], 4),
        "item_rate": round(entry["items_correct"] / entry["items_total"], 4) if entry["items_total"] else None,
        "occupied_item_recall": round(entry["occupied_items_correct"] / entry["occupied_items_total"], 4) if entry["occupied_items_total"] else None,
    }


class Scorer:
    """Scores one set at a time; contracts and edge geometry are cached across sets."""

    def __init__(self, contracts_dir: Path) -> None:
        self.contracts_dir = contracts_dir
        self.boards: dict[str, Board] = {}
        self.orientation: dict[str, str] = {}

    def board(self, state_id: str) -> Board:
        if state_id not in self.boards:
            self.boards[state_id] = Board(load_json_dict(self.contracts_dir / f"{state_id}.json"))
        return self.boards[state_id]

    def orientation_of(self, token: str) -> str:
        if not self.orientation:  # the atlas geometry is the same on every contract
            contract = next(iter(sorted(self.contracts_dir.glob("*.json"))), None)
            if contract is None:
                raise FileNotFoundError(f"no contracts under {self.contracts_dir}")
            self.orientation = edge_orientations(load_json_dict(contract))
        return self.orientation[token]

    def token_entry(self, token: str, stats: TokenStats) -> JsonDict:
        entry: JsonDict = {"n": stats.n, "errors": stats.errors, "error_rate": round(stats.errors / stats.n, 4)}
        entry["dominant_wrong"] = stats.wrong.most_common(1)[0][0] if stats.wrong else None
        entry.update({"head_flips": stats.head_flips, "glitches": stats.glitches})
        if token.startswith("<E"):
            entry["orientation"] = self.orientation_of(token)
        return entry

    def score_set(self, records: list[JsonDict], eval_rows: list[JsonDict]) -> JsonDict:
        by_id = {str(row.get("id", row.get("row_id"))): row for row in eval_rows}
        tokens: defaultdict[str, TokenStats] = defaultdict(TokenStats)
        counters: tuple[Counter[str], ...] = tuple(Counter() for _ in range(7))
        classes, heads, head_tokens, glitches, glitch_tokens, blind_piece, blind_color = counters
        recall_color: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        recall_piece: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        orientation = {"vertical": [0, 0], "slanted": [0, 0]}
        readouts: defaultdict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(READOUT_FIELDS, 0))
        predicted_empty = [0, 0]  # correct, predicted
        terrain: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # category -> correct, total
        errors = unjoined = 0
        for record in records:
            row = eval_row_for(record, by_id, eval_rows)
            score = as_dict(record["score"])
            correct = bool(score["correct"])
            expected = str(score["expected_normalized"]).strip()
            response = str(score["response_normalized"]).strip()
            truth, answer = expected.lower(), response.lower()
            errors += not correct
            metadata = as_dict(record.get("metadata", {}))
            category = str(metadata.get("category", ""))
            if category in TERRAIN_RECALL.values():
                terrain[category][0] += correct
                terrain[category][1] += 1
            if category.endswith(".readout") or len(readout_items(expected)) >= 4:
                entry = readouts[category or "readout"]
                skips = readout_skips(expected, response)
                entry["count"] += 1
                entry["exact"] += correct
                for key in SCORED_ITEM_FIELDS:
                    entry[key] += as_int(score.get(key, 0))
                entry["missing_tokens"] += skips["missing_tokens"]
                entry["shifted_values"] += skips["shifted_values"]
                entry["sequence_skips"] += skips["skipped"]
                continue
            prompt = user_prompt(row)
            token = subject_token(prompt, expected, record, row)
            if token is not None:
                tokens[token].n += 1
                if not correct:
                    tokens[token].errors += 1
                    tokens[token].wrong[response] += 1
            # A head prompt or a one-token answer always yields a subject token.
            if token is not None and (match := HEAD_RE.match(prompt)) and (kind := response_type(answer)) not in (None, HEAD_TYPE[match.group(1)]):
                heads[f"{match.group(1)}->{kind}"] += 1
                head_tokens[token] += 1
                tokens[token].head_flips += 1
            if token is not None and ONE_TOKEN_RE.match(expected) and not ONE_TOKEN_RE.match(response):
                glitches[glitch_kind(response)] += 1
                glitch_tokens[token] += 1
                tokens[token].glitches += 1
            if metadata.get("category") in OCCUPANCY_CATEGORIES and answer == "empty":
                predicted_empty[1] += 1
                predicted_empty[0] += truth == "empty"
            if metadata.get("category") not in OCCUPANCY_CATEGORIES or token is None:
                continue
            if row is None:
                unjoined += 1
                continue
            base = self.board(as_str(row.get("state_id") or as_dict(row.get("metadata", {})).get("state_id") or metadata["state_id"]))
            board = synthetic_board(base, row) if is_synthetic(row) else base
            if token in board.piece:
                color, piece = board.piece[token]
                for bucket, key in ((recall_color, color_words(color)), (recall_piece, piece.lower())):
                    bucket[key][0] += correct
                    bucket[key][1] += 1
            if token.startswith("<E"):
                edge_bucket = orientation[self.orientation_of(token)]
                edge_bucket[0] += not correct
                edge_bucket[1] += 1
            if correct:
                continue
            label = classify(board, token, truth, answer)
            classes[label] += 1
            if label == "false_negative":
                color, _, piece = truth.rpartition(" ")
                blind_color[color] += 1
                blind_piece[piece] += 1
        vertical_rate, slanted_rate = (error_rate(*orientation[key]) for key in ("vertical", "slanted"))
        ratio = None
        if vertical_rate is not None and slanted_rate:
            ratio = round(vertical_rate / slanted_rate, 4)
        recall = finalize_recall(recall_color)
        weakest = min(recall, key=lambda key: recall[key]["recall"]) if recall else None
        min_recall = recall[weakest]["recall"] if weakest else None
        mode_counts = {
            "blindness": classes["false_negative"],
            "neighbor_confusion": sum(classes[n] for n in NEIGHBOR_CLASSES),
            "far_false_positive": sum(classes[n] for n in FAR_CLASSES),
            "other_occupancy_miss": sum(classes[n] for n in OTHER_CLASSES),
            "head_flip": sum(heads.values()),
            "token_glitch": sum(glitches.values()),
        }
        modes: JsonDict = {
            "blindness": {"count": mode_counts["blindness"], "by_piece": dict(sorted(blind_piece.items())), "by_color": dict(sorted(blind_color.items()))},
            "neighbor_confusion": class_group(classes, NEIGHBOR_CLASSES),
            "far_false_positive": class_group(classes, FAR_CLASSES),
            "other_occupancy_miss": class_group(classes, OTHER_CLASSES),
            "head_flip": {"count": mode_counts["head_flip"], "by_head": dict(sorted(heads.items())), "tokens_with_3plus": {t: n for t, n in sorted(head_tokens.items()) if n >= 3}},
            "token_glitch": {"count": mode_counts["token_glitch"], "by_kind": dict(sorted(glitches.items())), "tokens": dict(sorted(glitch_tokens.items()))},
            "orientation": {"vertical": rate_entry(*orientation["vertical"]), "slanted": rate_entry(*orientation["slanted"]), "ratio": ratio},
            "colour_dropout": {"recall_by_color": recall_json(recall), "min_color": weakest, "min_recall": min_recall},
            "readouts": {name: readout_json(entry) for name, entry in sorted(readouts.items())},
        }
        counts: JsonDict = {"rows": len(records), "errors": errors, **{key: mode_counts[key] for key in COUNT_MODES}}
        counts.update({"orientation_ratio": ratio, "colour_dropout_min_recall": min_recall})
        piece_recall = finalize_recall(recall_piece)
        occupied_items = sum(entry["occupied_items_total"] for entry in readouts.values())
        counts.update({
            "road_recall": piece_recall["road"]["recall"] if "road" in piece_recall else None,
            "settlement_recall": piece_recall["settlement"]["recall"] if "settlement" in piece_recall else None,
            "city_recall": piece_recall["city"]["recall"] if "city" in piece_recall else None,
            "empty_precision": round(predicted_empty[0] / predicted_empty[1], 4) if predicted_empty[1] else None,
            **{key: (round(terrain[category][0] / terrain[category][1], 4) if terrain[category][1] else None) for key, category in TERRAIN_RECALL.items()},
            "readouts": sum(entry["count"] for entry in readouts.values()),
            "readouts_exact": sum(entry["exact"] for entry in readouts.values()),
            "readout_occupied_item_recall": round(sum(entry["occupied_items_correct"] for entry in readouts.values()) / occupied_items, 4) if occupied_items else None,
            "sequence_skips": sum(entry["sequence_skips"] for entry in readouts.values()),
        })
        return {
            "counts": counts,
            "modes": modes,
            "occupied_recall_by_piece": recall_json(piece_recall),
            "occupancy_classes": {label: classes[label] for label in CLASS_ORDER if classes[label]},
            "unjoined_records": unjoined,
            "per_token": {token: self.token_entry(token, stats) for token, stats in sorted(tokens.items())},
        }
