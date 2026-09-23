"""Pure helpers for a paired, small visual-delta compression evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import TypedDict

import torch

from sft.json_types import JsonDict, as_bool, as_dict, as_int, as_str, json_list
from sft.scripts.eval.eval_qwen_vl_adapter import expected_text, normalize_text, user_text

TASK_COUNTS = {"tile_resource": 19, "tile_number": 19, "port_type": 9, "terrain_readout": 1}


class TerrainSelection(TypedDict):
    source_rows: int
    rows: int
    layouts: int
    states: dict[str, str]
    tasks: dict[str, int]
    row_ids: list[str]
    content_sha256: str
    selection: str


class PairedSummary(TypedDict):
    rows: int
    heads: dict[str, dict[str, float]]
    readouts: dict[str, int]
    wrong_rows: int
    failure_examples: list[JsonDict]


def _row_id(row: JsonDict) -> str:
    return as_str(row.get("row_id") or row["id"])


def select_terrain_boards(rows: list[JsonDict]) -> tuple[list[JsonDict], TerrainSelection]:
    """One median state per held-out layout; all 48 questions for that state."""
    layouts: dict[str, set[str]] = defaultdict(set)
    ids = [row.get("row_id") or row.get("id") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate source row IDs")
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("only validation rows are eligible")
        layouts[as_str(row["layout_id"])].add(as_str(row["state_id"]))
    states = {layout: sorted(values)[len(values) // 2] for layout, values in sorted(layouts.items())}
    chosen = sorted(
        [row for row in rows if states[as_str(row["layout_id"])] == row["state_id"]],
        key=lambda row: (as_str(row["layout_id"]), _row_id(row)),
    )
    for state in states.values():
        subset = [row for row in chosen if row["state_id"] == state]
        if Counter(as_str(row["task_type"]) for row in subset) != TASK_COUNTS:
            raise ValueError(f"incomplete terrain board: {state}")
        for task, count in TASK_COUNTS.items():
            tokens = [row["target_token"] for row in subset if row["task_type"] == task]
            if len(set(tokens)) != count:
                raise ValueError(f"duplicate/missing target addresses: {state} {task}")
    payload: list[JsonDict] = [
        {"id": _row_id(row), "layout": row["layout_id"],
         "state": row["state_id"], "task": row["task_type"],
         "prompt": user_text(row), "expected": expected_text(row)}
        for row in chosen
    ]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return chosen, {
        "source_rows": len(rows), "rows": len(chosen), "layouts": len(layouts),
        "states": states, "tasks": dict(Counter(as_str(row["task_type"]) for row in chosen)),
        "row_ids": [as_str(item["id"]) for item in payload], "content_sha256": digest,
        "selection": "median lexicographic state_id per layout; all terrain questions; independent of model predictions",
    }


def reconstruct_weight(base: torch.Tensor, trained: torch.Tensor,
                       factors: Mapping[str, torch.Tensor] | None,
                       rank: int | None) -> torch.Tensor:
    """Always reconstruct from the original base, never the previous variant.

    Nonmatrix tensors stay at v2. Rank zero resets matrices only, NOT all
    visual tensors and NOT the language/token parameters.
    """
    if rank is not None and rank < 0:
        raise ValueError("rank must be nonnegative or None for intact v2")
    if base.shape != trained.shape:
        raise ValueError("base and trained shapes differ")
    if rank is None or trained.ndim < 2:
        return trained.float().clone()
    original = base.bfloat16().float()
    if rank == 0:
        return original.clone()
    if factors is None:
        raise ValueError("matrix reconstruction needs SVD factors")
    a, b = factors["lora_A"].float(), factors["lora_B"].float()
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError("invalid A/B factor shapes")
    if rank > a.shape[0]:
        raise ValueError("requested rank exceeds saved factors")
    shape = (trained.shape[0], trained.numel() // trained.shape[0])
    if (b.shape[0], a.shape[1]) != shape:
        raise ValueError("factors do not match weight shape")
    return original + (b[:, :rank] @ a[:rank, :]).reshape(trained.shape)


def parse_strict_readout(text: str) -> tuple[list[tuple[str, str]], bool]:
    pairs: list[tuple[str, str]] = []
    valid = True
    for chunk in normalize_text(text).strip().rstrip(";").split(";"):
        match = re.fullmatch(r"\s*(<[NETP][0-9_]+>)\s+(.+?)\s*", chunk, re.DOTALL)
        if match is None:
            valid = False
            continue
        value = " ".join(match[2].split())
        if "<" in value or ">" in value:
            valid = False
        pairs.append((match[1], value))
    return pairs, valid


def strict_readout_score(expected: str, response: str) -> JsonDict:
    gold, gold_valid = parse_strict_readout(expected)
    if not gold_valid or len({token for token, _ in gold}) != len(gold):
        raise ValueError("invalid reference readout")
    predicted, syntax_valid = parse_strict_readout(response)
    counts = Counter(token for token, _ in predicted)
    duplicate_tokens = sorted(token for token, count in counts.items() if count > 1)
    unique = {token: value for token, value in predicted if counts[token] == 1}
    correct = sum(unique.get(token) == value for token, value in gold)
    semantic_exact = syntax_valid and not duplicate_tokens and unique == dict(gold)
    return {"items_correct": correct, "items_total": len(gold),
            "semantic_exact": semantic_exact, "ordered_exact": semantic_exact and predicted == gold,
            "syntax_valid": syntax_valid, "duplicate_tokens": json_list(duplicate_tokens),
            "missing_tokens": json_list(sorted(set(dict(gold)) - set(counts))),
            "extra_tokens": json_list(sorted(set(counts) - set(dict(gold))))}


def paired_summary(records: list[JsonDict], expected_ids: list[str]) -> PairedSummary:
    actual_ids = [record["id"] for record in records]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError("missing, extra, or duplicate prediction rows")
    heads: defaultdict[str, dict[str, float]] = defaultdict(lambda: {"correct": 0, "total": 0})
    readouts = {"ordered_exact": 0, "semantic_exact": 0, "total": 0,
                "items_correct": 0, "items_total": 0, "duplicate_rows": 0}
    failures: list[JsonDict] = []
    for record in records:
        task = as_str(as_dict(record["metadata"])["task_type"])
        expected, response = as_str(record["expected"]), as_str(record["response"])
        if task == "terrain_readout":
            score = strict_readout_score(expected, response)
            record["strict_readout"] = score
            readouts["total"] += 1
            for field in ("ordered_exact", "semantic_exact"):
                readouts[field] += as_bool(score[field])
            for field in ("items_correct", "items_total"):
                readouts[field] += as_int(score[field])
            readouts["duplicate_rows"] += bool(score["duplicate_tokens"])
            correct = as_bool(score["ordered_exact"])
        else:
            correct = normalize_text(expected) == normalize_text(response)
            heads[task]["total"] += 1
            heads[task]["correct"] += correct
        if not correct:
            failures.append({"id": record["id"], "task": task,
                             "expected": record["expected"], "response": record["response"]})
    for values in heads.values():
        values["accuracy"] = values["correct"] / values["total"]
    return {"rows": len(records), "heads": dict(heads), "readouts": readouts,
            "wrong_rows": len(failures), "failure_examples": failures[:20]}
