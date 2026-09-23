from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from data_pipeline.board_recognition.sources import canonical_sha256
from sft.board.board_fluency_scoring import score_board_fluency
from sft.board.symbolic_board_tasks import atlas_geometry
from sft.json_types import JsonDict, JsonLikeDict, JsonValue, as_dict, as_list, as_str
from sft.scripts.builders import build_board_fluency_review as build_board_fluency_review
from sft.scripts.builders.build_symbolic_board_dataset import graph_case_coverage, read_json
from sft.scripts.builders.build_symbolic_board_dataset._types import SourceRecord

from ._rows import exposure, profile, queried_positions, validation_subset
from ._selection import quotas, roster_positions
from ._sources import (
    COUNTS,
    OPERATIONS,
    PIP_OPERATIONS,
    SCHEMA,
    SPLITS,
    Donor,
    ValidationReport,
    donor_for,
    meta,
    require,
)

review = build_board_fluency_review


def _query_arg(value: JsonValue) -> str | int:
    """Review query arguments are board tokens/colors (text) or dice rolls (integers)."""
    if not isinstance(value, str | int):
        raise TypeError(f"Expected a text or integer query argument, got {type(value).__name__}")
    return value


def validate_rows(
    files: dict[str, list[JsonDict]], sources: dict[str, list[SourceRecord]], exclusions: JsonLikeDict,
) -> ValidationReport:
    """Source admission, split/exclusion checks, independent gold and strict self-score."""
    require({name: len(rows) for name, rows in files.items()} == COUNTS, "corpus counts differ")
    atlas, checks = atlas_geometry(), Counter[str]()
    source_index = {split: {r["provenance"]["state_id"]: r for r in sources[split]} for split in SPLITS}
    blocked = {key: set(as_list(exclusions[key])) for key in ("state_ids", "state_sha256", "board_fact_sha256")}
    groups: dict[str, defaultdict[JsonValue, set[str]]] = {key: defaultdict(set) for key in
              ("state_id", "state_sha256", "board_fact_sha256", "board_map_sha256", "terrain_sha256",
               "game_id", "trajectory_id", "layout_id")}
    seen_ids: set[JsonValue] = set()
    contracts: dict[str, JsonDict] = {}
    donors: dict[str, Donor] = {}
    for split in SPLITS:
        rows = files[split]
        presentations: set[tuple[str, str, str]] = set()
        prompts: set[str] = set()
        require(Counter(meta(r)["operation"] for r in rows) == quotas(split), f"{split}: operation quotas")
        for position, row in enumerate(rows):
            m, messages = meta(row), [as_dict(msg) for msg in as_list(row["messages"])]
            require(set(row) == {"schema", "split", "task_role", "review_only", "admitted_for_training",
                                 "id", "row_id", "messages", "metadata"}, "unexpected model row keys")
            for key, value in (("schema", SCHEMA), ("split", split), ("review_only", False),
                               ("admitted_for_training", split == "train"),
                               ("task_role", "train" if split == "train" else "component_eval")):
                require(row.get(key) == m.get(key) == value and type(row[key]) is type(value)
                        and type(m[key]) is type(value), f"conflicting declaration: {key}")
            require(row["id"] == row["row_id"] and row["id"] not in seen_ids, "duplicate row identity")
            seen_ids.add(row["id"])
            require(m["row_position"] == position, "row order mismatch")
            require(len(messages) == 2 and [msg["role"] for msg in messages] == ["user", "assistant"]
                    and all(set(msg) == {"role", "content"} and isinstance(msg["content"], str) for msg in messages),
                    "only two text messages are admitted")
            record, target = source_index[split].get(m["state_id"]), as_dict(m["target"])
            if record is None or record["provenance"] != m["provenance"] or record["state"] != target["state"]:
                raise ValueError("row is not an exact admitted source")
            state_id = as_str(m["state_id"])
            if state_id not in donors:
                donors[state_id] = donor_for(record)
            donor = donors[state_id]
            p, op, q = donor.provenance, as_str(m["operation"]), as_dict(target["query"])
            require(bool(donor.data["buildings"] and donor.data["roads"]), "empty donor")
            require(m["state_sha256"] == donor.state_hash and m["terrain_sha256"] == donor.terrain_hash
                    and m["query_sha256"] == canonical_sha256(q), "state/query/terrain hash mismatch")
            require(m["state_id"] not in blocked["state_ids"] and donor.state_hash not in blocked["state_sha256"]
                    and p["board_fact_sha256"] not in blocked["board_fact_sha256"], "review state/content leakage")
            args = {key: _query_arg(value) for key, value in q.items() if key != "operation"}
            candidate = review.Candidate(donor, op, args, as_str(m["selection_stratum"]), 0, 0.0)
            require(m["queried_roster_positions"] == list(roster_positions(candidate)), "roster metadata mismatch")
            text = review.question(op, args)
            gold = review.answer(review.Facts(donor.data, atlas), op, args)
            require(m["question"] == text and messages[0]["content"] == review.prompt(donor.state, text),
                    "prompt not exact symbolic state plus reviewed query")
            require(m["answer"] == messages[1]["content"] == gold, "stored gold failed state recomputation")
            if donor.state_hash not in contracts:
                contracts[donor.state_hash] = read_json(Path(as_str(as_dict(p["paths"])["contract"])))
            require(gold == review.reference_answer(candidate, contracts[donor.state_hash], checks),
                    f"independent oracle mismatch: {op}")
            scored = score_board_fluency(gold, gold, m)
            require(bool(scored and scored["correct"]), "gold failed shared scorer")
            checks["shared_scorer_gold_checks"] += 1
            presentation = (donor.terrain_hash if op in PIP_OPERATIONS else donor.state_hash, op, review.compact(q))
            prompt = as_str(messages[0]["content"])
            require(presentation not in presentations and prompt not in prompts, "duplicate question")
            presentations.add(presentation)
            prompts.add(prompt)
            identities: JsonDict = {"state_id": m["state_id"], "state_sha256": donor.state_hash,
                                    "terrain_sha256": donor.terrain_hash, **p, **as_dict(p["source"])}
            for name, group in groups.items():
                if identities[name] is not None:
                    group[identities[name]].add(split)
        if split == "train":
            require(max(Counter(meta(r)["state_sha256"] for r in rows).values()) <= 4,
                    "training state cap exceeded")
            roster_cells: defaultdict[tuple[str, str], Counter[tuple[int, ...]]] = defaultdict(Counter)
            for row in rows:
                m = meta(row)
                roster_cells[as_str(m["operation"]), as_str(m["selection_stratum"])][queried_positions(m)] += 1
            for (op, cell), counts in roster_cells.items():
                require(max(counts.values()) - min(counts.values()) <= 1,
                        f"training roster imbalance: {op}/{cell}")
                if any(len(pos) == 1 for pos in counts):
                    require(set(counts) == {(i,) for i in range(4)}, f"missing queried roster position: {op}/{cell}")
        else:
            for op in PIP_OPERATIONS:
                require(len({as_dict(meta(r)["provenance"])["board_map_sha256"] for r in rows
                             if meta(r)["operation"] == op}) == 5, "heldout pip rows need all five maps")
    for name, group in groups.items():
        require(all(len(splits) == 1 for splits in group.values()), f"cross-split {name} leakage")
    require(files["validation_eval"] == validation_subset(files["validation"]), "fixed validation subset changed")
    require(Counter(meta(r)["operation"] for r in files["validation_eval"]) == quotas("validation_eval"),
            "validation eval operation quota")
    first = Counter(meta(r)["operation"] for r in files["train"][:1024])
    require(set(first) == set(OPERATIONS) and max(first.values()) - min(first.values()) <= 1,
            "first 1024 operation coverage is not near-uniform")
    heldout_graphs = {split: graph_case_coverage(sources[split]) for split in ("validation", "test")}
    return {"checks": dict(checks), "split_disjoint_keys": list(groups), "review_overlap": 0,
            "heldout_graph_coverage": heldout_graphs,
            "profiles": {split: profile(rows) for split, rows in files.items()},
            "first_1024": exposure(files["train"][:1024]), "all_train": exposure(files["train"])}
