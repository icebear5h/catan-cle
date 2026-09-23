from __future__ import annotations

from collections import Counter
from typing import TypeVar, cast

from data_pipeline.board_recognition.sources import canonical_sha256, file_sha256
from sft.board.board_fluency_scoring import score_board_fluency
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_list, as_str
from sft.scripts.builders import build_board_fluency_review as build_board_fluency_review
from sft.scripts.builders.build_symbolic_board_dataset import read_jsonl
from sft.scripts.builders.build_symbolic_board_dataset._types import SourceRecord

from ._sources import (
    OPERATIONS,
    PIP_OPERATIONS,
    REVIEW,
    REVIEW_SHA256,
    SPLITS,
    Donor,
    donor_for,
    pin,
    require,
)

review = build_board_fluency_review

# Cells are keyed by stratum name; flow groups by (operation, stratum, roster positions).
CellKey = TypeVar("CellKey", str, tuple[str, str, tuple[int, ...]])


def review_exclusions() -> JsonLikeDict:
    require(file_sha256(REVIEW) == REVIEW_SHA256, "approved review bytes changed")
    rows = read_jsonl(REVIEW)
    ids: set[str] = set()
    hashes: set[str] = set()
    facts: set[str] = set()
    for row in rows:
        m: JsonDict = {"schema": row["schema"], **as_dict(row["metadata"])}
        require(row["schema"] == review.SCHEMA, "review schema changed")
        gold = as_str(as_dict(as_list(row["messages"])[1])["content"])
        scored = score_board_fluency(gold, gold, m)
        require(bool(scored and scored["correct"]), "review gold/flags changed")
        require(m["state_sha256"] == canonical_sha256(as_dict(m["target"])["state"]),
                "review state hash")
        ids.add(as_str(m["state_id"]))
        hashes.add(as_str(m["state_sha256"]))
        facts.add(as_str(as_dict(m["provenance"])["board_fact_sha256"]))
    require(len(rows) == len(ids) == len(hashes) == 200, "expected 200 distinct reviewed states")
    return {"file": pin(REVIEW), "state_ids": sorted(ids), "state_sha256": sorted(hashes),
            "board_fact_sha256": sorted(facts)}


def eligible_sources(
    sources: dict[str, list[SourceRecord]], exclusions: JsonLikeDict,
) -> tuple[dict[str, list[Donor]], JsonLikeDict]:
    ids, hashes, facts = (set(cast("list[str]", exclusions[key])) for key in
                          ("state_ids", "state_sha256", "board_fact_sha256"))
    result: dict[str, list[Donor]] = {}
    report: JsonLikeDict = {}
    for split in SPLITS:
        unique: dict[str, Donor] = {}
        skipped: Counter[str] = Counter()
        for record in sources[split]:
            donor = donor_for(record)
            p = donor.provenance
            if p["state_id"] in ids or donor.state_hash in hashes or p["board_fact_sha256"] in facts:
                skipped["review_id_or_content"] += 1
            elif not donor.data["buildings"] or not donor.data["roads"]:
                skipped["no_buildings_or_no_roads"] += 1
            elif donor.state_hash in unique:
                skipped["duplicate_state_content"] += 1
            else:
                unique[donor.state_hash] = donor
        result[split] = list(unique.values())
        report[split] = {"admitted_source_states": len(sources[split]), "eligible_states": len(unique),
                         "terrain_layouts": len({d.terrain_hash for d in unique.values()}),
                         "skipped": dict(skipped)}
    return result, report


def roster_positions(candidate: review.Candidate[Donor]) -> tuple[int, ...]:
    colors = cast("list[str]", candidate.donor.state["colors"])
    return tuple(colors.index(cast("str", candidate.query[key]))
                 for key in ("color", "a", "b") if key in candidate.query)


def quotas(split: str) -> dict[str, int]:
    dynamic = 160 if split == "train" else 10 if split == "validation_eval" else 20
    return {op: (5 if split != "train" and op in PIP_OPERATIONS else dynamic) for op in OPERATIONS}


def apportion(capacities: dict[CellKey, int], total: int) -> dict[CellKey, int]:
    """Equal supported cells, saturating scarce cells rather than padding them."""
    require(sum(capacities.values()) >= total,
            f"unsupported quota: need {total}, unique capacity {sum(capacities.values())}")
    counts = dict.fromkeys(sorted(capacities), 0)
    for _ in range(total):
        key = min((k for k in counts if counts[k] < capacities[k]), key=lambda k: (counts[k], k))
        counts[key] += 1
    return counts
