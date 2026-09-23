"""Randomized pack composition and single-board pack generation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence

from sft.board.board_packs._model import (
    SCHEMA,
    Pack,
    PackEntry,
    _share_signature,
    sample_entry_count,
)
from sft.board.board_packs._scoring import render_pack
from sft.board.symbolic_board_tasks._types import Atlas
from sft.json_types import JsonLikeDict, as_str
from sft.scripts.builders.build_board_fluency_review import (
    Candidate,
    Donor,
    Facts,
    candidates_for,
)
from sft.scripts.builders.build_board_fluency_review import (
    answer as gold_answer,
)


def compose_pack(
    candidates: Sequence[Candidate[Donor]],
    *,
    rng: random.Random,
    max_entries: int = 48,
    share_rate: float = 0.5,
    max_per_operation: int | None = None,
) -> List[Candidate[Donor]]:
    """Pick a randomized, deduplicated composition of queries.

    With probability `share_rate` the next pick is drawn from candidates touching
    a board location already in the pack, which is what creates the
    retrieval-then-composition pairings; otherwise it is drawn uniformly so
    coverage does not collapse onto a few nodes.
    """
    if not candidates:
        return []

    pool = list(candidates)
    rng.shuffle(pool)
    target = sample_entry_count(rng, len(pool), max_entries)

    by_location: Dict[str, List[Candidate[Donor]]] = defaultdict(list)
    for candidate in pool:
        for location in _share_signature(candidate.query):
            by_location[location].append(candidate)

    chosen: List[Candidate[Donor]] = []
    seen: set[tuple[str, tuple[tuple[str, str | int], ...]]] = set()
    per_operation: Dict[str, int] = defaultdict(int)
    touched: set[str] = set()

    def key(candidate: Candidate[Donor]) -> tuple[str, tuple[tuple[str, str | int], ...]]:
        return (candidate.operation, tuple(sorted(candidate.query.items())))

    def admissible(candidate: Candidate[Donor]) -> bool:
        if key(candidate) in seen:
            return False
        if max_per_operation is not None:
            return per_operation[candidate.operation] < max_per_operation
        return True

    def take(candidate: Candidate[Donor]) -> None:
        chosen.append(candidate)
        seen.add(key(candidate))
        per_operation[candidate.operation] += 1
        touched.update(_share_signature(candidate.query))

    cursor = 0
    picked: Candidate[Donor] | None
    while len(chosen) < target and cursor < len(pool):
        picked = None
        if touched and rng.random() < share_rate:
            location = rng.choice(sorted(touched))
            options = [c for c in by_location.get(location, ()) if admissible(c)]
            if options:
                picked = rng.choice(options)
        if picked is None:
            while cursor < len(pool) and not admissible(pool[cursor]):
                cursor += 1
            if cursor >= len(pool):
                break
            picked = pool[cursor]
            cursor += 1
        take(picked)

    rng.shuffle(chosen)
    return chosen


def build_pack(
    board_text: str,
    facts: Facts,
    chosen: Sequence[Candidate[Donor]],
) -> Pack:
    entries = [
        PackEntry(
            index=i,
            operation=c.operation,
            query=c.query,
            expected=gold_answer(facts, c.operation, c.query),
            metadata={"stratum": c.stratum},
        )
        for i, c in enumerate(chosen, start=1)
    ]
    return Pack(board=board_text, entries=entries)


def generate_packs(
    donors: Iterable[Donor],
    atlas: Atlas,
    *,
    rng: random.Random,
    packs_per_board: int = 1,
    max_entries: int = 48,
    share_rate: float = 0.5,
    shared_rubrics: bool = True,
) -> Iterator[JsonLikeDict]:
    for donor in donors:
        candidates = candidates_for(donor, atlas, rng)
        if not candidates:
            continue
        facts = Facts(donor.data, atlas)
        board_text = as_str(donor.state["board"])
        for _ in range(packs_per_board):
            chosen = compose_pack(
                candidates, rng=rng, max_entries=max_entries, share_rate=share_rate
            )
            if not chosen:
                continue
            pack = build_pack(board_text, facts, chosen)
            prompt, target = render_pack(pack, shared_rubrics=shared_rubrics)
            yield {
                "schema": SCHEMA,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "schema": SCHEMA,
                    "class": "board_pack",
                    "entries": len(pack.entries),
                    "operations": pack.operations,
                    "shared_argument_entries": pack.shared_argument_entries(),
                    "queries": [
                        {"index": e.index, "operation": e.operation, "query": e.query,
                         "answer": e.expected, "stratum": e.metadata.get("stratum")}
                        for e in pack.entries
                    ],
                },
            }
