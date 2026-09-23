"""Board-state readouts that anchor a query pack.

The board-fluency corpus asks one question per board and discards the board, so
the connectivity operations (`reachable_nodes`, `shortest_distance`,
`component_count`, `component_roads`) each have to rediscover the same road
partition from scratch, and `reachable_nodes` sits at 1/10 on the review panel.

A readout states that partition outright: given a colour, list its road
components. It is a perception task, not a traversal one -- the model groups
edges it can see in the prompt rather than running a search -- and once it is in
the sequence the traversal queries riding along behind it are lookups into an
answer already given.

Component semantics are not redefined here. `Facts.components` in
`sft.scripts.builders.build_board_fluency_review` is the authority (two edges join through
a shared node unless an opponent building occupies that junction), and
`connected_roads` calls it so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

from sft.json_types import JsonLikeDict
from sft.scripts.builders.build_board_fluency_review import COMPONENT_RULES, Facts

SCHEMA = "catan_board_readout/v1"
OPERATION = "connected_roads"
NONE_ANSWER = "NONE"

QUESTION = (
    COMPONENT_RULES
    + "For {color}, list every road-edge component. Output one component per line "
    "as INDEX: EDGES, components ordered by their lowest edge token and edges "
    "sorted within each. Output NONE if the player has no roads. No explanation."
)


def connected_roads(facts: Facts, color: str) -> List[List[str]]:
    """Canonical partition of `color`'s road edges, deterministically ordered."""
    parts = [sorted(part) for part in facts.components(color)]
    parts.sort(key=lambda part: part[0] if part else "")
    return parts


def render_answer(parts: Sequence[Sequence[str]]) -> str:
    if not parts:
        return NONE_ANSWER
    return "\n".join(f"{i}: {' '.join(part)}" for i, part in enumerate(parts, start=1))


def render_question(color: str) -> str:
    return QUESTION.format(color=color)


def parse_answer(text: str) -> List[List[str]]:
    stripped = text.strip()
    if stripped == NONE_ANSWER:
        return []
    parts: List[List[str]] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        _, _, value = line.partition(":")
        edges = (value if ":" in line else line).split()
        if edges:
            parts.append(edges)
    return parts


@dataclass
class ReadoutScore:
    """Partition scoring.

    Exact match alone is close to useless for grading: one misplaced edge among
    twenty reads the same as answering nothing. `pair_agreement` is the
    Rand-style fraction of edge pairs placed consistently (same component in both
    gold and prediction, or different in both), which degrades smoothly, and the
    edge-level counts separate "grouped wrongly" from "not listed at all".
    """

    exact: bool
    components_expected: int
    components_predicted: int
    components_matched: int
    edges_expected: int
    edges_correct: int
    edges_missing: int
    edges_spurious: int
    pairs_total: int
    pairs_agreeing: int

    @property
    def component_recall(self) -> float:
        return (
            self.components_matched / self.components_expected
            if self.components_expected
            else 1.0
        )

    @property
    def edge_recall(self) -> float:
        return self.edges_correct / self.edges_expected if self.edges_expected else 1.0

    @property
    def pair_agreement(self) -> float:
        return self.pairs_agreeing / self.pairs_total if self.pairs_total else 1.0

    def to_dict(self) -> JsonLikeDict:
        return {
            "exact": self.exact,
            "components_expected": self.components_expected,
            "components_predicted": self.components_predicted,
            "component_recall": self.component_recall,
            "edge_recall": self.edge_recall,
            "edges_missing": self.edges_missing,
            "edges_spurious": self.edges_spurious,
            "pair_agreement": self.pair_agreement,
        }


def _membership(parts: Iterable[Sequence[str]]) -> Dict[str, int]:
    return {edge: i for i, part in enumerate(parts) for edge in part}


def score_readout(gold: Sequence[Sequence[str]], response: str) -> ReadoutScore:
    predicted = parse_answer(response)
    gold_sets = [frozenset(p) for p in gold]
    pred_sets = [frozenset(p) for p in predicted]

    gold_member = _membership(gold)
    pred_member = _membership(predicted)
    gold_edges = set(gold_member)
    pred_edges = set(pred_member)

    matched = sum(1 for part in gold_sets if part in pred_sets)

    pairs_total = pairs_agreeing = 0
    for a, b in combinations(sorted(gold_edges), 2):
        pairs_total += 1
        same_gold = gold_member[a] == gold_member[b]
        same_pred = (
            a in pred_member and b in pred_member and pred_member[a] == pred_member[b]
        )
        if same_gold == same_pred:
            pairs_agreeing += 1

    return ReadoutScore(
        exact=sorted(map(sorted, gold_sets)) == sorted(map(sorted, pred_sets)),
        components_expected=len(gold_sets),
        components_predicted=len(pred_sets),
        components_matched=matched,
        edges_expected=len(gold_edges),
        edges_correct=len(gold_edges & pred_edges),
        edges_missing=len(gold_edges - pred_edges),
        edges_spurious=len(pred_edges - gold_edges),
        pairs_total=pairs_total,
        pairs_agreeing=pairs_agreeing,
    )


def implied_component_count(parts: Sequence[Sequence[str]]) -> int:
    """What `component_count` must report if this readout is right."""
    return len(parts)


def implied_component_roads(parts: Sequence[Sequence[str]], edge: str) -> List[str]:
    """What `component_roads` must report for `edge` if this readout is right."""
    for part in parts:
        if edge in part:
            return list(part)
    raise KeyError(f"{edge} is not an owned road in this readout")


def build_readout(facts: Facts, color: str) -> Tuple[str, str, List[List[str]]]:
    parts = connected_roads(facts, color)
    return render_question(color), render_answer(parts), parts
