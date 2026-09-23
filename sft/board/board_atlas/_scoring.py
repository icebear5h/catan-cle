"""Per-entry and per-atom scoring for queried fact subsets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from sft.board.board_atlas._tables import FactTable
from sft.board.board_atlas._tokens import NONE_ANSWER
from sft.json_types import JsonLikeDict

# --- scoring ---------------------------------------------------------------
#
# Exact match over a 24-entry dump scores a 23/24 answer as zero, which makes
# progress invisible. Atlas answers are scored per entry instead.


@dataclass
class AtlasScore:
    entries_total: int
    entries_correct: int
    entries_missing: int
    entries_malformed: int
    atoms_expected: int
    atoms_correct: int
    atoms_spurious: int
    exact: bool

    @property
    def entry_accuracy(self) -> float:
        return self.entries_correct / self.entries_total if self.entries_total else 0.0

    @property
    def atom_precision(self) -> float:
        got = self.atoms_correct + self.atoms_spurious
        return self.atoms_correct / got if got else 0.0

    @property
    def atom_recall(self) -> float:
        return self.atoms_correct / self.atoms_expected if self.atoms_expected else 0.0

    def to_dict(self) -> JsonLikeDict:
        return {
            "entries_total": self.entries_total,
            "entries_correct": self.entries_correct,
            "entries_missing": self.entries_missing,
            "entries_malformed": self.entries_malformed,
            "entry_accuracy": self.entry_accuracy,
            "atom_precision": self.atom_precision,
            "atom_recall": self.atom_recall,
            "exact": self.exact,
        }


def parse_answer(text: str) -> Dict[str, List[str]]:
    parsed: Dict[str, List[str]] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        atoms = value.split()
        parsed[key.strip()] = [] if atoms == [NONE_ANSWER] else atoms
    return parsed


def score_atlas(table: FactTable, keys: Sequence[str], response: str) -> AtlasScore:
    """Per-entry and per-atom scoring; ordering of atoms within an entry is free
    for set answers and significant for sequences."""
    got = parse_answer(response)
    entries_correct = entries_missing = entries_malformed = 0
    atoms_expected = atoms_correct = atoms_spurious = 0

    for key in keys:
        gold = table.facts[key]
        atoms_expected += len(gold)
        if key not in got:
            entries_missing += 1
            continue
        pred = got[key]
        if table.answer_kind == "sequence":
            if pred == gold:
                entries_correct += 1
                atoms_correct += len(gold)
            else:
                entries_malformed += 1
                atoms_correct += sum(1 for a, b in zip(pred, gold) if a == b)
                atoms_spurious += max(0, len(pred) - len(gold))
            continue
        gold_counts, pred_counts = Counter(gold), Counter(pred)
        overlap = sum((gold_counts & pred_counts).values())
        atoms_correct += overlap
        atoms_spurious += max(0, len(pred) - overlap)
        if pred_counts == gold_counts:
            entries_correct += 1
        else:
            entries_malformed += 1

    extra_keys = [k for k in got if k not in set(keys)]
    atoms_spurious += sum(len(got[k]) for k in extra_keys)

    return AtlasScore(
        entries_total=len(keys),
        entries_correct=entries_correct,
        entries_missing=entries_missing,
        entries_malformed=entries_malformed,
        atoms_expected=atoms_expected,
        atoms_correct=atoms_correct,
        atoms_spurious=atoms_spurious,
        exact=entries_correct == len(keys) and not extra_keys,
    )
