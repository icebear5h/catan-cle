"""Pack rendering and per-entry scoring through the board-fluency scorer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sft.board.board_fluency_scoring import (
    OPERATION_FAMILY,
    _parse_answer,
    _strip_transport,
    _typed_equal,
)
from sft.board.board_packs._model import PACK_INSTRUCTIONS, Pack, render_query
from sft.json_types import JsonLikeDict
from sft.scripts.builders.build_board_fluency_review import question as render_question


def render_pack(pack: Pack, *, shared_rubrics: bool = True) -> Tuple[str, str]:
    """Render prompt and target.

    With `shared_rubrics` the long per-operation rules (pip definitions, coverage
    definitions, component rules) are stated once for the whole pack instead of
    once per query, which is the second amortisation after the board itself. The
    rubric text is lifted verbatim from `question` so it cannot drift; it is the
    per-query phrasing that is replaced by a compact argument line.
    """
    lines = [pack.board, "", PACK_INSTRUCTIONS]

    if shared_rubrics:
        for operation in pack.operations:
            sample = next(e for e in pack.entries if e.operation == operation)
            lines.append(f"\n[{operation}] {render_question(operation, sample.query)}")
        lines.append("")
        for entry in pack.entries:
            lines.append(f"{entry.index}. {render_query(entry.operation, entry.query)}")
    else:
        lines.append("")
        for entry in pack.entries:
            lines.append(
                f"{entry.index}. {render_question(entry.operation, entry.query)}"
            )

    target = "\n".join(f"{e.index}: {e.expected}" for e in pack.entries)
    return "\n".join(lines), target


def parse_pack_answer(text: str) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for line in text.strip().splitlines():
        head, _, value = line.partition(":")
        head = head.strip()
        if head.isdigit():
            parsed[int(head)] = value.strip()
    return parsed


def entry_correct(operation: str, expected: str, response: str) -> bool:
    """Type-aware answer comparison for one entry.

    This deliberately does not route through `score_board_fluency`. That function
    first runs `validate_board_fluency_metadata`, which enforces dataset-admission
    declarations -- review_only, split, task_role, provenance -- and fails closed.
    Those checks exist to stop review-only rows leaking into training, and a pack
    would have to forge them just to compare two answers. Comparing an answer and
    validating a row's admission are separate concerns, so the comparison path is
    used directly: the same parse, transport-stripping and typed equality the real
    scorer applies, with no metadata claim attached.
    """
    if operation not in OPERATION_FAMILY:
        raise KeyError(f"unknown board-fluency operation: {operation}")
    gold = _parse_answer(expected, operation)
    try:
        return _typed_equal(gold, _parse_answer(_strip_transport(response), operation))
    except (ValueError, TypeError, RecursionError):
        return False  # Malformed predictions score wrong, never raise.


@dataclass
class PackScore:
    entries_total: int
    entries_correct: int
    entries_missing: int
    by_operation: Dict[str, Tuple[int, int]]

    @property
    def accuracy(self) -> float:
        return self.entries_correct / self.entries_total if self.entries_total else 0.0

    def to_dict(self) -> JsonLikeDict:
        return {
            "entries_total": self.entries_total,
            "entries_correct": self.entries_correct,
            "entries_missing": self.entries_missing,
            "accuracy": self.accuracy,
            "by_operation": {
                op: {"correct": c, "total": t} for op, (c, t) in self.by_operation.items()
            },
        }


def score_pack(pack: Pack, response: str) -> PackScore:
    """Per-entry scoring through the existing board-fluency scorer.

    Exact match over a whole pack would report 47 correct answers and one wrong as
    zero; each entry is scored independently and attributed to its operation so a
    pack stays as legible as the one-question-per-example format it replaces.
    """
    parsed = parse_pack_answer(response)
    by_operation: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    correct = missing = 0

    for entry in pack.entries:
        bucket = by_operation[entry.operation]
        bucket[1] += 1
        if entry.index not in parsed:
            missing += 1
            continue
        if entry_correct(entry.operation, entry.expected, parsed[entry.index]):
            correct += 1
            bucket[0] += 1

    return PackScore(
        entries_total=len(pack.entries),
        entries_correct=correct,
        entries_missing=missing,
        by_operation={op: (c, t) for op, (c, t) in by_operation.items()},
    )
