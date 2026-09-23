"""Pack data model, shared-argument helpers, and entry-count sampling."""

from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from sft.json_types import JsonLikeDict
from sft.scripts.builders.build_board_fluency_review import QueryDict

SCHEMA = "catan_board_pack/v1"

PACK_INSTRUCTIONS = (
    "Answer every numbered query about the board above. "
    "Output one line per query as NUMBER: ANSWER, in the order asked. No explanation."
)

# Query-dict fields that name a board location, used to detect shared arguments.
_SHARED_KEYS = ("node", "start", "end", "edge", "remove_edge", "from_tile", "to_tile")


def _share_signature(query: QueryDict) -> frozenset[str]:
    """Board locations a query touches, for opportunistic co-selection."""
    return frozenset(
        str(value) for key, value in query.items() if key in _SHARED_KEYS
    )


def render_query(operation: str, query: QueryDict) -> str:
    """Compact, generic argument line.

    Derived straight from the query dict that `answer` consumes, so a new
    operation or parameter cannot drift out of sync with the gold computation.
    """
    args = " ".join(f"{k}={query[k]}" for k in sorted(query) if k != "operation")
    return f"{operation} {args}".strip()


@dataclass
class PackEntry:
    index: int
    operation: str
    query: QueryDict
    expected: str
    metadata: JsonLikeDict = field(default_factory=dict)


@dataclass
class Pack:
    board: str
    entries: List[PackEntry]

    @property
    def operations(self) -> List[str]:
        return list(OrderedDict.fromkeys(e.operation for e in self.entries))

    def shared_argument_entries(self) -> int:
        """Entries whose board locations are also touched by another entry."""
        seen: Dict[str, int] = defaultdict(int)
        for entry in self.entries:
            for location in _share_signature(entry.query):
                seen[location] += 1
        return sum(
            1
            for entry in self.entries
            if any(seen[loc] > 1 for loc in _share_signature(entry.query))
        )


def sample_entry_count(rng: random.Random, available: int, max_entries: int) -> int:
    """Log-uniform over octaves, so small packs stay common as max_entries grows."""
    hi = max(1, min(max_entries, available))
    if hi == 1:
        return 1
    octave = rng.randint(0, hi.bit_length() - 1)
    return rng.randint(1 << octave, min(hi, (1 << (octave + 1)) - 1))
