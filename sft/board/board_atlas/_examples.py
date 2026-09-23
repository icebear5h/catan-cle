"""Randomized and exhaustive sampling of queried fact subsets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

from sft.board.board_atlas._render import render_example
from sft.board.board_atlas._tables import FactTable
from sft.board.board_atlas._tokens import SCHEMA
from sft.json_types import JsonLikeDict


@dataclass(frozen=True)
class AtlasExample:
    schema: str
    table: str
    keys: Tuple[str, ...]
    prompt: str
    answer: str

    def to_row(self) -> JsonLikeDict:
        return {
            "schema": self.schema,
            "id": f"{self.schema}/{self.table}/{'_'.join(k.strip('<>') for k in self.keys)}",
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.answer},
            ],
            "metadata": {
                "schema": self.schema,
                "class": "board_atlas",
                "table": self.table,
                "keys": list(self.keys),
                "answer": self.answer,
                "cardinality": len(self.keys),
            },
        }


def is_empty_fact(table: FactTable, key: str) -> bool:
    """Whether this fact's answer renders as NONE."""
    return not table.facts[key]


def weighted_sample(
    keys: Sequence[str], k: int, weights: Sequence[float], rng: random.Random
) -> List[str]:
    """Weighted sampling without replacement (Efraimidis-Spirakis).

    `rng.sample` is uniform, and uniform sampling over `node_step` would make 56%
    of its examples NONE, since most nodes have two or three neighbours across six
    directions. Drawing key = u^(1/w) and taking the top k biases selection by
    weight while still never repeating a key inside one example.
    """
    if k >= len(keys):
        return list(keys)
    scored = []
    for key, weight in zip(keys, weights):
        if weight <= 0:
            continue
        scored.append((rng.random() ** (1.0 / weight), key))
    scored.sort(reverse=True)
    return [key for _, key in scored[:k]]


def sample_cardinality(rng: random.Random, n_keys: int, max_k: int) -> int:
    """Log-uniform over [1, min(max_k, n_keys)].

    Uniform-over-k would make k=1 examples vanishingly rare relative to the mass
    of large-k ones, and single-key queries are the form traversal consumes, so
    they are deliberately oversampled at the small end.
    """
    hi = max(1, min(max_k, n_keys))
    if hi == 1:
        return 1
    # Pick an octave uniformly, then a value within it: k=1 and k=2 stay common
    # however large `hi` grows.
    top_octave = (hi).bit_length() - 1
    octave = rng.randint(0, top_octave)
    low = 1 << octave
    high = min(hi, (1 << (octave + 1)) - 1)
    return rng.randint(low, high)


def generate_examples(
    tables: Dict[str, FactTable],
    *,
    n_examples: int,
    rng: random.Random,
    max_k: int = 24,
    table_weights: Dict[str, float] | None = None,
    none_weight: float = 0.2,
) -> Iterator[AtlasExample]:
    """Sample repetition on top of an exhaustive pass.

    `none_weight` scales how often empty-answer keys are drawn relative to
    non-empty ones. It defaults well below 1.0 because the r04 corpus was 35%
    NONE against 9% in its eval, and 10 of 72 review failures were "answered NONE
    when the answer was non-empty" -- a prior learned straight from the sampler.
    Coverage of the empty facts is guaranteed by `generate_exhaustive`, so
    down-weighting here reduces their repetition without losing any of them.
    """
    names = list(tables)
    weights = [(table_weights or {}).get(name, 1.0) for name in names]
    for _ in range(n_examples):
        table = tables[rng.choices(names, weights=weights, k=1)[0]]
        keys = list(table.facts)
        key_weights = [none_weight if is_empty_fact(table, k) else 1.0 for k in keys]
        k = sample_cardinality(rng, len(keys), max_k)
        chosen = weighted_sample(keys, k, key_weights, rng)
        rng.shuffle(chosen)
        prompt, answer = render_example(table, chosen)
        yield AtlasExample(SCHEMA, table.name, tuple(chosen), prompt, answer)


def generate_exhaustive(
    tables: Dict[str, FactTable],
    *,
    rng: random.Random,
    max_k: int = 24,
) -> Iterator[AtlasExample]:
    """Cover every fact at least once, packing keys into randomized batches."""
    for table in tables.values():
        keys = list(table.facts)
        rng.shuffle(keys)
        i = 0
        while i < len(keys):
            k = sample_cardinality(rng, len(keys) - i, max_k)
            chosen = keys[i : i + k]
            prompt, answer = render_example(table, chosen)
            yield AtlasExample(SCHEMA, table.name, tuple(chosen), prompt, answer)
            i += k
