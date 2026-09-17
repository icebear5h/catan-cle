"""Static board-topology atlas: exhaustive, closed-world training facts.

The Catan land graph never changes across games -- only resources, numbers,
buildings, roads and the robber do. `cle.game_engine.models.board.STATIC_GRAPH`
and `base_map` are the ground truth for that fixed topology, and everything here
is derived from them rather than restated.

The serialized board states used by the board-fluency corpus say what sits *on*
each element but never how elements *connect*: node->tile incidence, node->port
incidence and node->node adjacency appear nowhere in the prompt, and the atlas
tokens (<N39>, <E19_21>, ...) are atomic added tokens, so endpoints are not even
lexically recoverable from an edge token. The whole graph therefore has to live
in 154 learned embeddings. This module generates data that teaches it directly
instead of leaving it to leak out of downstream queries.

Two properties make this dataset unusual and are worth preserving in any change:

* It is a closed world. The fact set is finite and fully enumerable, so there is
  no held-out split -- `coverage_report` checks that every fact was emitted, and
  100% is both the target and exhaustively verifiable.
* Facts are queried, not recited. Examples draw a random subset of keys in random
  order (see `generate_examples`), because a model trained only on full ordered
  dumps learns the sequence rather than the facts and cannot answer about one
  node in isolation -- which is the form traversal actually consumes.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import networkx as nx  # type: ignore

from cle.game_engine.models.board import STATIC_GRAPH, base_map
from cle.game_engine.models.map import (
    NUM_EDGES,
    NUM_NODES,
    NUM_TILES,
    PORT_DIRECTION_TO_NODEREFS,
)

SCHEMA = "catan_board_atlas/v1"

LAND_NODES: Tuple[int, ...] = tuple(sorted(base_map.land_nodes))
LAND_GRAPH: nx.Graph = STATIC_GRAPH.subgraph(LAND_NODES).copy()

NONE_ANSWER = "NONE"


def node_token(node_id: int) -> str:
    return f"<N{node_id:02d}>"


def tile_token(tile_id: int) -> str:
    return f"<T{tile_id:02d}>"


def port_token(port_id: int) -> str:
    return f"<P{port_id:02d}>"


def edge_token(a: int, b: int) -> str:
    lo, hi = (a, b) if a < b else (b, a)
    return f"<E{lo:02d}_{hi:02d}>"


# --- fact tables -----------------------------------------------------------
#
# Every table maps a string key (already tokenized) to a list of answer atoms.
# An empty list renders as NONE. Distances render as a single numeric atom.


def _node_neighbors() -> Dict[str, List[str]]:
    return {
        node_token(n): [node_token(m) for m in sorted(LAND_GRAPH.neighbors(n))]
        for n in LAND_NODES
    }


def _node_edges() -> Dict[str, List[str]]:
    return {
        node_token(n): [edge_token(n, m) for m in sorted(LAND_GRAPH.neighbors(n))]
        for n in LAND_NODES
    }


def _edge_endpoints() -> Dict[str, List[str]]:
    return {
        edge_token(a, b): [node_token(x) for x in sorted((a, b))]
        for a, b in LAND_GRAPH.edges()
    }


def _node_tiles() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for n in LAND_NODES:
        tiles = base_map.adjacent_tiles.get(n, [])
        out[node_token(n)] = [tile_token(t.id) for t in sorted(tiles, key=lambda t: t.id)]
    return out


def _tile_nodes() -> Dict[str, List[str]]:
    return {
        tile_token(tile.id): [node_token(n) for n in sorted(tile.nodes.values())]
        for tile in base_map.tiles_by_id.values()
    }


def _tile_neighbors() -> Dict[str, List[str]]:
    """Land tiles sharing an edge, i.e. two or more corner nodes."""
    corners = {tile.id: set(tile.nodes.values()) for tile in base_map.tiles_by_id.values()}
    out: Dict[str, List[str]] = {}
    for tid, own in corners.items():
        adjacent = sorted(
            other for other, nodes in corners.items() if other != tid and len(own & nodes) >= 2
        )
        out[tile_token(tid)] = [tile_token(t) for t in adjacent]
    return out


def _port_nodes() -> Dict[str, List[str]]:
    """The two nodes a port actually grants access through.

    Not every land node of the port hex counts: the port sits on one edge of that
    hex, and `PORT_DIRECTION_TO_NODEREFS` names the two corners of that edge.
    Intersecting the hex's six node refs with the land set instead yields three
    nodes for several ports and silently overstates access.
    """
    out: Dict[str, List[str]] = {}
    for port in base_map.ports_by_id.values():
        refs = PORT_DIRECTION_TO_NODEREFS[port.direction]
        nodes = sorted(port.nodes[ref] for ref in refs)
        out[port_token(port.id)] = [node_token(n) for n in nodes]
    return out


def _node_port() -> Dict[str, List[str]]:
    owner: Dict[int, List[str]] = {n: [] for n in LAND_NODES}
    for token, nodes in _port_nodes().items():
        for node in nodes:
            owner[int(node[2:4])].append(token)
    return {node_token(n): sorted(set(v)) for n, v in owner.items()}


def _pair_key(a: int, b: int) -> str:
    lo, hi = (a, b) if a < b else (b, a)
    return f"{node_token(lo)} {node_token(hi)}"


def _node_distance() -> Dict[str, List[str]]:
    lengths = dict(nx.all_pairs_shortest_path_length(LAND_GRAPH))
    return {
        _pair_key(a, b): [str(lengths[a][b])]
        for a, b in itertools.combinations(LAND_NODES, 2)
    }


def _node_path() -> Dict[str, List[str]]:
    """One canonical shortest path per pair, endpoints included.

    Ties are broken by networkx's traversal order, which is deterministic for a
    fixed graph; the path is a *witness*, so any shortest path is a valid target
    as long as the same one is used consistently between training and scoring.
    """
    out: Dict[str, List[str]] = {}
    for a, b in itertools.combinations(LAND_NODES, 2):
        path = nx.shortest_path(LAND_GRAPH, a, b)
        out[_pair_key(a, b)] = [node_token(n) for n in path]
    return out


@dataclass(frozen=True)
class FactTable:
    name: str
    question: str
    answer_kind: str  # "set" | "scalar" | "sequence"
    facts: Dict[str, List[str]]

    def __len__(self) -> int:
        return len(self.facts)


def build_fact_tables() -> Dict[str, FactTable]:
    """Every static-topology fact, keyed by table name. Pure, no board state."""
    specs: Sequence[Tuple[str, str, str, Callable[[], Dict[str, List[str]]]]] = (
        (
            "node_neighbors",
            "For each listed node, give the nodes joined to it by one edge.",
            "set",
            _node_neighbors,
        ),
        (
            "node_edges",
            "For each listed node, give the edges incident to it.",
            "set",
            _node_edges,
        ),
        (
            "edge_endpoints",
            "For each listed edge, give its two endpoint nodes.",
            "set",
            _edge_endpoints,
        ),
        (
            "node_tiles",
            "For each listed node, give the land tiles touching it.",
            "set",
            _node_tiles,
        ),
        (
            "tile_nodes",
            "For each listed tile, give its six corner nodes.",
            "set",
            _tile_nodes,
        ),
        (
            "tile_neighbors",
            "For each listed tile, give the land tiles sharing an edge with it.",
            "set",
            _tile_neighbors,
        ),
        (
            "port_nodes",
            "For each listed port, give the nodes attached to it.",
            "set",
            _port_nodes,
        ),
        (
            "node_port",
            "For each listed node, give the port attached to it.",
            "set",
            _node_port,
        ),
        (
            "node_distance",
            "For each listed node pair, give the minimum number of edges between them.",
            "scalar",
            _node_distance,
        ),
        (
            "node_path",
            "For each listed node pair, give a shortest node path between them, endpoints included.",
            "sequence",
            _node_path,
        ),
    )
    return {name: FactTable(name, q, kind, fn()) for name, q, kind, fn in specs}


# --- rendering -------------------------------------------------------------


def render_answer(atoms: List[str]) -> str:
    return " ".join(atoms) if atoms else NONE_ANSWER


def render_entry(key: str, atoms: List[str]) -> str:
    return f"{key}: {render_answer(atoms)}"


def render_example(table: FactTable, keys: Sequence[str]) -> Tuple[str, str]:
    """Prompt lists the queried keys in the given order; answer mirrors it.

    Keys travel in the prompt so the model must content-address the fact rather
    than recite a fixed sequence, and so a k=1 example is the same format as a
    k=54 one.
    """
    suffix = (
        "Output one entry per line as KEY: VALUE, in the order asked. "
        "Use NONE for an empty value. No explanation."
    )
    prompt = f"{table.question} {' '.join(keys)}\n{suffix}"
    answer = "\n".join(render_entry(k, table.facts[k]) for k in keys)
    return prompt, answer


# --- example generation ----------------------------------------------------


@dataclass(frozen=True)
class AtlasExample:
    schema: str
    table: str
    keys: Tuple[str, ...]
    prompt: str
    answer: str

    def to_row(self) -> dict:
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
) -> Iterator[AtlasExample]:
    names = list(tables)
    weights = [(table_weights or {}).get(name, 1.0) for name in names]
    for _ in range(n_examples):
        table = tables[rng.choices(names, weights=weights, k=1)[0]]
        keys = list(table.facts)
        k = sample_cardinality(rng, len(keys), max_k)
        chosen = rng.sample(keys, k)
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

    def to_dict(self) -> dict:
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


# --- coverage --------------------------------------------------------------


def coverage_report(
    tables: Dict[str, FactTable], examples: Iterable[AtlasExample]
) -> dict:
    """Per-table and per-key emission counts. The atlas is a closed world, so
    uncovered keys are a generation bug, not a sampling outcome."""
    seen: Dict[str, Counter] = {name: Counter() for name in tables}
    for ex in examples:
        seen[ex.table].update(ex.keys)

    report: Dict[str, dict] = {}
    for name, table in tables.items():
        counts = seen[name]
        uncovered = sorted(k for k in table.facts if not counts[k])
        occurrences = [counts[k] for k in table.facts]
        report[name] = {
            "facts": len(table.facts),
            "covered": len(table.facts) - len(uncovered),
            "uncovered": uncovered,
            "min_occurrences": min(occurrences) if occurrences else 0,
            "max_occurrences": max(occurrences) if occurrences else 0,
        }
    total = sum(r["facts"] for r in report.values())
    covered = sum(r["covered"] for r in report.values())
    return {
        "schema": SCHEMA,
        "total_facts": total,
        "covered_facts": covered,
        "fully_covered": covered == total,
        "by_table": report,
    }


def assert_topology_invariants() -> None:
    """Guard the assumptions the corpus is built on."""
    if len(LAND_NODES) != NUM_NODES:
        raise AssertionError(f"expected {NUM_NODES} land nodes, got {len(LAND_NODES)}")
    if LAND_GRAPH.number_of_edges() != NUM_EDGES:
        raise AssertionError(
            f"expected {NUM_EDGES} land edges, got {LAND_GRAPH.number_of_edges()}"
        )
    if len(base_map.tiles_by_id) != NUM_TILES:
        raise AssertionError(
            f"expected {NUM_TILES} land tiles, got {len(base_map.tiles_by_id)}"
        )
    if not nx.is_connected(LAND_GRAPH):
        raise AssertionError("land graph is not connected; distance table would be partial")
    degrees = {d for _, d in LAND_GRAPH.degree()}
    if not degrees <= {2, 3}:
        raise AssertionError(f"unexpected node degrees on the land graph: {sorted(degrees)}")


def write_jsonl(examples: Iterable[AtlasExample], path: str) -> int:
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_row(), separators=(",", ":")) + "\n")
            written += 1
    return written
