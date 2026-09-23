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

import functools as functools
import itertools as itertools
import json as json
import random as random
from collections import Counter as Counter
from dataclasses import dataclass as dataclass
from typing import Callable as Callable
from typing import Dict as Dict
from typing import Iterable as Iterable
from typing import Iterator as Iterator
from typing import List as List
from typing import Sequence as Sequence
from typing import Tuple as Tuple

import networkx as nx

from cle.game_engine.models.board import STATIC_GRAPH as STATIC_GRAPH
from cle.game_engine.models.board import base_map as base_map
from cle.game_engine.models.map import NUM_EDGES as NUM_EDGES
from cle.game_engine.models.map import NUM_NODES as NUM_NODES
from cle.game_engine.models.map import NUM_TILES as NUM_TILES
from cle.game_engine.models.map import PORT_DIRECTION_TO_NODEREFS as PORT_DIRECTION_TO_NODEREFS
from sft.board.board_atlas._coverage import answer_shape_report as answer_shape_report
from sft.board.board_atlas._coverage import assert_topology_invariants as assert_topology_invariants
from sft.board.board_atlas._coverage import coverage_report as coverage_report
from sft.board.board_atlas._coverage import write_jsonl as write_jsonl
from sft.board.board_atlas._examples import AtlasExample as AtlasExample
from sft.board.board_atlas._examples import generate_examples as generate_examples
from sft.board.board_atlas._examples import generate_exhaustive as generate_exhaustive
from sft.board.board_atlas._examples import is_empty_fact as is_empty_fact
from sft.board.board_atlas._examples import sample_cardinality as sample_cardinality
from sft.board.board_atlas._examples import weighted_sample as weighted_sample
from sft.board.board_atlas._render import render_answer as render_answer
from sft.board.board_atlas._render import render_entry as render_entry
from sft.board.board_atlas._render import render_example as render_example
from sft.board.board_atlas._scoring import AtlasScore as AtlasScore
from sft.board.board_atlas._scoring import parse_answer as parse_answer
from sft.board.board_atlas._scoring import score_atlas as score_atlas
from sft.board.board_atlas._tables import FactTable as FactTable
from sft.board.board_atlas._tables import _edge_endpoints as _edge_endpoints
from sft.board.board_atlas._tables import _node_distance as _node_distance
from sft.board.board_atlas._tables import _node_edges as _node_edges
from sft.board.board_atlas._tables import _node_neighbors as _node_neighbors
from sft.board.board_atlas._tables import _node_path as _node_path
from sft.board.board_atlas._tables import _node_port as _node_port
from sft.board.board_atlas._tables import _node_step as _node_step
from sft.board.board_atlas._tables import _node_tiles as _node_tiles
from sft.board.board_atlas._tables import _pair_key as _pair_key
from sft.board.board_atlas._tables import _port_nodes as _port_nodes
from sft.board.board_atlas._tables import _tile_neighbors as _tile_neighbors
from sft.board.board_atlas._tables import _tile_nodes as _tile_nodes
from sft.board.board_atlas._tables import build_fact_tables as build_fact_tables
from sft.board.board_atlas._tokens import LAND_GRAPH as LAND_GRAPH
from sft.board.board_atlas._tokens import LAND_NODES as LAND_NODES
from sft.board.board_atlas._tokens import NONE_ANSWER as NONE_ANSWER
from sft.board.board_atlas._tokens import SCHEMA as SCHEMA
from sft.board.board_atlas._tokens import _positions as _positions
from sft.board.board_atlas._tokens import edge_token as edge_token
from sft.board.board_atlas._tokens import node_token as node_token
from sft.board.board_atlas._tokens import port_token as port_token
from sft.board.board_atlas._tokens import tile_token as tile_token
from sft.board.symbolic_board_tasks import OFFSETS as OFFSETS
from sft.board.symbolic_board_tasks import _atlas as symbolic_atlas

__all__: list[str] = [
    "AtlasExample",
    "AtlasScore",
    "Callable",
    "Counter",
    "Dict",
    "FactTable",
    "Iterable",
    "Iterator",
    "LAND_GRAPH",
    "LAND_NODES",
    "List",
    "NONE_ANSWER",
    "NUM_EDGES",
    "NUM_NODES",
    "NUM_TILES",
    "OFFSETS",
    "PORT_DIRECTION_TO_NODEREFS",
    "SCHEMA",
    "STATIC_GRAPH",
    "Sequence",
    "Tuple",
    "annotations",
    "answer_shape_report",
    "assert_topology_invariants",
    "base_map",
    "build_fact_tables",
    "coverage_report",
    "dataclass",
    "edge_token",
    "functools",
    "generate_examples",
    "generate_exhaustive",
    "is_empty_fact",
    "itertools",
    "json",
    "node_token",
    "nx",
    "parse_answer",
    "port_token",
    "random",
    "render_answer",
    "render_entry",
    "render_example",
    "sample_cardinality",
    "score_atlas",
    "symbolic_atlas",
    "tile_token",
    "weighted_sample",
    "write_jsonl",
]
