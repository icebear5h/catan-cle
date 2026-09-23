"""Mechanistic geometry probe for Catan atlas parts (tile/node/edge/port)
using Qwen visual-language residuals.

Constants, row loading, atlas graphs, model IO, metrics, and the CLI live in
sibling modules. Every pre-split name stays importable at this path."""

from __future__ import annotations

from scripts.probes.probe_catan_board_mech_geometry.atlas import (
    build_atlas_graphs,
    find_subsequence_indices,
    token_index_maps,
)
from scripts.probes.probe_catan_board_mech_geometry.cli import main, parse_args
from scripts.probes.probe_catan_board_mech_geometry.constants import (
    CANONICAL_CATEGORY_MAP,
    PARTS,
    TOKEN_PATTERNS,
)
from scripts.probes.probe_catan_board_mech_geometry.metrics import (
    evaluate_identity_by_layer,
    evaluate_topology_by_layer,
)
from scripts.probes.probe_catan_board_mech_geometry.model_io import (
    choose_device,
    collect_hidden_vectors,
    load_model_and_processor,
    torch_dtype,
)
from scripts.probes.probe_catan_board_mech_geometry.rows import (
    ProbeRow,
    iter_jsonl,
    load_manifest,
    load_rows,
)

__all__ = [
    "CANONICAL_CATEGORY_MAP",
    "PARTS",
    "TOKEN_PATTERNS",
    "ProbeRow",
    "build_atlas_graphs",
    "choose_device",
    "collect_hidden_vectors",
    "evaluate_identity_by_layer",
    "evaluate_topology_by_layer",
    "find_subsequence_indices",
    "iter_jsonl",
    "load_manifest",
    "load_model_and_processor",
    "load_rows",
    "main",
    "parse_args",
    "token_index_maps",
    "torch_dtype",
]
