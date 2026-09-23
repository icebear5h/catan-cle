"""Read-only pre-cleanup source oracle pinned to the inspected clean revision."""

import ast
import subprocess
from functools import lru_cache
from pathlib import Path
from types import ModuleType

REVISION = "2a27dbcba5a4ba3f2e63e5438a5c78bb93cf7a45"
ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "adjacent_pair_localization", "dataset", "density_curriculum",
    "inverse_grounding", "mix_rung_data", "production_curriculum", "query_schedule",
    "replay_ms_swift", "replay_sft", "reweight_node_edge", "sft", "spatial_localization",
)
# Root modules whose exact repository path an external SFT builder hashes. Each
# must stay a single file: build_symbolic_board_dataset/_sources.py,
# build_spatial_continuation_dataset/_sources.py and
# build_board_fluency_review/_build.py open these paths directly.
HASHED_SOURCES = (
    "full_board_readout", "node_edge_readout", "replay_dataset",
    "single_piece_localization", "sources", "spatial_robber", "terrain_readout",
)


@lru_cache
def source(name: str) -> str:
    return subprocess.run(
        ["git", "show", f"{REVISION}:data_pipeline/board_recognition/{name}.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout


def baseline(name: str) -> ModuleType:
    module = ModuleType(f"recognition_reference_{name}")
    module.__file__ = str(ROOT / "data_pipeline/board_recognition" / f"{name}.py")
    exec(compile(source(name), module.__file__, "exec"), module.__dict__)
    return module


def owned_names(name: str) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(source(name)).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return names
