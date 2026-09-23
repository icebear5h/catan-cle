"""Shared helpers for adjacent-pair board recognition sampling, rendering, and rows."""
import json
from pathlib import Path
from typing import Any

from data_pipeline.board_recognition.adjacent_pair_localization import (
    cross_touching,
)
from data_pipeline.board_recognition.single_piece_localization import (
    neighbor_tokens,
)

FIXTURE_ROOT = Path("artifacts/fixtures/board_recognition/curriculum_smoke")

Contract = dict[str, Any]


def _fixture_contract() -> tuple[Contract, Contract]:
    manifest = [
        json.loads(line)
        for line in (FIXTURE_ROOT / "manifest.jsonl").read_text().splitlines()
    ]
    state = next(row for row in manifest if row["sample_id"] == "empty_setup_node_p000_base")
    return state, json.loads((FIXTURE_ROOT / state["contract_path"]).read_text())


def _touch(contract: Contract, first: str, second: str) -> bool:
    neighbors = neighbor_tokens(contract)
    touching = cross_touching(contract)
    return second in neighbors[first] or second in touching[first]
