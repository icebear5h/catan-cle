"""Exercise the split pair builder with real fixture pixels and spawned workers."""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from data_pipeline.board_recognition import adjacent_pair_localization as pairs
from data_pipeline.board_recognition.replay_dataset import load_render_style, read_jsonl

from .reference import ROOT, baseline
from .witnesses import FIXTURE


def image_bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.glob("*.png")}


def test_pair_rendering_matches_original_serial_and_spawned(tmp_path: Path) -> None:
    state: Any = next(row for row in read_jsonl(FIXTURE / "manifest.jsonl") if row["sample_id"] == "empty_setup_node_p000_base")
    state = {**state, "image_size": [256, 256]}
    contract = json.loads((FIXTURE / state["contract_path"]).read_text())
    style = load_render_style(ROOT / "configs/sft/renderer_style.json")
    original_dir, current_dir, spawned_dir = [tmp_path / name for name in ("original", "current", "spawned")]
    for directory in (original_dir, current_dir, spawned_dir):
        directory.mkdir()
    original = baseline("adjacent_pair_localization")
    expected_rows = original.build_pair_board(
        state=state, contract=contract, style=style, output_images=original_dir,
        images_per_kind={kind: 2 for kind in pairs.PAIR_KINDS}, colors=pairs.COLORS,
        tile_rows=True,
    )
    current_rows = pairs.build_pair_board(
        state=state, contract=contract, style=style, output_images=current_dir,
        images_per_kind={kind: 2 for kind in pairs.PAIR_KINDS}, colors=pairs.COLORS,
        tile_rows=True,
    )
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        spawned_rows = pairs.build_pair_board(
            state=state, contract=contract, style=style, output_images=spawned_dir,
            images_per_kind={kind: 2 for kind in pairs.PAIR_KINDS}, colors=pairs.COLORS,
            tile_rows=True, pool=pool,
        )
    assert current_rows == spawned_rows == expected_rows
    expected_images = image_bytes(original_dir)
    assert len(expected_images) == 16
    assert image_bytes(current_dir) == image_bytes(spawned_dir) == expected_images
