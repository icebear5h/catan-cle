"""Shared fixtures for spatial continuation dataset fixtures, corpus, and launcher contracts."""

import json
from pathlib import Path

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.board import Board
from cle.game_engine.models.player import Color
from data_pipeline.board_recognition.replay_dataset import read_jsonl
from evals.catan_board_bench.builder import CatanObservationSuite
from sft.scripts.builders.build_spatial_continuation_dataset import (
    DEFAULT_ROOT,
    FROZEN_BOARD_EVAL,
    build_dataset,
)

from .support import DatasetInputs, JsonDict


@pytest.fixture
def engine_board() -> tuple[Board, JsonDict]:
    game = GameEngine(
        [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE], seed=45, shuffle_players=False
    )
    board = game.state.board
    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_city(Color.RED, 0)
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_settlement(Color.BLUE, 8, initial_build_phase=True)
    board.build_city(Color.BLUE, 8)
    board.robber_coordinate = next(
        coord
        for coord, tile in board.map.land_tiles.items()
        if tile.resource is not None and 0 in tile.nodes.values()
    )
    contract = CatanObservationSuite().public_board_contract(
        game, sample={"id": "spatial_fixture", "index": 0}, source={"kind": "unit_test"}
    )
    return board, contract


@pytest.fixture(scope="module")
def local_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, DatasetInputs, JsonDict, list[JsonDict], dict[str, list[JsonDict]]]:
    required = [
        DEFAULT_ROOT / "manifest.jsonl",
        DEFAULT_ROOT / "contracts",
        DEFAULT_ROOT / "dense_labels",
        DEFAULT_ROOT / "full_board_readout_v1/images",
        DEFAULT_ROOT / "full_board_readout_v1/trainable_tokens.json",
        FROZEN_BOARD_EVAL,
    ]
    required += [
        DEFAULT_ROOT / "full_board_readout_v1/stage1" / f"{s}.jsonl"
        for s in ("train", "validation", "test")
    ]
    if not all(path.exists() for path in required):
        pytest.skip("full local board corpus or frozen validation source is absent")
    parent = tmp_path_factory.mktemp("spatial_continuation")
    inputs = build_dataset(parent / "first")
    metadata = json.loads(Path(inputs["metadata"]).read_text())
    train = read_jsonl(Path(inputs["train_jsonl"]))
    panels = {
        label: read_jsonl(Path(spec["eval_jsonl"])) for label, spec in inputs["new_panels"].items()
    }
    return parent, inputs, metadata, train, panels
