"""Synthetic engine layouts, image linking, and terrain fact extraction."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from cle.game_engine.game import GameEngine
from cle.sandbox.palette import balanced_datagen_colors
from data_pipeline.board_recognition.replay_dataset import load_render_style
from data_pipeline.board_recognition.single_piece_localization import (
    TileFact,
    render_contract,
    tile_facts,
)
from data_pipeline.board_recognition.sources import validate_public_board_contract
from data_pipeline.board_recognition.spatial_localization import SpatialLocalizationError
from data_pipeline.board_recognition.terrain_impl._config import SYNTHETIC_IMAGE_SIZE
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.builder import CatanObservationSuite


class PortFact(TypedDict):
    """One port's token and its production ``port?`` answer."""

    token: str
    answer: str


def synthetic_seed(seed: int, split: str, index: int) -> int:
    digest = hashlib.sha256(f"terrain-synthetic:{seed}:{split}:{index}".encode()).hexdigest()
    return int(digest[:8], 16)


def synthetic_board(seed: int, split: str, index: int) -> tuple[JsonDict, JsonDict]:
    """A freshly randomised empty engine board as a manifest-like state plus its contract."""

    game_seed = synthetic_seed(seed, split, index)
    engine = GameEngine(balanced_datagen_colors(index, seed=seed), seed=game_seed, shuffle_players=False)
    sample_id = f"synth{game_seed:010d}_s000000"
    contract = CatanObservationSuite().public_board_contract(
        engine,
        sample={"id": sample_id, "index": 0},
        source={"kind": "synthetic_layout", "engine_seed": game_seed, "split": split},
    )
    validate_public_board_contract(contract)
    state: JsonDict = {
        "sample_id": sample_id,
        "split": split,
        "image_path": f"images/{split}_{sample_id}.png",
        "density_bin": "empty",
        "image_size": [SYNTHETIC_IMAGE_SIZE, SYNTHETIC_IMAGE_SIZE],
        "source": {"kind": "synthetic_layout", "engine_seed": game_seed},
    }
    return state, contract


def _render_synthetic(seed: int, split: str, index: int, destination: Path, style_path: Path, image_size: int) -> tuple[JsonDict, JsonDict]:
    state, contract = synthetic_board(seed, split, index)
    render_contract(contract, image_size, load_render_style(style_path), destination)
    state["image_size"] = [image_size, image_size]
    return state, contract


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def layout_id(sample_id: str) -> str:
    """The replay a state belongs to; every state of a replay shares one board layout."""

    head, sep, tail = sample_id.rpartition("_s")
    return head if sep and tail.isdigit() else sample_id


def port_answer(port: JsonDict) -> str:
    """Production ``port?`` answer: ``3:1 port`` or ``<resource> port``."""

    if port.get("kind") == "generic" or port.get("resource") is None:
        return "3:1 port"
    return f"{str(port['resource']).lower()} port"


def terrain_facts(contract: JsonDict) -> tuple[list[TileFact], list[PortFact]]:
    tiles = sorted(tile_facts(contract), key=lambda tile: tile["token"])
    port_facts: list[PortFact] = [
        {"token": as_str(as_dict(port)["token"]), "answer": port_answer(as_dict(port))}
        for port in as_list(contract["ports"])
    ]
    ports = sorted(port_facts, key=lambda port: port["token"])
    if len(tiles) != 19 or len(ports) != 9:
        raise SpatialLocalizationError(f"expected 19 tiles and 9 ports, got {len(tiles)} and {len(ports)}")
    return tiles, ports


def readout_answer(tiles: Sequence[TileFact], ports: Sequence[PortFact]) -> str:
    """Every tile and port in token order, one canonical string."""

    parts = [f"{tile['token']} {tile['resource']} {tile['number']}" for tile in tiles]
    parts += [f"{port['token']} {port['answer']}" for port in ports]
    return "; ".join(parts)
