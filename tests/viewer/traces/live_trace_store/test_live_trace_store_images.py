"""Image metadata persists without image bytes, including legacy namespaces."""
import asyncio
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import (
    ModelResponse,
    default_suite_path,
    load_context_suite,
)
from cle.harness.catan_board_surface import ImageBoardPresenter
from cle.harness.communication import default_communication_suite_path, load_communication_suite
from cle.players.agent import AgentPlayer
from cle.players.baseline import FirstLegalPlayer
from cle.sandbox import CatanSandbox
from cle.traces import SQLiteLiveTraceStore
from cle.traces.sqlite import unpack_blob

from .support import COLORS, SequenceTransport


@pytest.mark.filterwarnings("ignore:Prompt suite .* is deprecated:DeprecationWarning")
def test_trace_store_persists_image_metadata_without_image_bytes(tmp_path: Path) -> None:
    response = ModelResponse(
        content=(
            "<game_plan>expand</game_plan>"
            "<action>0</action>"
        ),
        model="vision/model",
        provider_request_payload={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,RAW_IMAGE_BYTES"
                            },
                        }
                    ],
                }
            ]
        },
    )
    transport: Any = SequenceTransport([response])
    engine = GameEngine(COLORS, seed=8, shuffle_players=False)
    red = AgentPlayer(
        Color.RED,
        transport,
        session_id=f"{engine.id}:RED",
        suite=load_context_suite(default_suite_path().with_name("catan_v10.yaml")),
        communication_suite=load_communication_suite(default_communication_suite_path()),
        board_presenter=ImageBoardPresenter(image_size=512),
    )
    players: Any = {Color.RED: red}
    players.update({color: FirstLegalPlayer(color) for color in COLORS[1:]})
    sandbox = CatanSandbox(engine, players)
    store: Any = SQLiteLiveTraceStore(tmp_path / "image-traces.sqlite3")
    game_id: Any = str(engine.id)
    store.start_game(
        game_id,
        config={"mode": "llm_vs_random", "board_surface": "image"},
        snapshot=sandbox.snapshot(),
    )

    result = asyncio.run(sandbox.step())
    store.record_step(
        game_id,
        result=result,
        rejected_attempts=(),
        public_state={},
        snapshot=sandbox.snapshot(),
    )

    call: Any = store.get_game(game_id)["model_calls"][0]
    board: Any = call["request"]["board_presentation"]
    assert board["kind"] == "image"
    assert board["media_type"] == "image/png"
    assert board["data"] is None
    assert board["byte_length"] > 0
    persisted_url: Any = call["response"]["provider_request_payload"]["messages"][0][
        "content"
    ][0]["image_url"]["url"]
    assert persisted_url == f"local-board-image://sha256/{board['content_sha256']}"

    with sqlite3.connect(store.path) as connection:
        serialized: Any = "\n".join(
            unpack_blob(value).decode("utf-8")
            for row in connection.execute(
                "SELECT request_json, response_json FROM model_calls"
            )
            for value in row
            if value
        )
    assert "data:image" not in serialized
    assert "RAW_IMAGE_BYTES" not in serialized
    assert transport.requests[0].board_presentation.data.hex() not in serialized


def test_trace_store_loads_snapshots_from_legacy_engine_namespace(tmp_path: Path) -> None:
    engine = GameEngine(COLORS, seed=4, shuffle_players=False)
    sandbox = CatanSandbox(
        engine,
        {color: FirstLegalPlayer(color) for color in COLORS},
    )
    snapshot = sandbox.snapshot()
    payload = pickle.dumps(snapshot, protocol=0).replace(
        b"ccle.game_engine",
        b"cgame_engine",
    )
    assert b"cgame_engine" in payload

    store = SQLiteLiveTraceStore(tmp_path / "legacy.sqlite3")
    game_id = str(engine.id)
    store.start_game(
        game_id,
        config={"mode": "random", "seed": 4},
        snapshot=snapshot,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE live_games SET initial_snapshot = ? WHERE game_id = ?",
            (payload, game_id),
        )

    restored = store.load_resume_point(game_id).snapshot
    assert restored.engine.state.colors == snapshot.engine.state.colors
    assert restored.engine.state.rng.getstate() == snapshot.engine.state.rng.getstate()
