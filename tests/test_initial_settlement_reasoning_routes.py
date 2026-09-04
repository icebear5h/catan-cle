from __future__ import annotations

import json
from pathlib import Path

from flask import Flask

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.game_engine.public_board import snapshot_public_board
from playground.game_viewer.routes import bench
from playground.game_viewer.routes.bench import bench_bp


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _trace_artifacts(root: Path) -> None:
    models = ["model/a", "model/b"]
    colors = (Color.BLUE, Color.RED, Color.WHITE, Color.ORANGE)
    engine = GameEngine(colors, seed=117, shuffle_players=False)
    board_sha256 = snapshot_public_board(engine.observe(Color.BLUE)).facts_sha256
    _write_json(
        root / "plan.json",
        {
            "models": models,
            "seeds": [117, 883],
            "decision": "fresh first settlement",
            "actor": "BLUE",
            "colors": ["BLUE", "RED", "WHITE", "ORANGE"],
            "context_suite": "catan-agent@7.0.0",
            "board_surface": "indexed_tile_rows/v3",
            "reasoning_request": {"effort": "high", "exclude": False},
            "temperature": 0.2,
            "max_tokens_omitted": True,
            "replay_input": False,
            "human_action_labels": False,
            "scheduling": "sequential",
        },
    )
    row = {
        "trace_id": "seed-117:model/a",
        "seed": 117,
        "model_id": "model/a",
        "prompt_sha256": "a" * 64,
        "finish_reason": "stop",
        "native_reasoning_returned": True,
        "reasoning_tokens": 123,
        "final_response_characters": 42,
        "parsed_action_index": 7,
        "parse_error": None,
        "latency_ms": 1500,
        "cost_usd": 0.012,
        "json_path": "traces/seed_117/model_a.json",
    }
    _write_json(
        root / "summary.json",
        {
            "complete": False,
            "trace_count": 1,
            "recorded_cost_usd": 0.012,
            "traces": [row],
        },
    )
    _write_json(
        root / row["json_path"],
        {
            "schema": "catan-initial-settlement-reasoning-trace/v1",
            "trace_id": row["trace_id"],
            "input": {
                "seed": 117,
                "colors": [color.value for color in colors],
                "actor": "BLUE",
                "prompt_sha256": "a" * 64,
                "board_presentation": {"board_sha256": board_sha256},
                "messages": [{"role": "user", "content": "exact prompt"}],
            },
            "request": {
                "requested_model": "model/a",
                "max_tokens_omitted": True,
            },
            "response": {
                "native_reasoning": "native trace",
                "final_response": "<action>7</action>",
                "finish_reason": "stop",
                "usage": {"cost": 0.012},
            },
            "parse": {"error": None, "choice": {"action_index": 7}},
        },
    )


def _client(root: Path, monkeypatch):
    _trace_artifacts(root)
    monkeypatch.setattr(
        bench,
        "INITIAL_SETTLEMENT_REASONING_RUNS",
        {
            "fixture-v1": {
                "title": "Fixture v1",
                "description": "Fixture reasoning traces.",
                "path": root,
                "excluded_models": {"model/dq": "DQ evidence"},
            }
        },
    )
    monkeypatch.setattr(
        bench,
        "DEFAULT_INITIAL_SETTLEMENT_REASONING_RUN",
        "fixture-v1",
    )
    app = Flask(__name__)
    app.register_blueprint(bench_bp)
    return app.test_client()


def test_reasoning_trace_overview_represents_partial_and_unstarted_seeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    response = _client(tmp_path / "run", monkeypatch).get(
        "/api/catan-board-bench/reasoning-traces"
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["run_id"] == "fixture-v1"
    assert payload["run_title"] == "Fixture v1"
    assert payload["available_runs"] == [
        {
            "id": "fixture-v1",
            "title": "Fixture v1",
            "description": "Fixture reasoning traces.",
            "available": True,
            "excluded_models": {"model/dq": "DQ evidence"},
        }
    ]
    assert payload["excluded_models"] == {"model/dq": "DQ evidence"}
    assert payload["captured_traces"] == 1
    assert payload["planned_traces"] == 4
    assert payload["recorded_cost_usd"] == 0.012
    assert payload["conditions"]["max_tokens_omitted"] is True
    assert payload["conditions"]["replay_input"] is False
    assert payload["conditions"]["human_action_labels"] is False
    assert payload["seeds"] == [
        {
            "seed": 117,
            "status": "partial",
            "captured": 1,
            "expected": 2,
            "missing_models": ["model/b"],
        },
        {
            "seed": 883,
            "status": "not_started",
            "captured": 0,
            "expected": 2,
            "missing_models": ["model/a", "model/b"],
        },
    ]


def test_reasoning_trace_detail_keeps_native_reasoning_and_final_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path / "run", monkeypatch)
    response = client.get(
        "/api/catan-board-bench/reasoning-trace",
        query_string={"seed": 117, "model_id": "model/a"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["run_id"] == "fixture-v1"
    assert payload["response"]["native_reasoning"] == "native trace"
    assert payload["response"]["final_response"] == "<action>7</action>"
    assert payload["request"]["max_tokens_omitted"] is True
    assert payload["render_state"]["tiles"]
    assert len(payload["render_state"]["tiles"]) == 28
    assert payload["render_state_provenance"] == {
        "seed": 117,
        "board_sha256": payload["input"]["board_presentation"]["board_sha256"],
        "verified_against_trace": True,
        "renderer": "existing HexBoard contract",
    }
    assert "rationale" not in json.dumps(payload).lower()

    missing = client.get(
        "/api/catan-board-bench/reasoning-trace",
        query_string={"seed": 117, "model_id": "model/b"},
    )
    assert missing.status_code == 404
    assert missing.get_json()["error"] == "Reasoning trace has not been captured"


def test_reasoning_trace_routes_reject_unknown_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path / "run", monkeypatch)

    overview = client.get(
        "/api/catan-board-bench/reasoning-traces",
        query_string={"run_id": "missing"},
    )
    detail = client.get(
        "/api/catan-board-bench/reasoning-trace",
        query_string={
            "run_id": "missing",
            "seed": 117,
            "model_id": "model/a",
        },
    )

    assert overview.status_code == 404
    assert detail.status_code == 404
    assert overview.get_json()["error"] == "Unknown reasoning-trace run: missing"


def test_reasoning_trace_detail_rejects_board_sha_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "run"
    client = _client(root, monkeypatch)
    trace_path = root / "traces/seed_117/model_a.json"
    trace = json.loads(trace_path.read_text())
    trace["input"]["board_presentation"]["board_sha256"] = "0" * 64
    _write_json(trace_path, trace)

    response = client.get(
        "/api/catan-board-bench/reasoning-trace",
        query_string={"seed": 117, "model_id": "model/a"},
    )

    assert response.status_code == 409
    assert "does not match" in response.get_json()["error"]


def test_reasoning_trace_detail_rejects_indexed_path_escape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "run"
    client = _client(root, monkeypatch)
    summary = json.loads((root / "summary.json").read_text())
    summary["traces"][0]["json_path"] = "../../secret.json"
    _write_json(root / "summary.json", summary)

    response = client.get(
        "/api/catan-board-bench/reasoning-trace",
        query_string={"seed": 117, "model_id": "model/a"},
    )

    assert response.status_code == 400
    assert "outside the run directory" in response.get_json()["error"]
