"""Shared artifact builders and loaders for paired model-trace tests."""
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from playground.game_viewer.replay.model_traces import (
    load_model_trace_artifact,
)


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _artifact(tmp_path: Path) -> Path:
    plan = {
        "game_id": "game-1",
        "models": ["qwen/test"],
        "target_player_id": 2,
        "target_engine_color": "BLUE",
        "exact_decision_count": 2,
        "canonicalizations": [
            {
                "canonical_indices": [35, 36],
                "source_replay_indices": [35, 36],
            }
        ],
        "settings": {
            "context_version": "replay-decision-v2",
            "stateless_goals": True,
            "allow_lookahead": False,
            "execute_model_actions": False,
        },
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    _write_jsonl(
        tmp_path / "decision_manifest.jsonl",
        [
            {
                "decision_id": "game-1:35",
                "replay_index": 35,
                "source_replay_index": 36,
                "classification": "exact",
                "effective_action_type": "MOVE_ROBBER",
                "phase": "main_game",
                "forced": False,
                "actor": {
                    "colonist_player": 2,
                    "engine_index": 0,
                    "engine_color": "BLUE",
                },
                "available_actions": [{"index": 0}, {"index": 1}],
            },
            {
                "decision_id": "game-1:36",
                "replay_index": 36,
                "source_replay_index": 35,
                "classification": "exact",
                "effective_action_type": "STEAL",
                "phase": "main_game",
                "forced": True,
                "actor": {
                    "colonist_player": 2,
                    "engine_index": 0,
                    "engine_color": "BLUE",
                },
                "available_actions": [{"index": 0}],
            },
            {
                "decision_id": "game-1:40",
                "replay_index": 40,
                "source_replay_index": 40,
                "classification": "coarse",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "responses.jsonl",
        [
            {
                "decision_id": "game-1:35",
                "game_id": "game-1",
                "replay_index": 35,
                "model_id": "qwen/test",
                "recorded_at": "2026-08-17T00:00:00+00:00",
                "error": None,
                "result": {
                    "model": "qwen/test-provider",
                    "player_color": "BLUE",
                    "context_version": "replay-decision-v2",
                    "action_index": None,
                    "action": None,
                    "action_description": None,
                    "goals": "Build toward ore.",
                    "reasoning": "Choose a robber destination from the current board.",
                    "message": "",
                    "parse_error": "original parser warning",
                    "response_truncated": False,
                    "latency_ms": 1234,
                },
            },
            {
                "decision_id": "game-1:36",
                "game_id": "game-1",
                "replay_index": 36,
                "model_id": "qwen/test",
                "recorded_at": "2026-08-17T00:00:01+00:00",
                "error": None,
                "result": {
                    "model": "qwen/test-provider",
                    "player_color": "BLUE",
                    "context_version": "replay-decision-v2",
                    "action_index": 0,
                    "action": "Action(steal-red)",
                    "action_description": "Steal from RED",
                    "goals": "Build toward ore.",
                    "reasoning": "BLUE moved the robber to tile 12; now steal from RED.",
                    "message": "",
                    "parse_error": None,
                    "response_truncated": False,
                    "latency_ms": 900,
                },
            },
        ],
    )
    _write_jsonl(
        tmp_path / "comparisons.jsonl",
        [
            {
                "decision_id": "game-1:35",
                "human": {
                    "action_index": 1,
                    "description": "Upcoming human action must stay private",
                },
                "models": {
                    "qwen/test": {
                        "response_present": True,
                        "action_index": 0,
                        "action": "Action(model-selected)",
                        "description": "Move robber to the ore hex",
                        "agreement": False,
                        "parse_error": "recovered named action",
                    }
                },
            },
            {
                "decision_id": "game-1:36",
                "human": {
                    "action_index": 0,
                    "description": "Steal from the recorded victim",
                },
                "models": {
                    "qwen/test": {
                        "response_present": True,
                        "action_index": 0,
                        "action": "Action(steal-red)",
                        "description": "Steal from RED",
                        "agreement": True,
                        "parse_error": None,
                    }
                },
            },
        ],
    )
    return tmp_path


def _load_artifact(path: Path, narrator_color: int = 2) -> dict[str, Any]:
    return load_model_trace_artifact(
        path,
        expected_game_id="game-1",
        expected_model_id="qwen/test",
        expected_player_id=2,
        expected_engine_color="BLUE",
        model_label="Qwen test",
        narrator={"username": "Narrator", "colonist_color": narrator_color},
        archived_player_perspective=5,
    )


CURATED_TRACE_DIR = Path(
    "data_pipeline/training/reasoning/pilots/_2n5F2DxtPI/action_selection_diff/"
    "qwen3_8_27b_blue_20260817"
)


def _load_curated_override(
    override_path: Path, attempts_path: Path
) -> dict[str, Any]:
    return load_model_trace_artifact(
        CURATED_TRACE_DIR,
        expected_game_id="242781000",
        expected_model_id="qwen/qwen3.8-27b",
        expected_player_id=2,
        expected_engine_color="BLUE",
        model_label="Qwen 3.8 27B",
        narrator={"username": "FunDipDevRip", "colonist_color": 2},
        archived_player_perspective=5,
        response_overrides_path=override_path,
        response_attempts_path=attempts_path,
    )
