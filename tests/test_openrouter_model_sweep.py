import json

import pytest

from scripts.run_catan_openrouter_model_sweep import (
    DEFAULT_CONFIG,
    board_command,
    model_key,
    reasoning_trace_command,
    response_cost_usd,
    strategy_reasoning_trace_command,
    update_status,
    validate_config,
)


def load_config():
    return json.loads(DEFAULT_CONFIG.read_text())


def fake_catalog(config):
    return {
        "data": [
            {
                "id": row["model_id"],
                "name": row["model_id"],
                "context_length": 262144,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["max_tokens", "reasoning"],
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                "top_provider": {"context_length": 262144},
            }
            for row in config["candidates"]
        ]
    }


def test_sweep_config_is_unique_multimodal_and_within_budget(tmp_path):
    config = load_config()
    selected = validate_config(config, fake_catalog(config))

    assert len(selected) == len(config["candidates"]) == 13
    assert len({row["model_id"] for row in selected}) == 13
    assert sum(config["budget_allocation_usd"].values()) == 10.0
    assert all("image" in row["architecture"]["input_modalities"] for row in selected)

    update_status(config, tmp_path, selected)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["cost_usd"]["total"] == 0
    assert status["placement_reasoning_candidate_count"] == 9
    assert status["strategy_reasoning_candidate_count"] == 7


def test_sweep_rejects_catalog_candidate_without_image_input():
    config = load_config()
    catalog = fake_catalog(config)
    catalog["data"][0]["architecture"]["input_modalities"] = ["text"]

    with pytest.raises(ValueError, match="image input"):
        validate_config(config, catalog)


def test_sweep_commands_lock_board_and_direct_reasoning_conditions(tmp_path):
    config = load_config()
    model_id = config["candidates"][0]["model_id"]
    board = board_command(config, tmp_path, model_id, full=False)
    placement_models = [
        row["model_id"]
        for row in config["candidates"]
        if row["initial_placement_reasoning"]
    ]
    reasoning = reasoning_trace_command(
        config,
        tmp_path,
        placement_models,
        preflight=True,
    )

    assert "--provider" in board and "openrouter" in board
    assert board[board.index("--max-tokens") + 1] == "256"
    assert board[board.index("--categories") + 1] == "node_state"
    assert reasoning[1] == "scripts/eval_catan_initial_settlement_reasoning.py"
    assert reasoning[reasoning.index("--reasoning-effort") + 1] == "high"
    assert reasoning[reasoning.index("--max-seeds") + 1] == "1"
    assert "--max-tokens" not in reasoning
    assert "--game-id" not in reasoning
    assert "diff_replay_actions.py" not in reasoning
    assert len(placement_models) == 9

    excluded = set(config["strategy_guided_reasoning"]["excluded_models"])
    strategy_models = [model for model in placement_models if model not in excluded]
    strategy = strategy_reasoning_trace_command(
        config,
        tmp_path,
        strategy_models,
    )
    assert "--prompt-variant" in strategy
    assert strategy[strategy.index("--prompt-variant") + 1].endswith(
        "catan_initial_settlement_strategy_v1.json"
    )
    assert "--max-tokens" not in strategy
    assert "--game-id" not in strategy
    assert len(strategy_models) == 7
    assert "qwen/qwen3.5-27b" not in strategy_models
    assert "qwen/qwen3.5-35b-a3b" not in strategy_models


def test_sweep_cost_parser_and_model_key_are_stable():
    assert model_key("Qwen/Qwen3.8-27B") == "qwen_qwen3_8_27b"
    assert response_cost_usd({"usage": {"cost": 0.25}}) == 0.25
    assert response_cost_usd({"result": {"usage": {"cost": 0.5}}}) == 0.5
    assert response_cost_usd({"response": {"usage": {"cost": 0.75}}}) == 0.75
    assert response_cost_usd({"result": None}) == 0.0
