"""Approved config, budgets, and mixture contracts."""

import copy
from dataclasses import asdict

import pytest

from sft.launchers.spatial import modal_spatial_continuation as launcher

from .support import row, training_rows


def test_approved_config_budgets_and_new_output(plan: dict[str, object]) -> None:
    config = launcher.validate_config(plan["config"], "test-run")
    assert config.max_steps == 128
    assert config.per_device_train_batch_size * config.gradient_accumulation_steps == 8
    assert config.initial_bundle == launcher.PARENT_CHECKPOINT
    assert config.output_dir not in launcher.PARENT_CHECKPOINT
    assert config.resume_from_checkpoint is None and config.token_init == "keep"
    assert not config.publish_to_hub and config.save_steps == config.eval_steps == 32
    options = launcher.CONTINUATION_GPU_OPTIONS
    assert options["gpu"] == "H200" and options["cpu"] == (16.0, 16.0)
    assert options["memory"] == (131072, 131072)
    assert options["retries"] == 0 and options["max_containers"] == 1
    assert options["startup_timeout"] == 300 and options["timeout"] == 3600
    total_wait = launcher.PREFLIGHT_TIMEOUT + 3 * launcher.GPU_TIMEOUT + 4 * (launcher.STARTUP_TIMEOUT + launcher.WAIT_GRACE)
    assert launcher.COORDINATOR_TIMEOUT > total_wait
    for label, (_, batch, budget) in launcher.PANEL_BUDGETS.items():
        args = launcher.panel_args(plan, launcher.PARENT_CHECKPOINT, label)
        assert args.batch_size == args.long_batch_size == batch
        assert args.max_new_tokens == args.long_max_new_tokens == budget
        assert args.preserve_visual_fp32 and args.bits == 16
        assert not args.candidate_scoring and not args.enable_thinking and not args.do_sample


@pytest.mark.parametrize("change", [
    {"max_steps": 129}, {"max_steps": 127}, {"max_steps": 0}, {"max_steps": None},
    {"gradient_accumulation_steps": 4}, {"per_device_train_batch_size": 8},
    {"resume_from_checkpoint": "/old"}, {"token_init": "vocab_gaussian"},
    {"publish_to_hub": True}, {"lora_rank": 16}, {"vision_learning_rate": 1e-4},
    {"save_steps": 64}, {"eval_steps": 64}, {"save_total_limit": 8},
    {"output_dir": "/runs/parent"}, {"initial_bundle": "/older/checkpoint-128"},
    {"input_mode": "text"}, {"max_sequence_length": 8192},
])
def test_rejects_unapproved_training(
    plan: dict[str, object], change: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        launcher.validate_config({**plan["config"], **change}, "test-run")


def test_historical_missing_input_defaults_validate_without_rehashing(plan: dict[str, object]) -> None:
    legacy = {k: v for k, v in plan["config"].items()
              if k not in ("input_mode", "max_sequence_length")}
    original = copy.deepcopy(legacy)
    original_digest = launcher.digest(legacy)
    config = launcher.validate_config(legacy, "test-run")
    assert config.input_mode == "vision" and config.max_sequence_length is None
    launcher.verify_runtime({**plan, "config": legacy, "config_sha256": original_digest})
    assert legacy == original and launcher.digest(legacy) == original_digest
    assert launcher.digest(asdict(config)) != original_digest


def test_exact_mixture_owns_128_homogeneous_steps() -> None:
    rows = training_rows()
    result = launcher.validate_mixture(rows)
    assert result["steps"] == 128 and result["rows"] == 1024
    assert result["family_steps"] == launcher.FAMILY_STEPS
    assert result["family_step_share"]["full_board_readout"] == 0.25
    rows[0], rows[8] = rows[8], rows[0]
    with pytest.raises(ValueError, match="homogeneous"):
        launcher.validate_mixture(rows)


@pytest.mark.parametrize("problem", ["count", "ids", "quota", "task", "metadata", "curriculum"])
def test_bad_mixture_is_rejected(problem: str) -> None:
    rows = training_rows()
    if problem == "count":
        rows.pop()
    elif problem == "ids":
        rows[1]["row_id"] = rows[0]["row_id"]
    elif problem == "quota":
        for i in range(8):
            rows[i] = row(i, "full_board_readout")
    elif problem == "task":
        rows[0]["task_type"] = "full_board_readout"
    elif problem == "metadata":
        rows[0]["metadata"]["task_type"] = "other"
    else:
        rows[0]["curriculum_stage"] = None
    with pytest.raises(ValueError):
        launcher.validate_mixture(rows)
