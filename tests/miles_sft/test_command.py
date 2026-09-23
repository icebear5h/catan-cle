"""The smoke command must admit only the intended single-GPU supervised update."""

from pathlib import Path

import pytest

from sft.miles_sft.config import TrainPlan, miles_arguments
from sft.miles_sft.run import execute


def test_smoke_uses_fresh_language_lora_without_generation(tmp_path: Path) -> None:
    plan = TrainPlan(tmp_path / "base", tmp_path / "data", tmp_path / "run")
    command = miles_arguments(plan)
    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert value("--train-backend") == "megatron"
    assert value("--load") == value("--hf-checkpoint")
    assert value("--start-rollout-id") == "0"
    assert value("--actor-num-gpus-per-node") == "1"
    assert value("--rollout-num-gpus") == value("--eval-num-gpus") == "0"
    assert value("--loss-type") == "sft_loss"
    assert value("--lora-rank") == "16"
    assert value("--num-rollout") == "2"
    assert value("--save-hf").endswith("/rollout-{rollout_id}/bridge")
    assert "self_attention.in_proj" in value("--target-modules")
    assert "--sequence-parallel" not in command
    assert "--lora-adapter-path" not in command
    assert "--apply-chat-template" not in command
    report = execute(plan, Path("/miles"), Path("/megatron"), Path("/bridge"))
    assert report["status"] == "plan_only"
    assert not plan.output.exists()


@pytest.mark.parametrize("overrides", [
    {"steps": 0}, {"batch_size": 0}, {"learning_rate": float("nan")},
    {"max_tokens": 4097}, {"timeout_seconds": 1801},
])
def test_invalid_smoke_limits_fail(tmp_path: Path, overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainPlan(tmp_path / "base", tmp_path / "data", tmp_path / "run", **overrides)


def test_output_cannot_overwrite_its_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate directories"):
        TrainPlan(tmp_path / "base", tmp_path / "data", tmp_path / "base/run")
