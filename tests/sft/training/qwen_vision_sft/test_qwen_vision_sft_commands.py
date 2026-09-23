"""Launch command shapes and profile gating."""

import pytest

from sft.qwen_series_vision_sft import (
    DEFAULT_27B_MODEL_ID,
    H200_PROFILE,
    L40S_PROFILE,
    VISION_LANGUAGE_LORA,
    VISION_ONLY,
    VisionSftConfig,
    build_vision_sft_command,
    default_run_name,
    validate_hardware_profile,
)

from .support import _command, _option


def test_joint_vision_sft_command_matches_full_tower_pattern() -> None:
    command = _command(VISION_LANGUAGE_LORA)

    assert _option(command, "--bits") == "16"
    assert _option(command, "--freeze_llm") == "True"
    assert _option(command, "--freeze_vision_tower") == "False"
    assert _option(command, "--freeze_merger") == "False"
    assert _option(command, "--enable_reasoning") == "False"
    assert _option(command, "--vision_lr") == "1e-06"
    assert _option(command, "--merger_lr") == "1e-05"
    assert _option(command, "--lora_enable") == "True"
    assert "--catan_token_adapter_only" not in command


def test_vision_only_uses_adapter_lifecycle_without_language_lora() -> None:
    config = VisionSftConfig(profile=VISION_ONLY)
    command = build_vision_sft_command(
        config,
        train_json="/data/train.json",
        image_folder="/data/images",
        output_dir="/runs/output",
    )

    assert "--catan_token_adapter_only" in command
    assert _option(command, "--lora_enable") == "True"
    assert config.language_lora is False
    assert config.as_manifest_dict()["freeze_llm"] is True


def test_full_epoch_command_omits_max_steps() -> None:
    config = VisionSftConfig(max_steps=None, num_train_epochs=2)
    command = build_vision_sft_command(
        config,
        train_json="train.json",
        image_folder="images",
        output_dir="output",
    )

    assert "--max_steps" not in command
    assert _option(command, "--num_train_epochs") == "2"
    assert default_run_name(config).endswith("epochs-2")


def test_invalid_profile_and_known_undersized_hardware_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported vision SFT profile"):
        VisionSftConfig(profile="unknown")
    with pytest.raises(ValueError, match="not admitted on one L40S"):
        validate_hardware_profile(DEFAULT_27B_MODEL_ID, L40S_PROFILE)

    validate_hardware_profile(DEFAULT_27B_MODEL_ID, H200_PROFILE)


def test_vision_command_carries_remote_token_inventory() -> None:
    command = build_vision_sft_command(
        VisionSftConfig(profile=VISION_ONLY),
        train_json="/data/train.json",
        image_folder="/data/images",
        output_dir="/runs/output",
        token_inventory="/data/trainable_tokens.json",
    )

    assert _option(command, "--catan_token_inventory") == ("/data/trainable_tokens.json")
