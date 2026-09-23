"""args."""

from __future__ import annotations

import sys
from pathlib import Path

from sft.qwen_series_vision_sft import (
    VISION_ONLY,
    VISION_SFT_PROFILES,
)


def _arg_value(name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(sys.argv):
        if arg == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return default


def _pop_flag(name: str) -> bool:
    if name not in sys.argv:
        return False
    sys.argv.remove(name)
    return True


def _pop_option(name: str) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(sys.argv):
        if arg == name:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"Missing value for {name}")
            value = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return value
        if arg.startswith(prefix):
            value = arg[len(prefix) :]
            del sys.argv[index]
            return value
    return None


def _validate_profile_arguments(
    profile: str,
    token_adapter_only: bool,
    token_inventory_path: str | None,
) -> Path:
    if profile not in VISION_SFT_PROFILES:
        allowed = ", ".join(VISION_SFT_PROFILES)
        raise SystemExit(f"Unsupported --catan_trainable_profile {profile!r}; choose {allowed}")
    if (profile == VISION_ONLY) != token_adapter_only:
        raise SystemExit(
            "vision_only requires --catan_token_adapter_only and " "vision_language_lora forbids it"
        )
    if not token_inventory_path:
        raise SystemExit(f"{profile} requires --catan_token_inventory")

    expected = {
        "--bits": "16",
        "--lora_enable": "true",
        "--vision_lora": "false",
        "--freeze_llm": "true",
        "--freeze_vision_tower": "false",
        "--freeze_merger": "false",
        "--enable_reasoning": "false",
    }
    for name, expected_value in expected.items():
        actual = _arg_value(name)
        if actual is None or actual.lower() != expected_value:
            raise SystemExit(f"{profile} requires {name}={expected_value}; received {actual!r}")
    for name in ("--vision_lr", "--merger_lr"):
        value = _arg_value(name)
        if value is None or float(value) <= 0:
            raise SystemExit(f"{profile} requires a positive {name}")

    output_dir = _arg_value("--output_dir")
    if not output_dir:
        raise SystemExit(f"{profile} requires --output_dir")
    return Path(output_dir)
