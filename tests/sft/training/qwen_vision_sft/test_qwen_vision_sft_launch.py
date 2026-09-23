"""Launch identity and remote manifest idempotence."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import sft.launchers.qwen_series.modal_qwen_series_vision_train as modal_vision_train
from sft.qwen_series_vision_sft import (
    L40S_PROFILE,
    VISION_ONLY,
    VisionSftConfig,
    launch_identity,
)
from sft.scripts.eval.eval_qwen_vl_adapter import (
    expected_text,
    load_non_lora_adapter_weights,
    user_text,
)

from .support import _option


def test_launch_identity_is_order_independent() -> None:
    assert launch_identity({"a": 1, "b": 2}) == launch_identity({"b": 2, "a": 1})


def test_atomic_qwen_conversation_is_eval_compatible() -> None:
    row = {
        "image": "board.png",
        "conversations": [
            {"from": "human", "value": "<image>\n<E00_01><Q_EDGE_OWNER>"},
            {"from": "gpt", "value": "<A_EDGE_OWNER_EMPTY>"},
        ],
    }

    assert user_text(row) == "<E00_01><Q_EDGE_OWNER>"
    assert expected_text(row) == "<A_EDGE_OWNER_EMPTY>"


def test_new_vision_adapter_requires_non_lora_state(tmp_path: Path) -> None:
    (tmp_path / "launch_manifest.json").write_text(
        json.dumps({"schema": "catan_qwen_vision_sft_launch_identity/v1"})
    )

    with pytest.raises(FileNotFoundError, match="missing required non-LoRA"):
        load_non_lora_adapter_weights(object(), tmp_path, object())


def test_remote_launch_manifest_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commits = {"cache": 0, "runs": 0}
    calls = []

    monkeypatch.setattr(
        modal_vision_train,
        "hf_cache",
        SimpleNamespace(commit=lambda: commits.__setitem__("cache", commits["cache"] + 1)),
    )
    monkeypatch.setattr(
        modal_vision_train,
        "sft_runs",
        SimpleNamespace(commit=lambda: commits.__setitem__("runs", commits["runs"] + 1)),
    )

    def fake_run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))
        output_dir = Path(_option(command, "--output_dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "adapter_config.json",
            "tokenizer_config.json",
            "trainable_parameters.json",
        ):
            (output_dir / name).write_text("{}\n")
        (output_dir / "adapter_model.safetensors").write_bytes(b"adapter")
        (output_dir / "non_lora_state_dict.bin").write_bytes(b"vision")

    monkeypatch.setattr(modal_vision_train.subprocess, "run", fake_run)
    config = VisionSftConfig(
        profile=VISION_ONLY,
        model_id="Qwen/Qwen3-VL-4B-Instruct",
    )
    dataset = {
        "source_sha256": "a" * 64,
        "combined_sha256": "b" * 64,
        "rows": 4,
        "unique_images": 4,
        "max_prompt_characters": 128,
        "max_answer_characters": 20,
        "annotation_root": "artifacts/annotations",
        "image_root": "artifacts/images",
        "token_inventory": {
            "reference": "tokens.json",
            "sha256": "c" * 64,
            "counts": {"atlas": 154, "query": 6, "answer": 60, "total": 220},
        },
    }
    kwargs = {
        "hardware": L40S_PROFILE,
        "train_json": "/data/train.json",
        "image_folder": "/data/images",
        "output_dir": str(tmp_path / "run"),
        "token_inventory": "/data/trainable_tokens.json",
        "config_payload": asdict(config),
        "dataset": dataset,
    }

    first = modal_vision_train._run_remote(**kwargs)
    second = modal_vision_train._run_remote(**kwargs)

    assert first["status"] == "completed"
    assert second["status"] == "already_completed"
    assert len(calls) == 1
    assert commits == {"cache": 1, "runs": 1}
    manifest = json.loads((tmp_path / "run/launch_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["identity"] == first["identity"]
