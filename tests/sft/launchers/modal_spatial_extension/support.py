"""Shared helpers for modal spatial extension plan, preflight, training, and coordination."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from modal_spatial_continuation.support import FakeCall, fake_volumes, write_json, write_rows
from safetensors.torch import save_file

from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.launchers.spatial import modal_spatial_extension as extension


def write_checkpoint(path: Path, config: dict[str, object], step: int) -> None:
    scope = {"errors": [], "profile": "vision_tokens_lora",
             "semantic_tokens": {"tokens": extension.load_token_inventory(config["token_inventory"])["atlas_tokens"],
                                 "token_ids": list(range(154))},
             "groups": {name: {"parameters": 0 if name in ("forbidden", "vision_lora") else 1}
                        for name in ("vision", "merger", "atlas_input_rows", "atlas_output_rows",
                                     "language_lora", "forbidden", "vision_lora")},
             "dtypes": {name: {"torch.float32": 1} for name in ("vision", "merger")},
             "parameters": [{"category": name, "shape": [154, 2]}
                            for name in ("atlas_input_rows", "atlas_output_rows")]}
    for name, payload in (
        ("training_config.json", config), ("trainable_parameters.json", scope),
        ("trainer_state.json", {"global_step": step, "log_history": [
            {"step": i, "eval_loss": 1 / i, "epoch": i / 128} for i in range(32, step + 1, 32)]}),
        ("tokenizer_config.json", {}), ("tokenizer.json", {}),
        ("adapter_config.json", {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                                 "trainable_token_indices": {k: list(range(154)) for k in ("embed_tokens", "lm_head")}}),
    ):
        write_json(path / name, payload)
    save_file({"embed_tokens.trainable_tokens_delta": torch.zeros(154, 2),
               "lm_head.trainable_tokens_delta": torch.zeros(154, 2),
               "language.lora_A.weight": torch.zeros(8, 2),
               "language.lora_B.weight": torch.zeros(2, 8)}, path / "adapter_model.safetensors")
    save_file({f"visual.{i}": torch.zeros(1) for i in range(333)}, path / "visual_model.safetensors")


def volumes(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    events: list[object] = []
    fake_volumes(monkeypatch, events)
    monkeypatch.setattr(extension, "sft_runs", launcher.sft_runs)
    return events


def preflight(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    events = volumes(monkeypatch)
    tokens = extension.load_token_inventory(plan["config"]["token_inventory"])["atlas_tokens"]
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [tokens.index(text)] if text in tokens else [0],
                                add_tokens=lambda added: len(added))
    loaded: list[str] = []

    def load(source: str, local_files_only: bool) -> SimpleNamespace:
        assert local_files_only
        loaded.append(source)
        return tokenizer

    monkeypatch.setattr(extension.AutoTokenizer, "from_pretrained", load)
    is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda path: str(path) == extension.SNAPSHOT or is_dir(path))
    result = extension.extension_preflight.get_raw_f()(plan)
    assert loaded == [extension.PARENT_CHECKPOINT, extension.SNAPSHOT]
    assert events[:3] == ["hf_cache.reload", "sft_data.reload", "sft_runs.reload"]
    return result


def train_success(
    plan: dict[str, object],
    audit: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_at: int | None = None,
) -> dict[str, object]:
    calls: list[object] = []

    def train(config: object) -> dict[str, int]:
        calls.append(config)
        assert config.max_steps == 256 and config.resume_from_checkpoint is None
        for step in extension.CHECKPOINT_STEPS:
            if step == fail_at:
                raise RuntimeError("training failed")
            write_checkpoint(Path(config.output_dir) / "checkpoints" / f"checkpoint-{step}", asdict(config), step)
        return {"global_step": 256}

    monkeypatch.setattr(extension, "run_training", train)
    result = extension.train_bounded.get_raw_f()(plan, audit)
    assert calls == [extension.TrainConfig(**plan["config"])]
    return result


def setup_coordinator(
    plan: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    failure: tuple[str, BaseException | str] | None = None,
) -> tuple[list[object], list[FakeCall], Path]:
    volumes(monkeypatch)
    monkeypatch.setattr(extension.modal, "current_function_call_id", lambda: "fc-coordinator")
    directory = extension.RUN_ROOT / "pipelines" / plan["run_name"]
    plan["reservation_id"] = "reservation-test"
    write_json(directory / "launch.json", plan)
    events: list[object] = []
    calls: list[FakeCall] = []
    results = {"preflight": {"status": "completed", "baselines": plan["saved_baselines"]},
               "training": {"status": "completed"},
               "post": {"status": "completed", "panels": plan["saved_baselines"]}}

    def spawn(name: str, args: object) -> FakeCall:
        persisted = extension.read_json(directory / "result.json")
        assert persisted["phase"] == name
        assert extension.read_json(directory / "launch.json")["coordinator_call_id"] == "fc-coordinator"
        if name == "training":
            assert args[1] == results["preflight"] and calls[-1].object_id == "fc-preflight"
        if name == "post":
            assert args[2] == results["training"] and calls[-1].object_id == "fc-training"
        error = failure[1] if failure and failure[0] == name else None
        result = results[name]
        if error == "status":
            result, error = {"status": "failed"}, None
        call = FakeCall(name, result, events, error)
        original_get = call.get

        def get(timeout: float) -> dict[str, object]:
            assert extension.read_json(directory / "result.json")["stages"][name]["call_id"] == call.object_id
            return original_get(timeout)

        call.get = get
        events.append(("spawn", name))
        calls.append(call)
        return call

    for name, function in (("preflight", "extension_preflight"), ("training", "train_bounded"), ("post", "evaluate_bounded")):
        monkeypatch.setattr(extension, function, SimpleNamespace(spawn=lambda *args, name=name: spawn(name, args)))
    return events, calls, directory


__all__ = [
    "FakeCall",
    "fake_volumes",
    "preflight",
    "setup_coordinator",
    "train_success",
    "volumes",
    "write_checkpoint",
    "write_json",
    "write_rows",
]
