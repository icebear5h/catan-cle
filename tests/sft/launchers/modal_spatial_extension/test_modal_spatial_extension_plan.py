"""Plan inheritance, config approval, and bounded options."""

import copy
from pathlib import Path

import pytest

from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.launchers.spatial import modal_spatial_extension as extension

from .support import write_checkpoint, write_json, write_rows


def test_plan_inherits_every_field_and_two_passes_without_upload(plan: dict[str, object]) -> None:
    parent = plan["parent_config"]
    config = extension.validate_config(plan["config"], "test-run", parent)
    changed = {k for k in parent if parent[k] != plan["config"][k]}
    assert changed == {"max_steps", "save_total_limit", "initial_bundle", "output_dir"}
    assert config.max_steps == 256 and config.seed == 44 and config.num_train_epochs == parent["num_train_epochs"]
    assert config.resume_from_checkpoint is None and config.initial_bundle == extension.PARENT_CHECKPOINT
    assert config.eval_steps == config.save_steps == 32 and config.save_total_limit == 8
    assert config.max_steps * config.per_device_train_batch_size * config.gradient_accumulation_steps == 2 * 1024
    assert plan["mixture"]["steps"] == 128 and plan["policy"]["cumulative_mixed_updates"] == 384
    assert plan["policy"]["family_steps"]["full_board_readout"] == 64
    assert set(plan["saved_baselines"]) == set(extension.PANEL_BUDGETS)
    assert plan["policy"]["stages"] == ["cpu_preflight", "train256", "post_all_six"]
    assert plan["parent_audit"]["files_sha256"]["visual_model.safetensors"] != launcher.VISUAL_SHA256
    assert plan["input_files_sha256"] and plan["remote_receipts_sha256"]
    assert plan["parent_checkpoint"] == extension.PARENT_CHECKPOINT
    assert {"config", "dataset_inputs", "panels", "source_sha256"} <= plan.keys()


@pytest.mark.parametrize("field,value", [
    ("max_steps", 128), ("max_steps", 384), ("seed", 42), ("num_train_epochs", 2),
    ("save_total_limit", 4), ("save_steps", 64), ("eval_steps", 64),
    ("per_device_train_batch_size", 8), ("gradient_accumulation_steps", 4),
    ("learning_rate", 0.001), ("resume_from_checkpoint", "parent"),
    ("initial_bundle", "older"), ("token_init", "vocab_gaussian"),
    ("eval_jsonl", "another-panel"), ("train_jsonl", "another-dataset"), ("image_max_pixels", 65536),
    ("input_mode", "text"), ("max_sequence_length", 8192),
])
def test_config_rejects_unapproved_or_noninherited_changes(plan: dict[str, object], field: str, value: int | float | str) -> None:
    with pytest.raises(ValueError):
        extension.validate_config({**plan["config"], field: value}, "test-run", plan["parent_config"])


@pytest.mark.parametrize("legacy_parent", [False, True])
@pytest.mark.parametrize("legacy_payload", [False, True])
def test_new_defaults_normalize_only_comparisons_without_changing_receipt_hashes(plan: dict[str, object], legacy_parent: bool, legacy_payload: bool) -> None:
    parent, payload = copy.deepcopy(plan["parent_config"]), copy.deepcopy(plan["config"])
    for value, legacy in ((parent, legacy_parent), (payload, legacy_payload)):
        if legacy:
            value.pop("input_mode")
            value.pop("max_sequence_length")
    originals = copy.deepcopy((parent, payload))
    hashes = extension.digest(parent), extension.digest(payload)
    config = extension.validate_config(payload, "test-run", parent)
    assert config.input_mode == "vision" and config.max_sequence_length is None
    assert (parent, payload) == originals
    assert (extension.digest(parent), extension.digest(payload)) == hashes
    # Normalization must not fill other missing historical fields or drop unknowns.
    missing_seed = {k: v for k, v in payload.items() if k != "seed"}
    for bad in (missing_seed, {**payload, "unknown_option": 1}):
        with pytest.raises(ValueError, match="inherit ALL"):
            extension.validate_config(bad, "test-run", parent)


@pytest.mark.parametrize("legacy_saved", [False, True])
def test_checkpoint_semantic_defaults_preserve_raw_hashes_and_hard_constraints(plan: dict[str, object], tmp_path: Path, legacy_saved: bool) -> None:
    modern = copy.deepcopy(plan["config"])
    legacy = {k: v for k, v in modern.items() if k not in ("input_mode", "max_sequence_length")}
    saved, expected = (legacy, modern) if legacy_saved else (modern, legacy)
    path = tmp_path / "checkpoint-128"
    write_checkpoint(path, saved, 128)
    config_path = path / "training_config.json"
    original_bytes = config_path.read_bytes()
    audit = extension.checkpoint_audit(path, expected, parent=False)
    assert audit["files_sha256"]["training_config.json"] == extension.sha256_file(config_path)
    assert extension.checkpoint_audit(path, expected, parent=False, expected_audit=audit) == audit
    assert config_path.read_bytes() == original_bytes
    for change in ({"input_mode": "text"}, {"max_sequence_length": 8192}, {"seed": 999}):
        with pytest.raises(ValueError, match="configuration"):
            extension.checkpoint_audit(path, {**expected, **change}, parent=False)
    # Equal semantic configs cannot legitimize a changed pinned file hash.
    config_path.write_bytes(original_bytes + b"\n")
    with pytest.raises(ValueError, match="hashes"):
        extension.checkpoint_audit(path, expected, parent=False, expected_audit=audit)


def test_fixed_bounded_options_and_historical_limits() -> None:
    for options, timeout in ((extension.TRAIN_GPU_OPTIONS, 7200), (extension.POST_GPU_OPTIONS, 3600)):
        assert options["timeout"] == timeout and options["startup_timeout"] == 300
        assert options["gpu"] == "H200" and options["cpu"] == (16.0, 16.0)
        assert options["memory"] == (131072, 131072)
        assert options["retries"] == 0 and options["max_containers"] == 1
    assert extension.PREFLIGHT_TIMEOUT == 1200 and extension.STARTUP_TIMEOUT == 300
    total = 1200 + 7200 + 3600 + 3 * (300 + extension.WAIT_GRACE)
    assert extension.COORDINATOR_TIMEOUT > total
    assert launcher.FIXED_CONFIG["max_steps"] == 128
    assert launcher.FIXED_CONFIG["save_total_limit"] == 4


@pytest.mark.parametrize("problem", ["result", "post", "manifest", "train_order", "panel_bytes", "records", "summary"])
def test_plan_rejects_local_receipt_and_frozen_data_drift(receipts: Path, problem: str) -> None:
    if problem in ("result", "post"):
        path = extension.PARENT_RESULT if problem == "result" else extension.PARENT_RESULT.parent / "post/result.json"
        value = extension.read_json(path)
        value["status"] = "failed"
        write_json(path, value)
    elif problem == "manifest":
        value = extension.read_json(receipts)
        value["schema"] = "new-dataset"
        write_json(receipts, value)
    elif problem == "train_order":
        value = extension.read_json(receipts)
        path = Path(value["train_jsonl"])
        rows = [r for _, r in extension.iter_jsonl(path)]
        write_rows(path, list(reversed(rows)))
    elif problem == "panel_bytes":
        path = Path(extension.OLD_PANELS["fullboard"]["eval_jsonl"])
        path.write_text(path.read_text() + "\n")
    else:
        path = extension.PARENT_RESULT.parent / f"post/paths-{problem}.json{'l' if problem == 'records' else ''}"
        path.write_text(path.read_text() + " " if problem == "records" else "{}")
    with pytest.raises(ValueError):
        extension.build_plan(receipts, "test-run")
