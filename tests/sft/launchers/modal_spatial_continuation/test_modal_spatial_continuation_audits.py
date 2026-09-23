"""Baseline rescoring, preflight, and checkpoint audits."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sft.launchers.spatial import modal_spatial_continuation as launcher

from .support import fake_volumes, panel_result, write_json, write_rows


def test_manifest_uses_latest_evaluated_checkpoint_and_checks_fixed_baseline_hashes(plan: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "parent/result.json"
    parent_config = {**plan["parent_config"], "initial_bundle": "/older/checkpoint-128"}
    write_json(parent, {"status": "completed", "config": parent_config,
                        "evaluation": {"status": "completed", "checkpoint": launcher.PARENT_CHECKPOINT}})
    write_json(parent.with_name("launch.json"), {"source_sha256": {"sft/modal_full_board_pilot.py": "legacy"}})
    monkeypatch.setattr(launcher, "PARENT_RESULT", parent)
    monkeypatch.setattr(launcher, "OLD_PANELS", {label: {"eval_jsonl": p["eval_jsonl"], "image_root": p["image_root"],
                                                      "sha256": p["identity"]["sha256"]}
                                               for label, p in plan["panels"].items() if label not in launcher.NEW_LABELS})
    # Summary semantics are independently verified by CPU preflight, not invented
    # as part of the dataset manifest contract.
    summary = tmp_path / "summary.json"
    write_json(summary, {"saved": "unit-test receipt"})
    monkeypatch.setattr(launcher, "SPATIAL_RECEIPT", summary)
    monkeypatch.setattr(launcher, "BOARD_RECEIPT", summary)
    metadata = tmp_path / "metadata.json"
    write_json(metadata, {"builder_owned": True})
    inputs = {"schema": "catan_spatial_continuation_inputs/v1", "metadata": str(metadata),
              **{k: plan["config"][k] for k in ("train_jsonl", "image_root", "token_inventory")},
              "new_panels": {launcher.NEW_PANEL_TASKS[k]: {name: plan["panels"][k][name] for name in
                             ("eval_jsonl", "image_root", "max_new_tokens", "batch_size")}
                             for k in launcher.NEW_LABELS}}
    manifest = tmp_path / "dataset_inputs.json"
    write_json(manifest, inputs)
    built = launcher.build_plan(manifest, "test-run")
    assert built["config"]["initial_bundle"] == launcher.PARENT_CHECKPOINT
    assert built["config"]["seed"] == parent_config["seed"]
    assert built["config_sha256"] == launcher.digest(built["config"])
    assert built["mixture"]["family_steps"] == launcher.FAMILY_STEPS
    assert built["dataset_inputs"] == inputs and built["metadata"]["sha256"] == launcher.sha256_file(metadata)
    assert not built["policy"]["blank_controls"]
    launcher.OLD_PANELS["spatial"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="fixed baseline hash"):
        launcher.build_plan(manifest, "test-run")


def test_saved_baseline_is_independently_rescored_and_identity_checked(plan: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel = plan["panels"]["spatial"]
    rows = [r for _, r in launcher.iter_jsonl(Path(panel["eval_jsonl"]))]
    records = [{"id": r["row_id"], "expected": launcher.evaluator.expected_text(r),
                "response": "<T00>" if i < 57 else "no", "score": {"correct": i < 57},
                "metadata": launcher.evaluator.evaluation_metadata(r, image_variant="original")}
               for i, r in enumerate(rows)]
    output = tmp_path / "baseline"
    write_rows(output / "records.jsonl", records)
    checkpoint = {"files_sha256": {"visual_model.safetensors": launcher.VISUAL_SHA256},
                  "semantic_tokens": {"token_ids": [1, 2]}}
    summary = {**launcher.evaluator.summarize(records), "adapter_dir": launcher.PARENT_CHECKPOINT,
               "adapter_evidence": {"adapter_dir": launcher.PARENT_CHECKPOINT, "adapter_loaded": True,
                                    "visual_state": {"sha256": launcher.VISUAL_SHA256},
                                    "semantic_tokens": {"token_ids": [1, 2]}},
               "model_id": launcher.SNAPSHOT, "eval_jsonl": panel["eval_jsonl"],
               "image_root": panel["image_root"], "image_variant": "original", "reasoning_enabled": False,
               "candidate_scoring": False, "bits": 16, "batch_size": 48, "max_new_tokens": 16,
               "long_max_new_tokens": 512, "eval_source_sha256": panel["identity"]["sha256"],
               "precision": {"preserve_visual_fp32": True}}
    write_json(output / "summary.json", summary)
    saved = {"summary": summary, "output_dir": str(output),
             "summary_sha256": launcher.sha256_file(output / "summary.json")}
    monkeypatch.setitem(launcher.OLD_PANELS, "spatial", {"sha256": panel["identity"]["sha256"]})
    result = launcher.matched_baseline("spatial", saved, panel, checkpoint)
    assert result["reused"] and result["original_only"] and result["summary"]["correct"] == 57
    assert result["records_sha256"] == launcher.sha256_file(output / "records.jsonl")
    bad = copy.deepcopy(checkpoint)
    bad["files_sha256"]["visual_model.safetensors"] = "different"
    with pytest.raises(ValueError, match="checkpoint identity"):
        launcher.matched_baseline("spatial", saved, panel, bad)
    records[0]["response"] = "no"
    write_rows(output / "records.jsonl", records)
    with pytest.raises(ValueError, match="scorer changes"):
        launcher.matched_baseline("spatial", saved, panel, checkpoint)


def test_cpu_preflight_checks_tokenizers_lengths_identities_and_commits(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    fake_volumes(monkeypatch, events)
    inventory = launcher.load_token_inventory(plan["config"]["token_inventory"])
    tokens = inventory["atlas_tokens"]
    checkpoint = {"semantic_tokens": {"tokens": tokens, "token_ids": list(range(154))}}
    monkeypatch.setattr(launcher, "checkpoint_audit", lambda *a, **k: checkpoint)
    write_json(Path(launcher.PARENT_CHECKPOINT) / "adapter_config.json",
               {"trainable_token_indices": {"embed_tokens": list(range(154)), "lm_head": list(range(154))}})
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [tokens.index(text)] if text in tokens else [0],
                                add_tokens=lambda added: len(added))
    sources: list[str] = []

    def load(source: str, local_files_only: bool) -> object:
        assert local_files_only
        sources.append(source)
        return tokenizer

    monkeypatch.setattr(launcher.AutoTokenizer, "from_pretrained", load)
    is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda path: str(path) == launcher.SNAPSHOT or is_dir(path))
    monkeypatch.setattr(launcher, "matched_baseline", lambda label, *args: panel_result(plan, label))
    plan["legacy_pilot_sha256"] = launcher.sha256_file(launcher.PROJECT_ROOT / "sft/launchers/full_board/modal_full_board_pilot.py")
    report = launcher.continuation_preflight.get_raw_f()(plan)
    assert report["status"] == "completed"
    assert sources == [launcher.PARENT_CHECKPOINT, launcher.SNAPSHOT]
    assert report["mixture"] == plan["mixture"]
    assert set(report["panel_tokens"]) == set(launcher.PANEL_BUDGETS)
    assert set(report["baselines"]) == {"spatial", "fullboard"}
    assert events[:3] == ["hf_cache.reload", "sft_data.reload", "sft_runs.reload"]
    assert events[-1] == "sft_runs.commit"
    tokenizer.encode = lambda text, add_special_tokens: [999]
    with pytest.raises(ValueError, match="tokenizer atlas IDs"):
        launcher.continuation_preflight.get_raw_f()(plan)


def test_runtime_rejects_source_config_budget_and_baseline_drift(plan: dict[str, object]) -> None:
    launcher.verify_runtime(plan)
    for problem in ("source", "config", "panel", "baseline"):
        changed = copy.deepcopy(plan)
        if problem == "source":
            changed["source_sha256"] = {}
        elif problem == "config":
            changed["config"]["seed"] += 1
        elif problem == "panel":
            changed["panels"]["paths"]["max_new_tokens"] = 512
        else:
            changed["saved_baselines"].pop("fullboard")
        with pytest.raises(ValueError):
            launcher.verify_runtime(changed)


def test_checkpoint_audit_checks_files_hash_profile_rows_rank_and_fp32(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    directory = Path(launcher.PARENT_CHECKPOINT)
    directory.mkdir(parents=True)
    config = plan["parent_config"]
    scope = {"errors": [], "profile": "vision_tokens_lora", "semantic_tokens": {"token_ids": list(range(154))},
             "groups": {name: {"parameters": 0 if name in ("forbidden", "vision_lora") else 1}
                        for name in ("vision", "merger", "atlas_input_rows", "atlas_output_rows", "language_lora", "forbidden", "vision_lora")},
             "dtypes": {name: {"torch.float32": 1} for name in ("vision", "merger")},
             "parameters": [{"category": name, "shape": [154, 2]} for name in ("atlas_input_rows", "atlas_output_rows")]}
    for name, payload in (("training_config.json", config), ("trainer_state.json", {"global_step": 128}),
                          ("trainable_parameters.json", scope), ("tokenizer_config.json", {}), ("tokenizer.json", {}),
                          ("adapter_config.json", {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05})):
        write_json(directory / name, payload)
    adapter = {"embed_tokens.trainable_tokens_delta": torch.zeros(154, 2),
               "lm_head.trainable_tokens_delta": torch.zeros(154, 2),
               "language.lora_A.weight": torch.zeros(8, 2), "language.lora_B.weight": torch.zeros(2, 8)}
    save_file(adapter, directory / "adapter_model.safetensors")
    save_file({f"visual.{i}": torch.zeros(1) for i in range(333)}, directory / "visual_model.safetensors")
    visual_sha = launcher.sha256_file(directory / "visual_model.safetensors")
    monkeypatch.setattr(launcher, "VISUAL_SHA256", visual_sha)
    result = launcher.checkpoint_audit(directory, config, parent=True)
    assert result["visual_dtypes"] == {"F32": 333}
    assert result["files_sha256"]["visual_model.safetensors"] == visual_sha
    monkeypatch.setattr(launcher, "VISUAL_SHA256", "wrong")
    with pytest.raises(ValueError, match="visual checkpoint SHA256"):
        launcher.checkpoint_audit(directory, config, parent=True)
    monkeypatch.setattr(launcher, "VISUAL_SHA256", visual_sha)
    adapter["embed_tokens.trainable_tokens_delta"] = torch.zeros(153, 2)
    save_file(adapter, directory / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="tensor headers"):
        launcher.checkpoint_audit(directory, config, parent=True)
    adapter["embed_tokens.trainable_tokens_delta"] = torch.zeros(154, 2)
    save_file(adapter, directory / "adapter_model.safetensors")
    save_file({f"visual.{i}": torch.zeros(1, dtype=torch.bfloat16) for i in range(333)}, directory / "visual_model.safetensors")
    with pytest.raises(ValueError, match="333 FP32"):
        launcher.checkpoint_audit(directory, config, parent=False)
    write_json(directory / "trainer_state.json", {"global_step": 127})
    with pytest.raises(ValueError, match="global_step"):
        launcher.checkpoint_audit(directory, config, parent=False)
