"""Offline receipts, real CPU scoring/tensor audits, and bounded transport tests."""

import copy
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import test_modal_spatial_continuation as previous
from sft import modal_spatial_extension as extension
from sft.spatial_tasks import node_tile_tokens, shortest_node_path

base_plan = previous.plan
offline_boundaries = previous.no_remote
write_json = previous.write_json
write_rows = previous.write_rows


@pytest.fixture(autouse=True)
def no_models_or_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline test attempted model loading or real training")

    monkeypatch.setattr(extension, "run_training", forbidden)
    monkeypatch.setattr(extension.evaluator, "load_model", forbidden)
    monkeypatch.setattr(extension.AutoTokenizer, "from_pretrained", forbidden)


def write_checkpoint(path, config, step):
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


@pytest.fixture
def receipts(base_plan, tmp_path, monkeypatch):
    old = previous.launcher
    monkeypatch.setattr(extension, "RUN_ROOT", old.RUN_ROOT)
    monkeypatch.setattr(extension, "LOCAL_RUN_ROOT", old.LOCAL_RUN_ROOT)
    checkpoint = old.RUN_ROOT / extension.PARENT_RUN / "checkpoints/checkpoint-128"
    parent_result = old.LOCAL_RUN_ROOT / extension.PARENT_RUN / "result.json"
    monkeypatch.setattr(extension, "PARENT_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(extension, "PARENT_RESULT", parent_result)
    monkeypatch.setattr(extension, "source_hashes", lambda: {"scorer.py": "version-one"})
    parent_config = {**base_plan["parent_config"], "output_dir": str(checkpoint.parents[1])}
    write_checkpoint(checkpoint, parent_config, 128)
    audit = extension.checkpoint_audit(checkpoint, parent_config, parent=False)
    # The legacy transport fixtures deliberately omit solver targets. Supply
    # valid gold here because these tests run the actual task-aware scorer.
    tiles = node_tile_tokens("<N00>")
    examples = {
        "node_tiles": ({"node": "<N00>"}, " ".join(tiles)),
        "paths": ({"start": "<N00>", "end": "<N03>"}, " ".join(shortest_node_path("<N00>", "<N03>"))),
        "local": ({"node": "<N00>"}, json.dumps({t: {"resource": "wood", "number": 6} for t in tiles})),
        "production": ({"color": "RED", "roll": 8}, json.dumps(dict.fromkeys(("wood", "brick", "sheep", "wheat", "ore"), 0))),
    }
    for label, (target, answer) in examples.items():
        panel = base_plan["panels"][label]
        rows = [r for _, r in extension.iter_jsonl(Path(panel["eval_jsonl"]))]
        for row in rows:
            row["metadata"]["target"] = target
            row["messages"][-1]["content"] = answer
        write_rows(Path(panel["eval_jsonl"]), rows)
        panel["identity"] = extension.dataset_identity(panel["eval_jsonl"], panel["image_root"])
    old_panels = {label: {**base_plan["panels"][label], "sha256": base_plan["panels"][label]["identity"]["sha256"]}
                  for label in ("spatial", "fullboard")}
    monkeypatch.setattr(extension, "OLD_PANELS", old_panels)
    monkeypatch.setattr(old, "OLD_PANELS", old_panels)
    metadata = tmp_path / "metadata.json"
    write_json(metadata, {"schema": "catan_spatial_continuation/v1"})
    inputs = {"schema": "catan_spatial_continuation_inputs/v1", "metadata": str(metadata),
              **{k: base_plan["config"][k] for k in ("train_jsonl", "image_root", "token_inventory")},
              "new_panels": {task: {key: base_plan["panels"][label][key] for key in
                                    ("eval_jsonl", "image_root", "batch_size", "max_new_tokens")}
                             for label, task in extension.NEW_PANEL_TASKS.items()}}
    manifest = tmp_path / "dataset_inputs.json"
    write_json(manifest, inputs)
    panels = {}
    for label, panel in base_plan["panels"].items():
        rows = [r for _, r in extension.iter_jsonl(Path(panel["eval_jsonl"]))]
        records = []
        for row in rows:
            target = extension.evaluator.expected_text(row)
            meta = extension.evaluator.evaluation_metadata(row, image_variant="original")
            score = extension.evaluator.score_response(target, target, metadata=meta)
            records.append({"id": row["row_id"], "expected": target, "response": target,
                            "metadata": meta, "score": score, "candidate_score": None})
        summary = {**extension.evaluator.summarize(records), "adapter_dir": str(checkpoint),
                   "adapter_evidence": {"adapter_dir": str(checkpoint), "adapter_loaded": True,
                       "visual_state": {"sha256": audit["files_sha256"]["visual_model.safetensors"], "loaded": True},
                       "semantic_tokens": audit["semantic_tokens"]},
                   "model_id": extension.SNAPSHOT, "eval_jsonl": panel["eval_jsonl"],
                   "image_root": panel["image_root"], "image_variant": "original", "reasoning_enabled": False,
                   "candidate_scoring": False, "bits": 16, "batch_size": panel["batch_size"],
                   "max_new_tokens": panel["max_new_tokens"], "long_max_new_tokens": panel["max_new_tokens"],
                   "eval_source_sha256": panel["identity"]["sha256"], "precision": {"preserve_visual_fp32": True}}
        output = old.RUN_ROOT / "pipelines" / extension.PARENT_RUN / "post" / label
        write_json(output / "summary.json", summary)
        write_rows(output / "records.jsonl", records)
        write_json(parent_result.parent / f"post/{label}-summary.json", summary)
        write_rows(parent_result.parent / f"post/{label}-records.jsonl", records)
        panels[label] = {"summary": summary, "identity": panel["identity"],
                         "records_path": str(output / "records.jsonl"),
                         "records_sha256": extension.sha256_file(output / "records.jsonl"),
                         "scorer_sha256": extension.digest(extension.source_hashes()),
                         "conditions": extension.evaluation_conditions(label)}
    post = {"status": "completed", "checkpoint": str(checkpoint), "checkpoint_audit": audit,
            "panels": panels, "source_sha256": extension.source_hashes()}
    preflight = {"status": "completed", "train_identity": base_plan["train_identity"],
                 "teacher_eval_identity": base_plan["panels"]["fullboard"]["identity"]}
    parent = {"status": "completed", "phase": "completed", "config": parent_config,
              "stages": {"post": {"status": "completed", "result": post},
                         "training": {"status": "completed"},
                         "preflight": {"status": "completed", "result": preflight}}}
    write_json(parent_result, parent)
    write_json(parent_result.parent / "post/result.json", post)
    write_json(old.RUN_ROOT / "pipelines" / extension.PARENT_RUN / "result.json", parent)
    write_json(old.RUN_ROOT / "pipelines" / extension.PARENT_RUN / "post/result.json", post)
    write_json(parent_result.with_name("launch.json"), {
        "config": parent_config, "config_sha256": extension.digest(parent_config),
        "dataset_inputs": inputs, "train_identity": base_plan["train_identity"], "mixture": base_plan["mixture"],
        "panels": base_plan["panels"], "input_files_sha256": {str(path): extension.sha256_file(path)
            for path in (manifest, metadata, Path(inputs["token_inventory"]))}})
    return manifest


@pytest.fixture
def plan(receipts):
    return extension.build_plan(receipts, "test-run")


def volumes(monkeypatch):
    events = []
    previous.fake_volumes(monkeypatch, events)
    monkeypatch.setattr(extension, "sft_runs", previous.launcher.sft_runs)
    return events


def preflight(plan, monkeypatch):
    events = volumes(monkeypatch)
    tokens = extension.load_token_inventory(plan["config"]["token_inventory"])["atlas_tokens"]
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: [tokens.index(text)] if text in tokens else [0],
                                add_tokens=lambda added: len(added))
    loaded = []

    def load(source, local_files_only):
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


def test_plan_inherits_every_field_and_two_passes_without_upload(plan):
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
    assert plan["parent_audit"]["files_sha256"]["visual_model.safetensors"] != previous.launcher.VISUAL_SHA256
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
def test_config_rejects_unapproved_or_noninherited_changes(plan, field, value):
    with pytest.raises(ValueError):
        extension.validate_config({**plan["config"], field: value}, "test-run", plan["parent_config"])


@pytest.mark.parametrize("legacy_parent", [False, True])
@pytest.mark.parametrize("legacy_payload", [False, True])
def test_new_defaults_normalize_only_comparisons_without_changing_receipt_hashes(plan, legacy_parent, legacy_payload):
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
def test_checkpoint_semantic_defaults_preserve_raw_hashes_and_hard_constraints(plan, tmp_path, legacy_saved):
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


def test_fixed_bounded_options_and_historical_limits():
    for options, timeout in ((extension.TRAIN_GPU_OPTIONS, 7200), (extension.POST_GPU_OPTIONS, 3600)):
        assert options["timeout"] == timeout and options["startup_timeout"] == 300
        assert options["gpu"] == "H200" and options["cpu"] == (16.0, 16.0)
        assert options["memory"] == (131072, 131072)
        assert options["retries"] == 0 and options["max_containers"] == 1
    assert extension.PREFLIGHT_TIMEOUT == 1200 and extension.STARTUP_TIMEOUT == 300
    total = 1200 + 7200 + 3600 + 3 * (300 + extension.WAIT_GRACE)
    assert extension.COORDINATOR_TIMEOUT > total
    assert previous.launcher.FIXED_CONFIG["max_steps"] == 128
    assert previous.launcher.FIXED_CONFIG["save_total_limit"] == 4


@pytest.mark.parametrize("problem", ["result", "post", "manifest", "train_order", "panel_bytes", "records", "summary"])
def test_plan_rejects_local_receipt_and_frozen_data_drift(receipts, problem):
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


def test_cpu_preflight_really_rescores_every_saved_record_and_entire_summary(plan, monkeypatch):
    calls = []
    score = extension.evaluator.score_response

    def track(expected, response, *, metadata):
        calls.append(metadata)
        return score(expected, response, metadata=metadata)

    monkeypatch.setattr(extension.evaluator, "score_response", track)
    result = preflight(plan, monkeypatch)
    assert len(calls) == sum(budget[0] for budget in extension.PANEL_BUDGETS.values()) == 430
    assert result["checkpoint_audit"] == plan["parent_audit"]
    assert set(result["baselines"]) == set(extension.PANEL_BUDGETS)
    for label, baseline in result["baselines"].items():
        assert baseline["reused"] and baseline["original_only"]
        assert baseline["scorer_sha256"] == extension.digest(plan["source_sha256"])
        for key, value in extension.evaluator.summarize([
            r for _, r in extension.iter_jsonl(Path(baseline["records_path"]))]).items():
            if key != "generated_at":
                assert baseline["summary"][key] == value


@pytest.mark.parametrize("label", list(extension.PANEL_BUDGETS))
@pytest.mark.parametrize("problem", ["records_hash", "metadata", "summary_dimension", "input", "original", "budget", "checkpoint"])
def test_any_baseline_mismatch_blocks_cpu_preflight(plan, monkeypatch, label, problem):
    saved = plan["saved_baselines"][label]
    output = Path(saved["output_dir"])
    if problem in ("records_hash", "metadata"):
        records = [r for _, r in extension.iter_jsonl(output / "records.jsonl")]
        records[0]["metadata"]["eval_variant"] = "blank"
        write_rows(output / "records.jsonl", records)
        if problem == "metadata":
            saved["records_sha256"] = extension.sha256_file(output / "records.jsonl")
    elif problem == "input":
        path = Path(saved["summary"]["eval_jsonl"])
        path.write_text(path.read_text() + "\n")
    else:
        if problem == "summary_dimension":
            saved["summary"]["by_suite"] = {"fabricated": {"correct": 999}}
        elif problem == "original":
            saved["summary"]["image_variant"] = "blank"
        elif problem == "budget":
            saved["summary"]["long_max_new_tokens"] += 1
        else:
            saved["summary"]["adapter_dir"] = "older-checkpoint"
        write_json(output / "summary.json", saved["summary"])
        saved["summary_sha256"] = extension.sha256_file(output / "summary.json")
    with pytest.raises(ValueError):
        preflight(plan, monkeypatch)


@pytest.mark.parametrize("filename", ["adapter_model.safetensors", "tokenizer.json", "training_config.json", "trainer_state.json"])
def test_parent_full_hash_audit_blocks_each_changed_checkpoint_file(plan, filename):
    path = Path(extension.PARENT_CHECKPOINT) / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hashes"):
        extension.audit_parent(plan)


def test_explicit_checkpoint_step_and_fp32_scope_do_not_weaken_legacy(plan, tmp_path):
    path = tmp_path / "checkpoint-256"
    write_checkpoint(path, plan["config"], 256)
    report = extension.checkpoint_audit(path, plan["config"], parent=False, expected_step=256)
    assert report["visual_dtypes"] == {"F32": 333}
    with pytest.raises(ValueError, match="global_step"):
        extension.checkpoint_audit(path, plan["config"], parent=False)
    save_file({f"visual.{i}": torch.zeros(1, dtype=torch.bfloat16) for i in range(333)}, path / "visual_model.safetensors")
    with pytest.raises(ValueError, match="333 FP32"):
        extension.checkpoint_audit(path, plan["config"], parent=False, expected_step=256)


def train_success(plan, audit, monkeypatch, *, fail_at=None):
    calls = []

    def train(config):
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


def test_worker_calls_trainer_directly_retains_eight_checkpoints_and_history(plan, monkeypatch):
    audit = preflight(plan, monkeypatch)
    result = train_success(plan, audit, monkeypatch)
    assert result["status"] == "completed"
    assert result["checkpoint"].endswith("checkpoint-256")
    assert result["checkpoint_audit"]["checkpoint"] == result["checkpoint"]
    assert set(result["checkpoints"]) == {str(i) for i in extension.CHECKPOINT_STEPS}
    assert [r["step"] for r in result["teacher_forced_history"]] == list(extension.CHECKPOINT_STEPS)
    assert extension.read_json(Path(plan["config"]["output_dir"]) / "result.json") == result
    with pytest.raises(FileExistsError):
        extension.train_bounded.get_raw_f()(plan, audit)


def test_training_failure_preserves_partial_history_and_commits(plan, monkeypatch):
    audit = preflight(plan, monkeypatch)
    events = volumes(monkeypatch)
    with pytest.raises(RuntimeError, match="training failed"):
        train_success(plan, audit, monkeypatch, fail_at=96)
    result = extension.read_json(Path(plan["config"]["output_dir"]) / "result.json")
    assert result["status"] == "failed" and result["ended_at"]
    assert [r["step"] for r in result["teacher_forced_history"]] == [32, 64]
    assert events[-1] == "sft_runs.commit"


@pytest.mark.parametrize("problem", ["missing_eval", "bad_step", "missing_checkpoint"])
def test_training_success_requires_real_periodic_history_and_all_checkpoints(plan, tmp_path, problem):
    for step in extension.CHECKPOINT_STEPS:
        if problem == "missing_checkpoint" and step == 128:
            continue
        path = tmp_path / "checkpoints" / f"checkpoint-{step}"
        write_checkpoint(path, plan["config"], step)
        state = extension.read_json(path / "trainer_state.json")
        if problem == "missing_eval":
            state["log_history"] = [r for r in state["log_history"] if r["step"] != 32]
        elif problem == "bad_step":
            state["global_step"] = 128
        write_json(path / "trainer_state.json", state)
    with pytest.raises((ValueError, FileNotFoundError)):
        extension.training_history(tmp_path, completed=True)


def test_post_only_generates_six_panels_on_final_checkpoint_once(plan, monkeypatch):
    audit = preflight(plan, monkeypatch)
    trained = train_success(plan, audit, monkeypatch)
    events = volumes(monkeypatch)
    loads, jobs = [], []
    monkeypatch.setattr(extension.evaluator, "load_model", lambda **kw: loads.append(kw) or ("model", "processor", {}))

    def evaluate(**kwargs):
        label = kwargs["output_dir"].name
        jobs.append(label)
        args = kwargs["args"]
        assert kwargs["image_variant"] == "original"
        assert args.adapter_dir == trained["checkpoint"] and args.preserve_visual_fp32
        assert not args.do_sample and not args.enable_thinking and not args.candidate_scoring
        assert args.max_new_tokens == args.long_max_new_tokens == extension.PANEL_BUDGETS[label][2]
        write_rows(kwargs["output_dir"] / "records.jsonl", [{"id": label}])
        return plan["saved_baselines"][label]["summary"]

    monkeypatch.setattr(extension.evaluator, "run_eval_job", evaluate)
    result = extension.evaluate_bounded.get_raw_f()(plan, audit, trained)
    assert len(loads) == 1 and loads[0]["adapter_dir"].endswith("checkpoint-256")
    assert jobs == list(extension.PANEL_BUDGETS)
    assert result["status"] == "completed" and result["checkpoint_audit"] == trained["checkpoint_audit"]
    assert events.count("sft_runs.commit") == 7
    assert {"status", "checkpoint", "checkpoint_audit", "source_sha256", "panels"} <= result.keys()
    with pytest.raises(FileExistsError):
        extension.evaluate_bounded.get_raw_f()(plan, audit, trained)


def setup_coordinator(plan, monkeypatch, failure=None):
    volumes(monkeypatch)
    monkeypatch.setattr(extension.modal, "current_function_call_id", lambda: "fc-coordinator")
    directory = extension.RUN_ROOT / "pipelines" / plan["run_name"]
    plan["reservation_id"] = "reservation-test"
    write_json(directory / "launch.json", plan)
    events, calls = [], []
    results = {"preflight": {"status": "completed", "baselines": plan["saved_baselines"]},
               "training": {"status": "completed"},
               "post": {"status": "completed", "panels": plan["saved_baselines"]}}

    def spawn(name, args):
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
        call = previous.FakeCall(name, result, events, error)
        original_get = call.get

        def get(timeout):
            assert extension.read_json(directory / "result.json")["stages"][name]["call_id"] == call.object_id
            return original_get(timeout)

        call.get = get
        events.append(("spawn", name))
        calls.append(call)
        return call

    for name, function in (("preflight", "extension_preflight"), ("training", "train_bounded"), ("post", "evaluate_bounded")):
        monkeypatch.setattr(extension, function, SimpleNamespace(spawn=lambda *args, name=name: spawn(name, args)))
    return events, calls, directory


def test_coordinator_has_exactly_three_sequential_stages_and_durable_ids(plan, monkeypatch):
    events, calls, directory = setup_coordinator(plan, monkeypatch)
    result = extension.coordinate.get_raw_f()(plan)
    assert [e[1] for e in events if e[0] == "spawn"] == ["preflight", "training", "post"]
    assert [e[2] for e in events if e[0] == "get"] == [1560, 7560, 3960]
    assert not any(c.cancelled for c in calls)
    assert result["status"] == "completed" and set(result["comparison"]) == set(extension.PANEL_BUDGETS)
    assert extension.read_json(directory / "result.json") == result


@pytest.mark.parametrize("phase,error", [("preflight", RuntimeError("rescore mismatch")),
    ("preflight", TimeoutError()), ("training", TimeoutError()), ("training", "status"),
    ("post", TimeoutError()), ("post", KeyboardInterrupt())])
def test_failure_cancels_current_container_and_never_spawns_later_stages(plan, monkeypatch, phase, error):
    events, calls, directory = setup_coordinator(plan, monkeypatch, (phase, error))
    with pytest.raises(type(error) if isinstance(error, BaseException) else RuntimeError):
        extension.coordinate.get_raw_f()(plan)
    assert calls[-1].cancelled and calls[-1].object_id == f"fc-{phase}"
    assert [e[1] for e in events if e[0] == "spawn"] == ["preflight", "training", "post"][:len(calls)]
    result = extension.read_json(directory / "result.json")
    assert result["status"] == "failed" and result["cancelled_call_id"] == f"fc-{phase}"
    assert result["stages"][phase]["status"] == "failed" and result["ended_at"]


def test_reservation_and_launch_are_no_overwrite_and_dry_by_default(plan, receipts, monkeypatch):
    result = extension.launch(inputs=str(receipts), run_name="test-run")
    assert result["dry_run"] and not result["remote_calls"]
    assert not (extension.LOCAL_RUN_ROOT / "test-run").exists()
    volumes(monkeypatch)
    plan["reservation_id"] = "one-reservation"
    extension.reserve.get_raw_f()(plan)
    with pytest.raises(FileExistsError):
        extension.reserve.get_raw_f()(plan)
    changed = copy.deepcopy(plan)
    changed["reservation_id"] = "another-reservation"
    with pytest.raises(ValueError, match="reservation"):
        extension.coordinate.get_raw_f()(changed)


def test_execute_detaches_reserves_before_spawn_and_pins_receipt(plan, receipts, monkeypatch):
    events = []
    receipt = extension.LOCAL_RUN_ROOT / "test-run/launch.json"
    monkeypatch.setattr(extension.app, "run", lambda detach: nullcontext() if detach else pytest.fail("must detach"))

    def reserve(payload):
        assert extension.read_json(receipt) == payload
        events.append("reserve")

    def spawn(payload):
        assert events == ["reserve"]
        assert extension.read_json(receipt) == payload
        assert payload["config"]["train_jsonl"] == plan["parent_config"]["train_jsonl"]
        events.append("spawn")
        return SimpleNamespace(object_id="fc-extension")

    monkeypatch.setattr(extension, "reserve", SimpleNamespace(remote=reserve))
    monkeypatch.setattr(extension, "coordinate", SimpleNamespace(spawn=spawn))
    result = extension.launch(inputs=str(receipts), run_name="test-run", execute=True)
    assert events == ["reserve", "spawn"] and result["coordinator_call_id"] == "fc-extension"
    assert extension.read_json(receipt)["coordinator_call_id"] == "fc-extension"
    with pytest.raises(FileExistsError):
        extension.launch(inputs=str(receipts), run_name="test-run", execute=True)


def test_cli_is_dry_by_default_and_accepts_only_explicit_execute(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(extension, "launch", lambda **kw: calls.append(kw) or {"dry_run": not kw["execute"]})
    monkeypatch.setattr(sys, "argv", ["extension", "--run-name", extension.DEFAULT_RUN_NAME])
    extension.main()
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert calls[-1]["execute"] is False
    monkeypatch.setattr(sys, "argv", ["extension", "--execute"])
    extension.main()
    assert calls[-1]["execute"] is True


def test_runtime_rejects_source_or_baseline_drift(plan):
    for key, value in (("source_sha256", {}), ("saved_baselines", {}), ("parent_checkpoint", "older")):
        with pytest.raises(ValueError):
            extension.verify_runtime({**plan, key: value})
    changed = copy.deepcopy(plan)
    changed["config"] = asdict(replace(extension.TrainConfig(**plan["config"]), seed=42))
    changed["config_sha256"] = extension.digest(changed["config"])
    with pytest.raises(ValueError):
        extension.verify_runtime(changed)


@pytest.mark.parametrize("label", list(extension.PANEL_BUDGETS))
def test_retained_real_parent_responses_still_match_all_summary_fields(label):
    """Optional local integration: use actual saved generations, never inference."""
    root = previous.launcher.PROJECT_ROOT / "artifacts/runs/sft" / extension.PARENT_RUN
    if not (root / "post/result.json").exists():
        pytest.skip("retained parent artifacts are not part of the repository")
    post = extension.read_json(root / "post/result.json")
    launch = extension.read_json(root / "launch.json")
    saved = post["panels"][label]
    summary = saved["summary"]
    local = (extension.OLD_PANELS[label] if label in extension.OLD_PANELS
             else launch["dataset_inputs"]["new_panels"][extension.NEW_PANEL_TASKS[label]])
    rows = {r.get("id") or r.get("row_id"): r for _, r in extension.iter_jsonl(Path(local["eval_jsonl"]))}
    records_path = root / f"post/{label}-records.jsonl"
    assert extension.sha256_file(records_path) == saved["records_sha256"]
    assert extension.read_json(root / f"post/{label}-summary.json") == summary
    records = [r for _, r in extension.iter_jsonl(records_path)]
    assert len(records) == len(rows) and {r["id"] for r in records} == set(rows)
    for record in records:
        row = copy.deepcopy(rows[record["id"]])
        row.setdefault("metadata", {}).update(eval_source_sha256=summary["eval_source_sha256"],
                                               eval_set_id=summary["eval_set_id"])
        metadata = extension.evaluator.evaluation_metadata(row, image_variant="original")
        assert metadata == record["metadata"]
        assert extension.evaluator.expected_text(row) == record["expected"]
        score = extension.evaluator.score_response(record["expected"], record["response"], metadata=metadata)
        assert score == record["score"]
        record["score"] = score
    rescored = extension.evaluator.summarize(records)
    assert all(value == summary[key] for key, value in rescored.items() if key != "generated_at")
