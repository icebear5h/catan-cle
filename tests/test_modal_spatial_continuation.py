"""Offline only: every Modal/upload boundary is blocked or a mock transport."""

import copy
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest
import torch
from safetensors.torch import save_file

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft import modal_spatial_continuation as launcher
from sft.board_state_readout import board_keys
from sft.scripts.train_trl_catan_vision import TrainConfig


FULL_BOARD_ANSWER = "; ".join(
    f"{key} " + ("wood 6" if key.startswith("<T") else "3:1 port" if key.startswith("<P")
                else "<T00>" if key == "robber" else "empty") for key in board_keys()
)


@pytest.fixture(autouse=True)
def no_remote(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline test crossed a Modal or upload boundary")

    for cls, names in ((modal.Function, ("spawn", "remote")),
                       (modal.Volume, ("reload", "commit", "batch_upload")),
                       (modal.App, ("run",))):
        for name in names:
            monkeypatch.setattr(cls, name, forbidden)
    monkeypatch.setattr(launcher, "upload_training_bundle", forbidden)
    monkeypatch.setattr(launcher, "upload_eval_jsonl", forbidden)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(index, family="node_tiles", task=None):
    task = task or {"directions": "node_direction_yes", "adjacency_connectivity": "node_adjacent_no"}.get(family, family)
    answer = "<T00>"
    if task == "full_board_readout":
        answer = FULL_BOARD_ANSWER
    return {"row_id": f"row-{index}", "images": ["image.png"], "task_type": task,
            "training_family": family, "metadata": {"task_type": task, "query_id": str(index)},
            "messages": [{"role": "user", "content": "<image>\nAnswer only."},
                         {"role": "assistant", "content": answer}]}


def training_rows():
    families = [k for k in launcher.FAMILY_STEPS if k != "full_board_readout"] + ["full_board_readout", "full_board_readout"]
    return [row(step * 8 + i, family) for step, family in enumerate(families * 16) for i in range(8)]


@pytest.fixture
def plan(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    local = tmp_path / "receipts"
    root.mkdir()
    local.mkdir()
    monkeypatch.setattr(launcher, "RUN_ROOT", root)
    monkeypatch.setattr(launcher, "LOCAL_RUN_ROOT", local)
    monkeypatch.setattr(launcher, "PARENT_CHECKPOINT", str(root / "parent/checkpoints/checkpoint-128"))
    monkeypatch.setattr(launcher, "source_hashes", lambda: {"scorer.py": "version-one"})
    images = tmp_path / "images"
    images.mkdir()
    (images / "image.png").write_bytes(b"unit-test-image-bytes")
    train_path = tmp_path / "train.jsonl"
    write_rows(train_path, training_rows())
    panels = {}
    tasks = {"spatial": "directions", "fullboard": "full_board_readout", "node_tiles": "node_tiles",
             "paths": "shortest_node_path", "local": "local_node_tiles", "production": "dice_production"}
    for label, (count, batch, budget) in launcher.PANEL_BUDGETS.items():
        path = tmp_path / f"{label}.jsonl"
        write_rows(path, [row(f"{label}-{i}", tasks[label]) for i in range(count)])
        panels[label] = {"eval_jsonl": str(path), "image_root": str(images), "batch_size": batch,
                         "max_new_tokens": budget, "identity": launcher.dataset_identity(str(path), str(images))}
    inventory = tmp_path / "tokens.json"
    write_json(inventory, semantic_recognition_token_inventory())
    config = asdict(TrainConfig(train_jsonl=str(train_path), image_root=str(images), token_inventory=str(inventory),
                               output_dir=str(root / "test-run"), eval_jsonl=panels["fullboard"]["eval_jsonl"],
                               eval_image_root=str(images), per_device_eval_batch_size=2,
                               initial_bundle=launcher.PARENT_CHECKPOINT, **launcher.FIXED_CONFIG))
    return {"run_name": "test-run", "config": config, "parent_config": {**config, "seed": 44},
            "config_sha256": launcher.digest(config),
            "train_identity": launcher.dataset_identity(str(train_path), str(images)),
            "panels": panels, "mixture": launcher.validate_mixture(training_rows()),
            "source_sha256": launcher.source_hashes(), "reservation_id": "reservation-test",
            "saved_baselines": {"spatial": {}, "fullboard": {}}, "legacy_pilot_sha256": "legacy"}


def panel_result(plan, label, correct=0):
    count = launcher.PANEL_BUDGETS[label][0]
    summary = {"rows": count, "attempted": count, "correct": correct, "exact_accuracy": correct / count}
    if label == "fullboard":
        summary["full_board"] = {"board_exact": correct, "occupied_layout_macro_accuracy": 0.75,
                                 "by_layout": {"layout": {"boards": count}}}
    return {"summary": summary, "identity": copy.deepcopy(plan["panels"][label]["identity"]),
            "scorer_sha256": launcher.digest(plan["source_sha256"]),
            "conditions": launcher.evaluation_conditions(label)}


def fake_volumes(monkeypatch, events):
    for name in ("hf_cache", "sft_data", "sft_runs"):
        monkeypatch.setattr(launcher, name, SimpleNamespace(
            reload=lambda name=name: events.append(f"{name}.reload"),
            commit=lambda name=name: events.append(f"{name}.commit")))


def test_approved_config_budgets_and_new_output(plan):
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
def test_rejects_unapproved_training(plan, change):
    with pytest.raises(ValueError):
        launcher.validate_config({**plan["config"], **change}, "test-run")


def test_historical_missing_input_defaults_validate_without_rehashing(plan):
    legacy = {k: v for k, v in plan["config"].items()
              if k not in ("input_mode", "max_sequence_length")}
    original = copy.deepcopy(legacy)
    original_digest = launcher.digest(legacy)
    config = launcher.validate_config(legacy, "test-run")
    assert config.input_mode == "vision" and config.max_sequence_length is None
    launcher.verify_runtime({**plan, "config": legacy, "config_sha256": original_digest})
    assert legacy == original and launcher.digest(legacy) == original_digest
    assert launcher.digest(asdict(config)) != original_digest


def test_exact_mixture_owns_128_homogeneous_steps():
    rows = training_rows()
    result = launcher.validate_mixture(rows)
    assert result["steps"] == 128 and result["rows"] == 1024
    assert result["family_steps"] == launcher.FAMILY_STEPS
    assert result["family_step_share"]["full_board_readout"] == 0.25
    rows[0], rows[8] = rows[8], rows[0]
    with pytest.raises(ValueError, match="homogeneous"):
        launcher.validate_mixture(rows)


@pytest.mark.parametrize("problem", ["count", "ids", "quota", "task", "metadata", "curriculum"])
def test_bad_mixture_is_rejected(problem):
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


def test_identity_preserves_prompts_metadata_order_and_pixels_through_upload(plan, tmp_path):
    panel = plan["panels"]["paths"]
    rows = [r for _, r in launcher.iter_jsonl(Path(panel["eval_jsonl"]))]
    image = Path(panel["image_root"]) / "image.png"
    for r in rows:
        r["image"] = str(image)
        r.pop("images")
        r["metadata"].update(eval_source_sha256=panel["identity"]["sha256"], eval_set_id="uploaded")
    uploaded = tmp_path / "uploaded.jsonl"
    write_rows(uploaded, rows)
    actual = launcher.dataset_identity(str(uploaded), None)
    launcher.check_identity(actual, panel["identity"])
    assert actual["sha256"] != panel["identity"]["sha256"]
    for mutation in ("prompt", "metadata", "order", "pixels"):
        changed = copy.deepcopy(rows)
        if mutation == "prompt":
            changed[0]["messages"][0]["content"] += " changed"
        elif mutation == "metadata":
            changed[0]["metadata"]["query_id"] = "different"
        elif mutation == "order":
            changed.reverse()
        else:
            image.write_bytes(b"changed pixels")
        write_rows(uploaded, changed)
        with pytest.raises(ValueError, match="identity changed"):
            launcher.check_identity(launcher.dataset_identity(str(uploaded), None), panel["identity"])


def test_token_lengths_not_char_lengths_and_separate_exposure():
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens: list(range(len(text.split()))))
    rows = [row(0, "directions"), row(1, "full_board_readout")]
    rows[0]["messages"][1]["content"] = "a" * 100
    rows[1]["messages"][1]["content"] = " ".join(["token"] * 1279)
    audit = launcher.completion_audit(rows, tokenizer)
    assert audit["max_tokens"] == {"directions": 1, "full_board_readout": 1279}
    assert audit["completion_token_share"]["full_board_readout"] == 1279 / 1280
    for family, limit in (("directions", 16), ("node_tiles", 128), ("full_board_readout", 1280)):
        example = row(1, family)
        example["messages"][1]["content"] = " ".join(["token"] * limit)
        with pytest.raises(ValueError, match="reaches/exceeds"):
            launcher.completion_audit([example], tokenizer)


def test_path_long_branch_receives_the_same_128_token_budget(plan):
    path = row(0, "shortest_node_path")
    path["messages"][1]["content"] = " ".join(f"<N{i:02d}>" for i in range(12))
    assert launcher.evaluator.is_long_answer(path)
    args = launcher.panel_args(plan, launcher.PARENT_CHECKPOINT, "paths")
    assert args.long_max_new_tokens == args.max_new_tokens == 128
    assert args.long_batch_size == args.batch_size == 16


def test_comparison_checks_inputs_scorer_conditions_and_retention(plan):
    before = {label: panel_result(plan, label) for label in launcher.PANEL_BUDGETS}
    after = {label: panel_result(plan, label, correct=2) for label in launcher.PANEL_BUDGETS}
    comparison = launcher.compare_panels(before, after)
    assert comparison["fullboard"]["board_exact_after"] == 2
    assert comparison["fullboard"]["occupied_layout_macro_accuracy_before"] == 0.75
    assert all(v["correct_after"] == 2 for v in comparison.values())
    for field, replacement in (("scorer_sha256", "different"), ("conditions", {}),
                               ("identity", {"rows": 64, "content_sha256": "different", "unique_images": 1})):
        changed = copy.deepcopy(after)
        changed["paths"][field] = replacement
        with pytest.raises(ValueError):
            launcher.compare_panels(before, changed)
    with pytest.raises(ValueError, match="six"):
        launcher.compare_panels({}, after)


class FakeCall:
    def __init__(self, name, result, events, error=None):
        self.object_id = f"fc-{name}"
        self.result = result
        self.events = events
        self.error = error
        self.cancelled = False

    def get(self, timeout):
        self.events.append(("get", self.object_id, timeout))
        if self.error:
            raise self.error
        return self.result

    def cancel(self, terminate_containers=False):
        assert terminate_containers
        self.cancelled = True
        self.events.append(("cancel", self.object_id))


def coordinator_setup(plan, monkeypatch, failure=None):
    events, calls = [], []
    fake_volumes(monkeypatch, events)
    monkeypatch.setattr(launcher.modal, "current_function_call_id", lambda: "fc-coordinator")
    directory = launcher.RUN_ROOT / "pipelines" / plan["run_name"]
    write_json(directory / "launch.json", plan)
    before = {label: panel_result(plan, label) for label in ("spatial", "fullboard")}

    def spawn(name, result):
        assert (directory / "launch.json").exists()
        assert not Path(plan["config"]["output_dir"]).exists()
        current = launcher.read_json(directory / "result.json")
        assert current["phase"] == name
        events.append(("spawn", name))
        error = failure[1] if failure and failure[0] == name else None
        if error == "status":
            result, error = {"status": "failed"}, None
        call = FakeCall(name, result, events, error)
        calls.append(call)
        return call

    monkeypatch.setattr(launcher, "continuation_preflight", SimpleNamespace(spawn=lambda p: spawn("preflight", {"status": "completed", "baselines": before})))
    monkeypatch.setattr(launcher, "train_bounded", SimpleNamespace(spawn=lambda p, a: spawn("training", {"status": "completed"})))
    monkeypatch.setattr(launcher, "evaluate_bounded", SimpleNamespace(spawn=lambda p, stage, a: spawn(stage,
        {"status": "completed", "panels": {label: panel_result(plan, label)
         for label in (launcher.NEW_LABELS if stage == "pre" else launcher.PANEL_BUDGETS)}})))
    return events, calls, directory


def test_coordinator_orders_preflight_parent_eval_training_post_and_commits(plan, monkeypatch):
    events, calls, directory = coordinator_setup(plan, monkeypatch)
    result = launcher.coordinate.get_raw_f()(plan)
    assert result["status"] == "completed"
    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "spawn"] == ["preflight", "pre", "training", "post"]
    assert not any(call.cancelled for call in calls)
    assert launcher.read_json(directory / "result.json")["comparison"] == result["comparison"]
    assert launcher.read_json(directory / "launch.json")["coordinator_call_id"] == "fc-coordinator"
    assert events.count("sft_runs.reload") == events.count("sft_runs.commit")
    assert all(stage["started_at"] and stage["ended_at"] and stage["call_id"] for stage in result["stages"].values())
    with pytest.raises(FileExistsError):
        launcher.coordinate.get_raw_f()(plan)


@pytest.mark.parametrize("phase,error", [("preflight", RuntimeError("bad checkpoint")),
                                         ("pre", TimeoutError("budget")),
                                         ("training", RuntimeError("failed training")),
                                         ("post", KeyboardInterrupt()), ("pre", "status")])
def test_coordinator_propagates_failure_cancels_and_never_starts_later_stage(plan, monkeypatch, phase, error):
    events, calls, directory = coordinator_setup(plan, monkeypatch, (phase, error))
    with pytest.raises(type(error) if isinstance(error, BaseException) else RuntimeError):
        launcher.coordinate.get_raw_f()(plan)
    assert calls[-1].object_id == f"fc-{phase}" and calls[-1].cancelled
    result = launcher.read_json(directory / "result.json")
    assert result["status"] == "failed" and result["phase"] == phase
    assert result["cancelled_call_id"] == calls[-1].object_id
    assert result["stages"][phase]["status"] == "failed" and result["ended_at"]


def test_reservation_is_durable_and_never_overwrites(plan, monkeypatch):
    events = []
    fake_volumes(monkeypatch, events)
    reserve = launcher.reserve.get_raw_f()
    assert reserve(plan)["status"] == "reserved"
    destination = launcher.RUN_ROOT / "pipelines/test-run/launch.json"
    assert launcher.read_json(destination)["reservation_id"] == plan["reservation_id"]
    with pytest.raises(FileExistsError):
        reserve(plan)
    Path(plan["config"]["output_dir"]).mkdir()
    with pytest.raises(FileExistsError):
        reserve(plan)
    assert events.count("sft_runs.commit") == 1


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_eval_loads_once_fp32_all_panels_original_reloads_and_commits(plan, monkeypatch, stage):
    events, loads, jobs = [], [], []
    fake_volumes(monkeypatch, events)
    checkpoint = {"checkpoint": launcher.PARENT_CHECKPOINT}
    monkeypatch.setattr(launcher, "checkpoint_audit", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(launcher.evaluator, "load_model", lambda **kwargs: loads.append(kwargs) or ("model", "processor", {}))

    def evaluate(**kwargs):
        assert events[:3] == ["hf_cache.reload", "sft_data.reload", "sft_runs.reload"]
        assert kwargs["model"] == "model" and kwargs["image_variant"] == "original"
        label = kwargs["output_dir"].name
        jobs.append(label)
        args = kwargs["args"]
        assert args.preserve_visual_fp32 and args.long_max_new_tokens == args.max_new_tokens
        write_rows(kwargs["output_dir"] / "records.jsonl", [{"id": label}])
        return panel_result(plan, label)["summary"]

    monkeypatch.setattr(launcher.evaluator, "run_eval_job", evaluate)
    audit = {"checkpoint": checkpoint, "panels": {k: p["identity"] for k, p in plan["panels"].items()}}
    evaluate_worker = launcher.evaluate_bounded.get_raw_f()
    result = evaluate_worker(plan, stage, audit)
    assert result["status"] == "completed"
    assert jobs == list(launcher.NEW_LABELS if stage == "pre" else launcher.PANEL_BUDGETS)
    assert len(loads) == 1 and loads[0]["preserve_visual_fp32"] is True
    assert events.count("sft_runs.commit") == len(jobs) + 1
    assert all(Path(p["records_path"]).is_file() for p in result["panels"].values())
    with pytest.raises(FileExistsError):
        evaluate_worker(plan, stage, audit)
    assert len(loads) == 1


def test_training_reuses_pilot_locally_and_checks_train_hash(plan, monkeypatch):
    events, trained = [], []
    fake_volumes(monkeypatch, events)
    checkpoint = {"checkpoint": launcher.PARENT_CHECKPOINT}
    monkeypatch.setattr(launcher, "checkpoint_audit", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(launcher, "pilot_train", SimpleNamespace(local=lambda payload: trained.append(payload) or {"status": "completed"}))
    audit = {"checkpoint": checkpoint, "train_identity": plan["train_identity"],
             "teacher_eval_identity": plan["panels"]["fullboard"]["identity"]}
    result = launcher.train_bounded.get_raw_f()(plan, audit)
    assert result["status"] == "completed" and trained == [plan["config"]]
    assert events[:3] == ["hf_cache.reload", "sft_data.reload", "sft_runs.reload"]
    assert events[-1] == "sft_runs.commit"
    path = Path(plan["config"]["train_jsonl"])
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="bytes changed"):
        launcher.train_bounded.get_raw_f()(plan, audit)
    assert len(trained) == 1


def test_dry_run_has_no_remote_or_upload_side_effects(plan, monkeypatch):
    monkeypatch.setattr(launcher, "build_plan", lambda *args: plan)
    result = launcher.launch(inputs="/dataset_inputs.json", run_name="test-run")
    assert result["dry_run"] and result["remote_calls"] is False
    assert not (launcher.LOCAL_RUN_ROOT / "test-run").exists()


def test_execute_reserves_before_uploads_and_writes_receipt_before_spawn(plan, monkeypatch):
    events = []
    monkeypatch.setattr(launcher, "build_plan", lambda *args: copy.deepcopy(plan))
    monkeypatch.setattr(launcher.app, "run", lambda detach: nullcontext() if detach else pytest.fail("must detach"))
    receipt = launcher.LOCAL_RUN_ROOT / "test-run/launch.json"

    def reserve(payload):
        assert launcher.read_json(receipt)["status"] == "reserved"
        events.append("reserve")

    def upload_training(*args, **kwargs):
        assert events == ["reserve"]
        assert kwargs["require_curriculum"] is False
        assert str(kwargs["eval_jsonl"]) == plan["panels"]["fullboard"]["eval_jsonl"]
        events.append("upload_training")
        return "/data/train", "/data/teacher_eval", "/data/images", "/data/tokens", {}, {}

    def upload_eval(path, remote_dir, **kwargs):
        events.append(remote_dir.rsplit("/", 1)[-1])
        return f"/data/{remote_dir}/eval.jsonl", None

    def spawn(payload):
        ready = launcher.read_json(receipt)
        assert ready["status"] == "ready" and ready["config"] == payload["config"]
        assert all(panel["image_root"] is None for panel in payload["panels"].values())
        events.append("spawn")
        return SimpleNamespace(object_id="fc-coordinator")

    monkeypatch.setattr(launcher, "reserve", SimpleNamespace(remote=reserve))
    monkeypatch.setattr(launcher, "upload_training_bundle", upload_training)
    monkeypatch.setattr(launcher, "upload_eval_jsonl", upload_eval)
    monkeypatch.setattr(launcher, "coordinate", SimpleNamespace(spawn=spawn))
    result = launcher.launch(inputs="/dataset_inputs.json", run_name="test-run", execute=True)
    assert events == ["reserve", "upload_training", *launcher.PANEL_BUDGETS, "spawn"]
    assert result["coordinator_call_id"] == "fc-coordinator"
    assert launcher.read_json(receipt)["status"] == "spawned"
    with pytest.raises(FileExistsError):
        launcher.launch(inputs="/dataset_inputs.json", run_name="test-run", execute=True)


def test_manifest_uses_latest_evaluated_checkpoint_and_checks_fixed_baseline_hashes(plan, tmp_path, monkeypatch):
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


def test_saved_baseline_is_independently_rescored_and_identity_checked(plan, tmp_path, monkeypatch):
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


def test_cpu_preflight_checks_tokenizers_lengths_identities_and_commits(plan, monkeypatch):
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
    sources = []

    def load(source, local_files_only):
        assert local_files_only
        sources.append(source)
        return tokenizer

    monkeypatch.setattr(launcher.AutoTokenizer, "from_pretrained", load)
    is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda path: str(path) == launcher.SNAPSHOT or is_dir(path))
    monkeypatch.setattr(launcher, "matched_baseline", lambda label, *args: panel_result(plan, label))
    plan["legacy_pilot_sha256"] = launcher.sha256_file(launcher.PROJECT_ROOT / "sft/modal_full_board_pilot.py")
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


def test_runtime_rejects_source_config_budget_and_baseline_drift(plan):
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


def test_checkpoint_audit_checks_files_hash_profile_rows_rank_and_fp32(plan, monkeypatch):
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
