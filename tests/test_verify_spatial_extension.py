"""Offline transport boundaries and independent rescoring of retained generations."""

import copy
import json
import socket
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest

from sft import modal_spatial_extension as extension
from sft.modal_spatial_continuation import FIXED_CONFIG
from sft.scripts import verify_spatial_extension as verify
from sft.scripts.train_trl_catan_vision import TrainConfig


RUN_NAME = "verify-extension-fixture"
PARENT = verify.LOCAL_RUN_ROOT / verify.PARENT_RUN


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(verify.json_bytes(value))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline test crossed a remote/model/training boundary")

    for cls, methods in (
        (modal.Function, ("remote", "spawn", "from_name")),
        (modal.FunctionCall, ("from_id", "get", "cancel")),
        (modal.Volume, ("from_name", "read_file", "iterdir", "reload", "commit", "batch_upload")),
        (modal.App, ("run",)),
        (socket.socket, ("connect", "connect_ex")),
    ):
        for method in methods:
            monkeypatch.setattr(cls, method, forbidden)
    monkeypatch.setattr(verify.evaluator, "load_model", forbidden)
    monkeypatch.setattr(verify.evaluator, "run_eval_job", forbidden)
    monkeypatch.setattr(extension, "run_training", forbidden)
    monkeypatch.setattr(extension.AutoTokenizer, "from_pretrained", forbidden)


@pytest.fixture
def launch(tmp_path):
    """Portable transport fixture; never claims these identities are real eval data."""
    parent = TrainConfig(**FIXED_CONFIG, seed=44, train_jsonl="/data/train.jsonl",
                         image_root="/data/images", eval_image_root="/data/images",
                         token_inventory="/data/tokens.json", eval_jsonl="/data/eval.jsonl",
                         per_device_eval_batch_size=2,
                         output_dir=f"/runs/catan-vision-sft/{verify.PARENT_RUN}")
    config = asdict(replace(parent, **extension.CONFIG_CHANGES,
                           initial_bundle=verify.PARENT_CHECKPOINT,
                           output_dir=f"/runs/catan-vision-sft/{RUN_NAME}"))
    extension.validate_config(config, RUN_NAME, asdict(parent))
    panels = {label: {"identity": {"rows": count}, "batch_size": batch,
                      "max_new_tokens": budget} for label, (count, batch, budget)
              in verify.PANEL_BUDGETS.items()}
    plan = {"schema": "catan_spatial_extension_launch/v1", "run_name": RUN_NAME,
            "config": config, "config_sha256": verify.digest(config),
            "parent_config": asdict(parent), "parent_config_sha256": verify.digest(asdict(parent)),
            "parent_checkpoint": verify.PARENT_CHECKPOINT, "panels": panels,
            "saved_baselines": {label: {} for label in panels},
            "source_sha256": {key: verify.sha256_file(verify.PROJECT_ROOT / key)
                              for key in verify.SCORER_FILES},
            "policy": {"checkpoint_steps": list(verify.CHECKPOINT_STEPS)},
            "coordinator_call_id": "fc-offline-coordinator"}
    run = tmp_path / RUN_NAME
    write_json(run / "launch.json", plan)
    return run, plan


def running_result(plan, *, status="running", phase="training"):
    return {"status": status, "phase": phase, "config": plan["config"],
            "source_sha256": plan["source_sha256"],
            "coordinator_call_id": plan["coordinator_call_id"],
            "stages": {"preflight": {"status": "completed", "call_id": "fc-offline-preflight"},
                       "training": {"status": "running", "call_id": "fc-offline-training"}}}


def volume_transport(monkeypatch, payloads, *, fail_at=None):
    """The sole allowed transport consumes whole iterators serially, with pacing."""
    events = []
    active = False

    def read_file(path):
        nonlocal active
        assert not active, "overlapping volume reads"
        active = True
        events.append(("read", path))
        try:
            value = payloads[path]
            yield value[:17]
            if fail_at == path:
                raise OSError("interrupted receipt stream")
            yield value[17:]
        finally:
            active = False

    def from_name(name, *, create_if_missing):
        assert name == "catan-sft-runs" and create_if_missing is False
        events.append(("volume", name))
        return SimpleNamespace(read_file=read_file)

    def sleep(seconds):
        assert not active and seconds == 1
        events.append(("sleep", seconds))

    monkeypatch.setattr(modal.Volume, "from_name", from_name)
    monkeypatch.setattr(verify.time, "sleep", sleep)
    return events


def remote_result(plan):
    return f"catan-vision-sft/pipelines/{plan['run_name']}/result.json"


@pytest.mark.parametrize("status,code", [("running", 0), ("failed", 1), ("completed", 0)])
def test_status_download_reads_only_coordinator_and_persists_exact_bytes(launch, monkeypatch, capsys, status, code):
    run, plan = launch
    result = running_result(plan, status=status)
    if status == "failed":
        result["error"] = "preflight rejected input"
    payload = verify.json_bytes(result)
    events = volume_transport(monkeypatch, {remote_result(plan): payload})
    assert verify.main(["--run-dir", str(run), "--download", "--status-only"]) == code
    output = json.loads(capsys.readouterr().out)
    assert output["phase"] == "training" and output["status"] == status
    assert output["coordinator_call_id"] == plan["coordinator_call_id"]
    assert output["stages"]["training"]["call_id"] == "fc-offline-training"
    assert events == [("volume", "catan-sft-runs"), ("read", remote_result(plan))]
    assert (run / "result.json").read_bytes() == payload
    assert not (run / "post").exists()


def test_running_download_reports_pending_without_claiming_verification(launch, monkeypatch, capsys):
    run, plan = launch
    result = running_result(plan)
    volume_transport(monkeypatch, {remote_result(plan): verify.json_bytes(result)})
    assert verify.main(["--run-dir", str(run), "--download"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "running"
    assert not (run / "post-verification.json").exists()


def test_local_status_reports_present_training_metrics_without_modal(launch, capsys):
    run, plan = launch
    result = running_result(plan, phase="post")
    result["stages"]["training"].update(status="completed", result={"training": {
        "metrics": {"train_loss": 0.5, "train_runtime": 100}, "eval_metrics": {"eval_loss": 0.25}},
        "teacher_forced_history": [{"step": 32, "eval_loss": 0.25}]})
    write_json(run / "result.json", result)
    assert verify.main(["--run-dir", str(run), "--status-only"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["training"]["metrics"]["train_loss"] == 0.5
    assert report["training"]["eval_metrics"]["eval_loss"] == 0.25
    assert report["training"]["teacher_forced_history"] == [{"step": 32, "eval_loss": 0.25}]


def test_real_historical_extension_launch_and_failed_status_load_offline_without_receipt_rewrites(capsys):
    run = verify.LOCAL_RUN_ROOT / "spatial-continuation-20260912-r01"
    if not (run / "launch.json").is_file() or not (run / "result.json").is_file():
        pytest.skip("historical extension receipts are local artifacts")
    paths = (run / "launch.json", run / "result.json")
    original_bytes = {path: path.read_bytes() for path in paths}
    original_hashes = {path: verify.sha256_file(path) for path in paths}
    loaded_run, plan = verify.load_launch(run)
    assert loaded_run == run.resolve()
    for key in ("config", "parent_config"):
        assert "input_mode" not in plan[key] and "max_sequence_length" not in plan[key]
        assert verify.digest(plan[key]) == plan[f"{key}_sha256"]
        assert verify.digest(extension.normalize_training_config(plan[key])) != plan[f"{key}_sha256"]
    # Failed/cancelled run status is valid receipt data, not a config validation error.
    assert verify.main(["--run-dir", str(run), "--status-only"]) == 1
    captured = capsys.readouterr()
    status = json.loads(captured.out)
    assert status["status"] == "failed" and status["phase"] == "training"
    assert "FAIL:" not in captured.err
    assert {path: path.read_bytes() for path in paths} == original_bytes
    assert {path: verify.sha256_file(path) for path in paths} == original_hashes


@pytest.mark.parametrize("problem", ["missing_dir", "missing_launch", "schema", "config", "parent", "panels"])
def test_invalid_launch_is_rejected_before_any_remote_access(launch, problem, capsys):
    run, plan = launch
    if problem == "missing_dir":
        run = run.parent / "never-created"
    elif problem == "missing_launch":
        run = run.parent / "empty"
        run.mkdir()
    else:
        if problem == "schema":
            plan["schema"] = "catan_spatial_continuation_launch/v1"
        elif problem == "config":
            plan["config"]["max_steps"] = 128
        elif problem == "parent":
            run = run.parent / verify.PARENT_RUN
            plan["run_name"] = verify.PARENT_RUN
        else:
            plan["panels"].pop("production")
        write_json(run / "launch.json", plan)
    assert verify.main(["--run-dir", str(run), "--download", "--status-only"]) == 1
    assert "FAIL:" in capsys.readouterr().err
    assert not (run / "result.json").exists()
    if problem == "missing_dir":
        assert not run.exists()


@pytest.mark.parametrize("problem", ["interrupted", "bad_json", "wrong_run", "wrong_call", "bad_stage"])
def test_failed_receipt_download_preserves_prior_bytes(launch, monkeypatch, problem, capsys):
    run, plan = launch
    original = verify.json_bytes(running_result(plan, phase="preflight"))
    (run / "result.json").write_bytes(original)
    result = copy.deepcopy(running_result(plan))
    if problem == "wrong_run":
        result["config"]["output_dir"] = "/runs/catan-vision-sft/another-run"
    elif problem == "wrong_call":
        result["coordinator_call_id"] = "fc-another"
    elif problem == "bad_stage":
        result["stages"]["training"] = None
    data = b"{incomplete" if problem == "bad_json" else verify.json_bytes(result)
    volume_transport(monkeypatch, {remote_result(plan): data},
                     fail_at=remote_result(plan) if problem == "interrupted" else None)
    assert verify.main(["--run-dir", str(run), "--download", "--status-only"]) == 1
    assert (run / "result.json").read_bytes() == original
    assert not list(run.glob(".result.json.*"))
    assert "FAIL:" in capsys.readouterr().err


@pytest.mark.parametrize("relative", ["result.json", "post/result.json", verify.STATE_PATH])
def test_symlinks_cannot_overwrite_parent_receipts(launch, tmp_path, relative):
    run, _ = launch
    historical = tmp_path / "parent-receipt.json"
    historical.write_bytes(b"historical receipt")
    target = run / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(historical)
    with pytest.raises(ValueError, match="symlink"):
        verify.atomic_write(run, relative, b"replacement")
    assert historical.read_bytes() == b"historical receipt"


def test_parent_directory_symlink_and_failed_replace_are_safe(launch, tmp_path, monkeypatch):
    run, _ = launch
    historical = tmp_path / "historical-post"
    historical.mkdir()
    (run / "post").symlink_to(historical, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verify.atomic_write(run, "post/result.json", b"replacement")
    assert not list(historical.iterdir())
    (run / "result.json").write_bytes(b"old receipt")

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(verify.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        verify.atomic_write(run, "result.json", b"new receipt")
    assert (run / "result.json").read_bytes() == b"old receipt"
    assert not list(run.glob(".result.json.*"))


@pytest.mark.parametrize("help_only", [False, True])
def test_cold_import_and_help_never_contact_modal(help_only):
    code = '''
import runpy
import socket
import sys
import modal

def forbidden(*args, **kwargs):
    raise AssertionError("network or remote compute during import/help")

socket.socket.connect = socket.socket.connect_ex = forbidden
modal.App.run = forbidden
modal.Function.remote = modal.Function.spawn = forbidden
modal.Volume.read_file = modal.Volume.commit = modal.Volume.reload = forbidden
sys.argv = ["verify", "--help"]
runpy.run_module("sft.scripts.verify_spatial_extension", run_name=NAME)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code.replace("NAME", repr("__main__" if help_only else "offline_import"))],
                            cwd=verify.PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert ("--status-only" in result.stdout) is help_only


@pytest.fixture(scope="module")
def derived_parent_256():
    """EXPLICIT TEST-ONLY derivation: parent generations are NOT extension results.

    Keep all 430 real responses/scores, input identities and parent eval-set IDs.
    Retarget checkpoint paths and create an eight-step history solely to exercise
    the verifier's checkpoint-256 contract without launching inference/training.
    """
    if not (PARENT / "post/result.json").exists():
        pytest.skip("retained parent generations are local artifacts")
    plan = extension.build_plan(extension.DEFAULT_INPUTS, RUN_NAME)
    plan["coordinator_call_id"] = "fc-offline-coordinator"
    post = copy.deepcopy(verify.read_json(PARENT / "post/result.json"))
    parent = verify.read_json(PARENT / "result.json")
    checkpoint = verify.expected_checkpoint(plan)
    post.update(checkpoint=checkpoint, source_sha256=plan["source_sha256"])
    post["checkpoint_audit"]["checkpoint"] = checkpoint
    # This history is explicit fixture data, never a runtime fallback.
    history = [{"step": step, "eval_loss": 1 / step} for step in verify.CHECKPOINT_STEPS]
    state = {"global_step": 256, "log_history": history}
    state_bytes = verify.json_bytes(state)
    post["checkpoint_audit"]["files_sha256"]["trainer_state.json"] = verify.hashlib.sha256(state_bytes).hexdigest()
    files = {}
    for label, panel in post["panels"].items():
        panel["records_path"] = f"/runs/catan-vision-sft/pipelines/{RUN_NAME}/post/{label}/records.jsonl"
        panel["scorer_sha256"] = verify.digest(plan["source_sha256"])
        summary = panel["summary"]
        summary["adapter_dir"] = summary["adapter_evidence"]["adapter_dir"] = checkpoint
        summary["adapter_evidence"]["visual_state"]["path"] = f"{checkpoint}/visual_model.safetensors"
        files[f"post/{label}-summary.json"] = verify.json_bytes(summary)
        files[f"post/{label}-records.jsonl"] = (PARENT / f"post/{label}-records.jsonl").read_bytes()
    worker = {"status": "completed", "config": plan["config"], "checkpoint": checkpoint,
              "source_sha256": plan["source_sha256"], "checkpoint_audit": post["checkpoint_audit"],
              "training": parent["stages"]["training"]["result"]["training"],
              "teacher_forced_history": history,
              "checkpoints": {str(step): {
                  "checkpoint": f"{plan['config']['output_dir']}/checkpoints/checkpoint-{step}",
                  "trainer_state_sha256": post["checkpoint_audit"]["files_sha256"]["trainer_state.json"]}
                  for step in verify.CHECKPOINT_STEPS}}
    result = running_result(plan, status="completed", phase="completed")
    result["stages"]["training"].update(status="completed", result=worker)
    result["stages"]["post"] = {"status": "completed", "call_id": "fc-offline-post", "result": post}
    files.update({"launch.json": verify.json_bytes(plan), "result.json": verify.json_bytes(result),
                  "post/result.json": verify.json_bytes(post)})
    return files, state_bytes


@pytest.fixture
def retained_run(derived_parent_256, tmp_path):
    files, _ = derived_parent_256
    run = tmp_path / RUN_NAME
    for name, data in files.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return run, verify.read_json(run / "launch.json")


def test_all_real_parent_responses_rescore_offline_with_parent_metadata(retained_run, capsys):
    run, plan = retained_run
    originals = [PARENT / "launch.json", PARENT / "result.json", *PARENT.glob("post/*")]
    original_hashes = {path: verify.sha256_file(path) for path in originals if path.is_file()}
    assert verify.main(["--run-dir", str(run)]) == 0
    report = verify.read_json(run / "post-verification.json")
    assert report["status"] == "PASS" and report["checkpoint"].endswith("checkpoint-256")
    assert sum(panel["rows"] for panel in report["panels"].values()) == 430
    for label, panel in report["panels"].items():
        assert panel["correct"] == plan["saved_baselines"][label]["summary"]["correct"]
        assert plan["saved_baselines"][label]["summary"]["eval_set_id"] == f"{verify.PARENT_RUN}-{label}"
    assert {path: verify.sha256_file(path) for path in original_hashes} == original_hashes
    assert "PASS checkpoint-256" in capsys.readouterr().out


@pytest.mark.parametrize("with_state", [False, True])
def test_completed_download_is_serial_paced_and_verifies_every_panel(retained_run, derived_parent_256, monkeypatch, tmp_path, with_state):
    source_run, plan = retained_run
    run = tmp_path / "download" / RUN_NAME
    write_json(run / "launch.json", plan)
    result = verify.read_json(source_run / "result.json")
    if with_state:
        result["stages"]["training"]["result"].pop("teacher_forced_history")
    remote = remote_result(plan).removesuffix("/result.json")
    payloads = {f"{remote}/result.json": verify.json_bytes(result),
                f"{remote}/post/result.json": (source_run / "post/result.json").read_bytes()}
    for label in verify.PANEL_BUDGETS:
        for name in ("summary.json", "records.jsonl"):
            payloads[f"{remote}/post/{label}/{name}"] = (source_run / f"post/{label}-{name}").read_bytes()
    if with_state:
        payloads[f"catan-vision-sft/{RUN_NAME}/{verify.STATE_PATH}"] = derived_parent_256[1]
    events = volume_transport(monkeypatch, payloads)
    assert verify.main(["--run-dir", str(run), "--download"]) == 0
    assert [event[1] for event in events if event[0] == "read"] == list(payloads)
    assert events[1:] == [event for index, path in enumerate(payloads)
                         for event in ([("sleep", 1)] if index else []) + [("read", path)]]
    assert (run / verify.STATE_PATH).exists() is with_state
    assert verify.read_json(run / "post-verification.json")["status"] == "PASS"


@pytest.mark.parametrize("problem", ["records_hash", "score", "target", "metadata", "new_run_id", "order", "missing", "duplicate", "response", "candidate"])
def test_saved_record_tampering_is_rejected_even_with_updated_receipt_hash(retained_run, problem):
    run, plan = retained_run
    post = verify.read_json(run / "post/result.json")
    path = run / "post/spatial-records.jsonl"
    records = [row for _, row in verify.evaluator.iter_jsonl(path)]
    if problem == "score":
        records[0]["score"]["response_normalized"] = "tampered"
    elif problem == "target":
        records[0]["expected"] = "tampered"
    elif problem in ("metadata", "records_hash"):
        records[0]["metadata"]["relationship"] = "tampered"
    elif problem == "new_run_id":
        records[0]["metadata"]["eval_set_id"] = f"{RUN_NAME}-spatial"
    elif problem == "order":
        records.reverse()
    elif problem == "missing":
        records.pop()
    elif problem == "duplicate":
        records[1] = records[0]
    elif problem == "response":
        records[0]["response"] = None
    else:
        records[0]["candidate_score"] = {"correct": True}
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    if problem != "records_hash":
        post["panels"]["spatial"]["records_sha256"] = verify.sha256_file(path)
    with pytest.raises(ValueError):
        verify.verify_panel(run, plan, post, "spatial")
    assert not (run / "post-verification.json").exists()


@pytest.mark.parametrize("problem", ["aggregate", "extra_aggregate", "budget", "bits", "reasoning", "autocast", "fp32", "visual_hash", "visual_path", "atlas", "checkpoint", "source", "eval_set_id", "conditions"])
def test_summary_settings_and_checkpoint_tampering_are_independently_rejected(retained_run, problem):
    run, plan = retained_run
    post = verify.read_json(run / "post/result.json")
    panel = post["panels"]["fullboard"]
    summary = panel["summary"]
    if problem == "aggregate":
        summary["full_board"]["occupied_layout_macro_accuracy"] = -1
    elif problem == "extra_aggregate":
        summary["candidate_rows"] = 64
    elif problem == "budget":
        summary["long_max_new_tokens"] += 1
    elif problem == "bits":
        summary["bits"] = 4
    elif problem == "reasoning":
        summary["reasoning_enabled"] = True
    elif problem == "autocast":
        summary["precision"]["generation_autocast"]["dtype"] = "torch.float32"
    elif problem == "fp32":
        summary["adapter_evidence"]["visual_precision"]["loaded_dtypes"] = {"torch.float32": 332}
    elif problem in ("visual_hash", "visual_path"):
        summary["adapter_evidence"]["visual_state"]["sha256" if problem == "visual_hash" else "path"] = "wrong"
    elif problem == "atlas":
        summary["adapter_evidence"]["semantic_tokens"]["token_ids"][0] += 1
    elif problem == "checkpoint":
        summary["adapter_dir"] = verify.PARENT_CHECKPOINT
    elif problem == "source":
        summary["eval_source_sha256"] = "wrong"
    elif problem == "eval_set_id":
        summary["eval_set_id"] = f"{RUN_NAME}-fullboard"
    else:
        panel["conditions"]["do_sample"] = True
    write_json(run / "post/fullboard-summary.json", summary)
    with pytest.raises(ValueError):
        verify.verify_panel(run, plan, post, "fullboard")


@pytest.mark.parametrize("problem", ["row_order", "pixels", "bytes", "remote_identity"])
def test_original_ordered_input_and_pixels_are_pinned(retained_run, tmp_path, problem):
    run, plan = retained_run
    post = verify.read_json(run / "post/result.json")
    local = plan["dataset_inputs"]["new_panels"]["node_tiles"]
    original = Path(local["eval_jsonl"])
    copied = tmp_path / "source.jsonl"
    data = original.read_bytes()
    if problem == "row_order":
        data = b"\n".join(reversed(data.splitlines())) + b"\n"
    elif problem == "bytes":
        data += b"\n"
    copied.write_bytes(data)
    local["eval_jsonl"] = str(copied)
    if problem == "pixels":
        images = tmp_path / "images"
        images.mkdir()
        for _, row in verify.evaluator.iter_jsonl(copied):
            reference = verify.evaluator.image_reference(row)
            destination = images / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((Path(local["image_root"]) / reference).read_bytes())
        local["image_root"] = str(images)
        assert verify.dataset_identity(str(copied), str(images)) == plan["panels"]["node_tiles"]["local_identity"]
        destination.write_bytes(destination.read_bytes() + b"changed image bytes")
    elif problem == "remote_identity":
        post["panels"]["node_tiles"]["identity"]["sha256"] = "different-upload"
    with pytest.raises(ValueError):
        verify.verify_panel(run, plan, post, "node_tiles")


@pytest.mark.parametrize("problem", ["missing_panel", "incomplete", "post_receipt", "checkpoint", "scorer", "history", "nonfinite", "audit"])
def test_pipeline_and_training_receipt_mismatches_block_pass(retained_run, problem, capsys):
    run, plan = retained_run
    result = verify.read_json(run / "result.json")
    post = result["stages"]["post"]["result"]
    worker = result["stages"]["training"]["result"]
    if problem == "missing_panel":
        post["panels"].pop("local")
    elif problem == "incomplete":
        result["stages"]["training"]["status"] = "running"
    elif problem == "post_receipt":
        post["status"] = "failed"
    elif problem == "checkpoint":
        post["checkpoint"] = verify.PARENT_CHECKPOINT
    elif problem == "scorer":
        plan["source_sha256"]["sft/board_state_readout.py"] = "changed"
        result["source_sha256"] = post["source_sha256"] = worker["source_sha256"] = plan["source_sha256"]
        write_json(run / "launch.json", plan)
    elif problem == "history":
        worker["teacher_forced_history"].pop()
    elif problem == "nonfinite":
        worker["teacher_forced_history"][0]["eval_loss"] = "NaN"
    else:
        worker["checkpoint_audit"]["files_sha256"]["adapter_model.safetensors"] = "0" * 64
    write_json(run / "result.json", result)
    if problem != "post_receipt":
        write_json(run / "post/result.json", post)
    assert verify.main(["--run-dir", str(run)]) == 1
    assert "FAIL:" in capsys.readouterr().err
    assert not (run / "post-verification.json").exists()


def test_trainer_state_hash_and_embedded_history_must_agree(retained_run, derived_parent_256):
    run, plan = retained_run
    result = verify.read_json(run / "result.json")
    audit = result["stages"]["post"]["result"]["checkpoint_audit"]
    path = run / verify.STATE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(derived_parent_256[1] + b" ")
    with pytest.raises(ValueError, match="hash"):
        verify.verify_training(run, plan, result, audit)
    path.write_bytes(derived_parent_256[1])
    result["stages"]["training"]["result"]["teacher_forced_history"][0]["eval_loss"] += 1
    with pytest.raises(ValueError, match="embedded history"):
        verify.verify_training(run, plan, result, audit)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), None, "NaN"])
def test_nonfinite_or_non_numeric_teacher_loss_is_rejected(retained_run, loss):
    run, plan = retained_run
    result = verify.read_json(run / "result.json")
    result["stages"]["training"]["result"]["teacher_forced_history"][0]["eval_loss"] = loss
    with pytest.raises(ValueError, match="non-finite"):
        verify.verify_training(run, plan, result, result["stages"]["post"]["result"]["checkpoint_audit"])


def test_matching_state_hash_cannot_hide_wrong_global_step(retained_run, derived_parent_256):
    run, plan = retained_run
    result = verify.read_json(run / "result.json")
    worker = result["stages"]["training"]["result"]
    audit = result["stages"]["post"]["result"]["checkpoint_audit"]
    state = json.loads(derived_parent_256[1])
    state["global_step"] = 128
    write_json(run / verify.STATE_PATH, state)
    audit["files_sha256"]["trainer_state.json"] = verify.sha256_file(run / verify.STATE_PATH)
    worker["checkpoint_audit"] = audit
    with pytest.raises(ValueError, match="step differs"):
        verify.verify_training(run, plan, result, audit)


@pytest.mark.parametrize("problem", ["incomplete_post", "summary", "records"])
def test_bad_post_download_stops_before_further_reads_or_pass(retained_run, monkeypatch, problem):
    run, plan = retained_run
    result = verify.read_json(run / "result.json")
    post = result["stages"]["post"]["result"]
    if problem == "incomplete_post":
        post["panels"].pop("production")
    remote = remote_result(plan).removesuffix("/result.json")
    payloads = {f"{remote}/result.json": verify.json_bytes(result),
                f"{remote}/post/result.json": verify.json_bytes(post)}
    prior = run / f"post/spatial-{'records.jsonl' if problem == 'records' else 'summary.json'}"
    original = prior.read_bytes()
    if problem != "incomplete_post":
        payloads[f"{remote}/post/spatial/summary.json"] = (
            b"{}" if problem == "summary" else (run / "post/spatial-summary.json").read_bytes())
    if problem == "records":
        payloads[f"{remote}/post/spatial/records.jsonl"] = original + b"tampered"
    events = volume_transport(monkeypatch, payloads)
    assert verify.main(["--run-dir", str(run), "--download"]) == 1
    assert [path for kind, path in events if kind == "read"] == list(payloads)
    assert prior.read_bytes() == original
    assert not (run / "post-verification.json").exists()


def test_unrelated_source_drift_does_not_require_entire_launch_source_tree(retained_run, monkeypatch):
    run, plan = retained_run
    sha256_file = verify.sha256_file

    def only_essential_source(path):
        path = Path(path)
        if path.suffix == ".py":
            assert str(path.relative_to(verify.PROJECT_ROOT)) in verify.SCORER_FILES
        return sha256_file(path)

    monkeypatch.setattr(verify, "sha256_file", only_essential_source)
    assert verify.verify_local(run, plan)["status"] == "PASS"
