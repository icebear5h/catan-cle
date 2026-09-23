"""Status reads, downloads, and offline launch handling."""
import json
from pathlib import Path

import pytest

from sft.launchers.spatial import modal_spatial_extension as extension
from sft.scripts.eval import verify_spatial_extension as verify

from .support import remote_result, running_result, volume_transport, write_json


@pytest.mark.parametrize("status,code", [("running", 0), ("failed", 1), ("completed", 0)])
def test_status_download_reads_only_coordinator_and_persists_exact_bytes(launch: tuple[Path, dict[str, object]], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: str, code: int) -> None:
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


def test_running_download_reports_pending_without_claiming_verification(launch: tuple[Path, dict[str, object]], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    run, plan = launch
    result = running_result(plan)
    volume_transport(monkeypatch, {remote_result(plan): verify.json_bytes(result)})
    assert verify.main(["--run-dir", str(run), "--download"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "running"
    assert not (run / "post-verification.json").exists()


def test_local_status_reports_present_training_metrics_without_modal(launch: tuple[Path, dict[str, object]], capsys: pytest.CaptureFixture[str]) -> None:
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


def test_real_historical_extension_launch_and_failed_status_load_offline_without_receipt_rewrites(capsys: pytest.CaptureFixture[str]) -> None:
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
def test_invalid_launch_is_rejected_before_any_remote_access(launch: tuple[Path, dict[str, object]], problem: str, capsys: pytest.CaptureFixture[str]) -> None:
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
