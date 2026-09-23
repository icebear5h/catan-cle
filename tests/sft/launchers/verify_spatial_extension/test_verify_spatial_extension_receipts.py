"""Receipt, trainer-state, and drift agreement."""

import json
from pathlib import Path

import pytest

from sft.scripts.eval import verify_spatial_extension as verify

from .support import remote_result, volume_transport, write_json


@pytest.mark.parametrize("problem", ["missing_panel", "incomplete", "post_receipt", "checkpoint", "scorer", "history", "nonfinite", "audit"])
def test_pipeline_and_training_receipt_mismatches_block_pass(retained_run: tuple[Path, dict[str, object]], problem: str, capsys: pytest.CaptureFixture[str]) -> None:
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


def test_trainer_state_hash_and_embedded_history_must_agree(retained_run: tuple[Path, dict[str, object]], derived_parent_256: tuple[dict[str, bytes], bytes]) -> None:
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
def test_nonfinite_or_non_numeric_teacher_loss_is_rejected(
    retained_run: tuple[Path, dict[str, object]], loss: float | str | None
) -> None:
    run, plan = retained_run
    result = verify.read_json(run / "result.json")
    result["stages"]["training"]["result"]["teacher_forced_history"][0]["eval_loss"] = loss
    with pytest.raises(ValueError, match="non-finite"):
        verify.verify_training(run, plan, result, result["stages"]["post"]["result"]["checkpoint_audit"])


def test_matching_state_hash_cannot_hide_wrong_global_step(retained_run: tuple[Path, dict[str, object]], derived_parent_256: tuple[dict[str, bytes], bytes]) -> None:
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
def test_bad_post_download_stops_before_further_reads_or_pass(retained_run: tuple[Path, dict[str, object]], monkeypatch: pytest.MonkeyPatch, problem: str) -> None:
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


def test_unrelated_source_drift_does_not_require_entire_launch_source_tree(retained_run: tuple[Path, dict[str, object]], monkeypatch: pytest.MonkeyPatch) -> None:
    run, plan = retained_run
    sha256_file = verify.sha256_file

    def only_essential_source(path: Path | str) -> str:
        path = Path(path)
        if path.suffix == ".py":
            assert str(path.relative_to(verify.PROJECT_ROOT)) in verify.SCORER_FILES
        return sha256_file(path)

    monkeypatch.setattr(verify, "sha256_file", only_essential_source)
    assert verify.verify_local(run, plan)["status"] == "PASS"
