"""Symlink, replace, and cold-import safety."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest

from sft.scripts.eval import verify_spatial_extension as verify

from .support import remote_result, running_result, volume_transport


@pytest.mark.parametrize("problem", ["interrupted", "bad_json", "wrong_run", "wrong_call", "bad_stage"])
def test_failed_receipt_download_preserves_prior_bytes(launch: tuple[Path, dict[str, object]], monkeypatch: pytest.MonkeyPatch, problem: str, capsys: pytest.CaptureFixture[str]) -> None:
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
def test_symlinks_cannot_overwrite_parent_receipts(
    launch: tuple[Path, dict[str, object]], tmp_path: Path, relative: str
) -> None:
    run, _ = launch
    historical = tmp_path / "parent-receipt.json"
    historical.write_bytes(b"historical receipt")
    target = run / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(historical)
    with pytest.raises(ValueError, match="symlink"):
        verify.atomic_write(run, relative, b"replacement")
    assert historical.read_bytes() == b"historical receipt"


def test_parent_directory_symlink_and_failed_replace_are_safe(launch: tuple[Path, dict[str, object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = launch
    historical = tmp_path / "historical-post"
    historical.mkdir()
    (run / "post").symlink_to(historical, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        verify.atomic_write(run, "post/result.json", b"replacement")
    assert not list(historical.iterdir())
    (run / "result.json").write_bytes(b"old receipt")

    def fail_replace(*args: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(verify.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        verify.atomic_write(run, "result.json", b"new receipt")
    assert (run / "result.json").read_bytes() == b"old receipt"
    assert not list(run.glob(".result.json.*"))


@pytest.mark.parametrize("help_only", [False, True])
def test_cold_import_and_help_never_contact_modal(help_only: bool) -> None:
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
runpy.run_module("sft.scripts.eval.verify_spatial_extension", run_name=NAME)
'''
    result = subprocess.run([sys.executable, "-B", "-c", code.replace("NAME", repr("__main__" if help_only else "offline_import"))],
                            cwd=verify.PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert ("--status-only" in result.stdout) is help_only
