"""CLI, remote, and series precision forwarding."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import sft.launchers.qwen_series.modal_qwen_series_eval as launcher
import sft.scripts.eval.eval_qwen_vl_adapter as evaluator


@pytest.mark.parametrize("preserve", [False, True])
def test_cli_forwards_opt_in_to_model_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool) -> None:
    argv = ["eval", "--eval-jsonl", "spatial.jsonl", "--output-dir", str(tmp_path)]
    if preserve:
        argv.append("--preserve-visual-fp32")
    monkeypatch.setattr(sys, "argv", argv)
    args = evaluator.parse_args()
    assert args.preserve_visual_fp32 is preserve
    load = Mock(return_value=(object(), object(), {}))
    job = Mock(return_value={"rows": 1, "exact_accuracy": 1})
    monkeypatch.setattr(evaluator, "load_model", load)
    monkeypatch.setattr(evaluator, "run_eval_job", job)
    evaluator.run_eval(args)
    assert load.call_args.kwargs["preserve_visual_fp32"] is preserve
    assert job.call_args.kwargs["args"].preserve_visual_fp32 is preserve


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("remote_name", ["eval_remote", "eval_h200"])
def test_remote_forwards_precision_through_subprocess_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preserve: bool, remote_name: str
) -> None:
    run = Mock()
    commit = Mock()
    monkeypatch.setattr(launcher.subprocess, "run", run)
    monkeypatch.setattr(launcher, "sft_runs", SimpleNamespace(commit=commit))
    options = {"preserve_visual_fp32": True} if preserve else {}
    getattr(launcher, remote_name).get_raw_f()(
        eval_jsonl="/data/spatial.jsonl",
        output_dir=str(tmp_path),
        adapter_dir="/runs/checkpoint-128",
        token_inventory="/data/tokens.json",
        **options,
    )
    command = run.call_args.args[0]
    assert ("--preserve-visual-fp32" in command) is preserve
    assert command[:3] == ["python", "-m", "sft.scripts.eval.eval_qwen_vl_adapter"]
    assert run.call_args.kwargs == {"check": True}
    commit.assert_called_once_with()


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("gpu,spawn", [("l40s", False), ("h200", True)])
def test_series_entrypoint_forwards_precision_without_paid_calls(monkeypatch: pytest.MonkeyPatch, preserve: bool, gpu: str, spawn: bool) -> None:
    remote = Mock(return_value={})
    spawn_call = Mock(return_value=SimpleNamespace(object_id="offline-test"))
    function = SimpleNamespace(remote=remote, spawn=spawn_call)
    monkeypatch.setattr(launcher, "eval_remote", function)
    monkeypatch.setattr(launcher, "eval_h200", function)
    monkeypatch.setattr(
        launcher,
        "upload_eval_jsonl",
        Mock(return_value=("/data/spatial.jsonl", "/data/tokens.json")),
    )
    options = {"preserve_visual_fp32": True} if preserve else {}
    launcher.main(eval_jsonl="spatial.jsonl", gpu=gpu, spawn_eval=spawn, **options)
    called, unused = (spawn_call, remote) if spawn else (remote, spawn_call)
    assert called.call_args.kwargs["preserve_visual_fp32"] is preserve
    unused.assert_not_called()
