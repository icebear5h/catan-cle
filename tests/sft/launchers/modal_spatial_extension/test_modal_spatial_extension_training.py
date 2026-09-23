"""Worker training, history retention, and panels."""

from pathlib import Path

import pytest

from sft.launchers.spatial import modal_spatial_extension as extension

from .support import preflight, train_success, volumes, write_checkpoint, write_json, write_rows


def test_worker_calls_trainer_directly_retains_eight_checkpoints_and_history(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_training_failure_preserves_partial_history_and_commits(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    audit = preflight(plan, monkeypatch)
    events = volumes(monkeypatch)
    with pytest.raises(RuntimeError, match="training failed"):
        train_success(plan, audit, monkeypatch, fail_at=96)
    result = extension.read_json(Path(plan["config"]["output_dir"]) / "result.json")
    assert result["status"] == "failed" and result["ended_at"]
    assert [r["step"] for r in result["teacher_forced_history"]] == [32, 64]
    assert events[-1] == "sft_runs.commit"


@pytest.mark.parametrize("problem", ["missing_eval", "bad_step", "missing_checkpoint"])
def test_training_success_requires_real_periodic_history_and_all_checkpoints(plan: dict[str, object], tmp_path: Path, problem: str) -> None:
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


def test_post_only_generates_six_panels_on_final_checkpoint_once(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    audit = preflight(plan, monkeypatch)
    trained = train_success(plan, audit, monkeypatch)
    events = volumes(monkeypatch)
    loads: list[dict[str, object]] = []
    jobs: list[str] = []
    monkeypatch.setattr(extension.evaluator, "load_model", lambda **kw: loads.append(kw) or ("model", "processor", {}))

    def evaluate(**kwargs: object) -> dict[str, object]:
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
