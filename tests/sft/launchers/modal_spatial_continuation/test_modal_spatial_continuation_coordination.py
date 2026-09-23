"""Stage ordering, reservation, and execution receipts."""

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft.launchers.spatial import modal_spatial_continuation as launcher

from .support import coordinator_setup, fake_volumes, panel_result, write_rows


def test_coordinator_orders_preflight_parent_eval_training_post_and_commits(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
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
def test_coordinator_propagates_failure_cancels_and_never_starts_later_stage(
    plan: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    error: BaseException | str,
) -> None:
    events, calls, directory = coordinator_setup(plan, monkeypatch, (phase, error))
    with pytest.raises(type(error) if isinstance(error, BaseException) else RuntimeError):
        launcher.coordinate.get_raw_f()(plan)
    assert calls[-1].object_id == f"fc-{phase}" and calls[-1].cancelled
    result = launcher.read_json(directory / "result.json")
    assert result["status"] == "failed" and result["phase"] == phase
    assert result["cancelled_call_id"] == calls[-1].object_id
    assert result["stages"][phase]["status"] == "failed" and result["ended_at"]


def test_reservation_is_durable_and_never_overwrites(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []
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
def test_eval_loads_once_fp32_all_panels_original_reloads_and_commits(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    events, loads, jobs = [], [], []
    fake_volumes(monkeypatch, events)
    checkpoint = {"checkpoint": launcher.PARENT_CHECKPOINT}
    monkeypatch.setattr(launcher, "checkpoint_audit", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(launcher.evaluator, "load_model", lambda **kwargs: loads.append(kwargs) or ("model", "processor", {}))

    def evaluate(**kwargs: object) -> dict[str, object]:
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


def test_training_reuses_pilot_locally_and_checks_train_hash(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_dry_run_has_no_remote_or_upload_side_effects(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "build_plan", lambda *args: plan)
    result = launcher.launch(inputs="/dataset_inputs.json", run_name="test-run")
    assert result["dry_run"] and result["remote_calls"] is False
    assert not (launcher.LOCAL_RUN_ROOT / "test-run").exists()


def test_execute_reserves_before_uploads_and_writes_receipt_before_spawn(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    monkeypatch.setattr(launcher, "build_plan", lambda *args: copy.deepcopy(plan))
    monkeypatch.setattr(launcher.app, "run", lambda detach: nullcontext() if detach else pytest.fail("must detach"))
    receipt = launcher.LOCAL_RUN_ROOT / "test-run/launch.json"

    def reserve(payload: dict[str, object]) -> None:
        assert launcher.read_json(receipt)["status"] == "reserved"
        events.append("reserve")

    def upload_training(*args: object, **kwargs: object) -> tuple[object, ...]:
        assert events == ["reserve"]
        assert kwargs["require_curriculum"] is False
        assert str(kwargs["eval_jsonl"]) == plan["panels"]["fullboard"]["eval_jsonl"]
        events.append("upload_training")
        return "/data/train", "/data/teacher_eval", "/data/images", "/data/tokens", {}, {}

    def upload_eval(
        path: str, remote_dir: str, **kwargs: object
    ) -> tuple[str, None]:
        events.append(remote_dir.rsplit("/", 1)[-1])
        return f"/data/{remote_dir}/eval.jsonl", None

    def spawn(payload: dict[str, object]) -> SimpleNamespace:
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
