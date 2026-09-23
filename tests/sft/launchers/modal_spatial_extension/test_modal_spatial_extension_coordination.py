"""Stage ordering, reservation, and CLI execution."""
import copy
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft.launchers.spatial import modal_spatial_extension as extension

from .support import setup_coordinator, volumes


def test_coordinator_has_exactly_three_sequential_stages_and_durable_ids(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
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
def test_failure_cancels_current_container_and_never_spawns_later_stages(
    plan: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    error: BaseException | str,
) -> None:
    events, calls, directory = setup_coordinator(plan, monkeypatch, (phase, error))
    with pytest.raises(type(error) if isinstance(error, BaseException) else RuntimeError):
        extension.coordinate.get_raw_f()(plan)
    assert calls[-1].cancelled and calls[-1].object_id == f"fc-{phase}"
    assert [e[1] for e in events if e[0] == "spawn"] == ["preflight", "training", "post"][:len(calls)]
    result = extension.read_json(directory / "result.json")
    assert result["status"] == "failed" and result["cancelled_call_id"] == f"fc-{phase}"
    assert result["stages"][phase]["status"] == "failed" and result["ended_at"]


def test_reservation_and_launch_are_no_overwrite_and_dry_by_default(plan: dict[str, object], receipts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_execute_detaches_reserves_before_spawn_and_pins_receipt(plan: dict[str, object], receipts: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    receipt = extension.LOCAL_RUN_ROOT / "test-run/launch.json"
    monkeypatch.setattr(extension.app, "run", lambda detach: nullcontext() if detach else pytest.fail("must detach"))

    def reserve(payload: dict[str, object]) -> None:
        assert extension.read_json(receipt) == payload
        events.append("reserve")

    def spawn(payload: dict[str, object]) -> SimpleNamespace:
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


def test_cli_is_dry_by_default_and_accepts_only_explicit_execute(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls = []
    monkeypatch.setattr(extension, "launch", lambda **kw: calls.append(kw) or {"dry_run": not kw["execute"]})
    monkeypatch.setattr(sys, "argv", ["extension", "--run-name", extension.DEFAULT_RUN_NAME])
    extension.main()
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert calls[-1]["execute"] is False
    monkeypatch.setattr(sys, "argv", ["extension", "--execute"])
    extension.main()
    assert calls[-1]["execute"] is True


def test_runtime_rejects_source_or_baseline_drift(plan: dict[str, object]) -> None:
    for key, value in (("source_sha256", {}), ("saved_baselines", {}), ("parent_checkpoint", "older")):
        with pytest.raises(ValueError):
            extension.verify_runtime({**plan, key: value})
    changed = copy.deepcopy(plan)
    changed["config"] = asdict(replace(extension.TrainConfig(**plan["config"]), seed=42))
    changed["config_sha256"] = extension.digest(changed["config"])
    with pytest.raises(ValueError):
        extension.verify_runtime(changed)
