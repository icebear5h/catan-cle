"""Shared helpers for offline only: every modal/upload boundary is blocked or a mock transport."""

import copy
import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft.board_state_readout import board_keys
from sft.launchers.spatial import modal_spatial_continuation as launcher

JsonDict = dict[str, object]

FULL_BOARD_ANSWER = "; ".join(
    f"{key} " + ("wood 6" if key.startswith("<T") else "3:1 port" if key.startswith("<P")
                else "<T00>" if key == "robber" else "empty") for key in board_keys()
)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_rows(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def row(index: object, family: str = "node_tiles", task: str | None = None) -> JsonDict:
    task = task or {"directions": "node_direction_yes", "adjacency_connectivity": "node_adjacent_no"}.get(family, family)
    answer = "<T00>"
    if task == "full_board_readout":
        answer = FULL_BOARD_ANSWER
    return {"row_id": f"row-{index}", "images": ["image.png"], "task_type": task,
            "training_family": family, "metadata": {"task_type": task, "query_id": str(index)},
            "messages": [{"role": "user", "content": "<image>\nAnswer only."},
                         {"role": "assistant", "content": answer}]}


def training_rows() -> list[JsonDict]:
    families = [k for k in launcher.FAMILY_STEPS if k != "full_board_readout"] + ["full_board_readout", "full_board_readout"]
    return [row(step * 8 + i, family) for step, family in enumerate(families * 16) for i in range(8)]


def panel_result(plan: dict[str, object], label: str, correct: int = 0) -> JsonDict:
    count = launcher.PANEL_BUDGETS[label][0]
    summary = {"rows": count, "attempted": count, "correct": correct, "exact_accuracy": correct / count}
    if label == "fullboard":
        summary["full_board"] = {"board_exact": correct, "occupied_layout_macro_accuracy": 0.75,
                                 "by_layout": {"layout": {"boards": count}}}
    return {"summary": summary, "identity": copy.deepcopy(plan["panels"][label]["identity"]),
            "scorer_sha256": launcher.digest(plan["source_sha256"]),
            "conditions": launcher.evaluation_conditions(label)}


def fake_volumes(monkeypatch: pytest.MonkeyPatch, events: list[object]) -> None:
    for name in ("hf_cache", "sft_data", "sft_runs"):
        monkeypatch.setattr(launcher, name, SimpleNamespace(
            reload=lambda name=name: events.append(f"{name}.reload"),
            commit=lambda name=name: events.append(f"{name}.commit")))


class FakeCall:
    def __init__(
        self,
        name: str,
        result: JsonDict,
        events: list[object],
        error: BaseException | str | None = None,
    ) -> None:
        self.object_id = f"fc-{name}"
        self.result = result
        self.events = events
        self.error = error
        self.cancelled = False

    def get(self, timeout: float) -> JsonDict:
        self.events.append(("get", self.object_id, timeout))
        if self.error:
            raise self.error
        return self.result

    def cancel(self, terminate_containers: bool = False) -> None:
        assert terminate_containers
        self.cancelled = True
        self.events.append(("cancel", self.object_id))


def coordinator_setup(
    plan: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    failure: tuple[str, BaseException | str] | None = None,
) -> tuple[list[object], list[FakeCall], Path]:
    events: list[object] = []
    calls: list[FakeCall] = []
    fake_volumes(monkeypatch, events)
    monkeypatch.setattr(launcher.modal, "current_function_call_id", lambda: "fc-coordinator")
    directory = launcher.RUN_ROOT / "pipelines" / plan["run_name"]
    write_json(directory / "launch.json", plan)
    before = {label: panel_result(plan, label) for label in ("spatial", "fullboard")}

    def spawn(name: str, result: JsonDict) -> FakeCall:
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
