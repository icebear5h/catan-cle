from types import SimpleNamespace

import pytest

import sft.modal_catan_vision_sft as launcher


def test_default_preserves_legacy_launcher_and_79_dollar_plan_reserves_evals():
    assert launcher.training_budget_plan(0) is None
    plan = launcher.training_budget_plan(79)
    assert plan["maximum_training_window_usd_at_list_rates"] == pytest.approx(45.272808)
    assert plan["standalone_eval_reserve_usd"] == 25
    assert plan["remaining_headroom_usd"] > 8
    assert plan["evals_auto_launched"] is False
    assert launcher.BUDGET_FUNCTION_OPTIONS["cpu"] == (16.0, 16.0)
    assert launcher.BUDGET_FUNCTION_OPTIONS["memory"] == (131072, 131072)
    assert launcher.BUDGET_FUNCTION_OPTIONS["retries"] == 0
    assert launcher.BUDGET_FUNCTION_OPTIONS["max_containers"] == 1


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf"), 40, 70])
def test_invalid_or_insufficient_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        launcher.training_budget_plan(budget)


class FakeCall:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.cancelled = False
        self.timeouts = []

    def get(self, timeout):
        self.timeouts.append(timeout)
        if self.error:
            raise self.error
        return self.result

    def cancel(self, terminate_containers=False):
        assert terminate_containers
        self.cancelled = True


def test_guard_returns_completed_result_without_cancelling(monkeypatch):
    monkeypatch.setattr(launcher.time, "time", lambda: 100)
    call = FakeCall(result={"status": "completed"})
    assert launcher.wait_for_budgeted_call(call, 200) == {"status": "finished", "result": {"status": "completed"}}
    assert not call.cancelled
    assert call.timeouts == [30]


def test_guard_cancels_at_absolute_deadline_even_after_poll_timeout(monkeypatch):
    times = iter([100, 111])
    monkeypatch.setattr(launcher.time, "time", lambda: next(times))
    call = FakeCall(error=TimeoutError())
    assert launcher.wait_for_budgeted_call(call, 110)["status"] == "budget_deadline_cancelled"
    assert call.cancelled and call.timeouts == [10]


def test_expired_guard_cancels_without_waiting(monkeypatch):
    monkeypatch.setattr(launcher.time, "time", lambda: 100)
    call = FakeCall()
    assert launcher.wait_for_budgeted_call(call, 99)["status"] == "budget_deadline_cancelled"
    assert call.cancelled and not call.timeouts


def test_guard_fails_closed_on_errors(monkeypatch):
    monkeypatch.setattr(launcher.time, "time", lambda: 100)
    call = FakeCall(error=RuntimeError("training failed"))
    with pytest.raises(RuntimeError, match="training failed"):
        launcher.wait_for_budgeted_call(call, 200)
    assert call.cancelled


def test_budgeted_function_calls_trainer_locally_with_no_resume(monkeypatch):
    monkeypatch.setattr(launcher.time, "time", lambda: 100)
    calls = []
    monkeypatch.setattr(launcher, "train_h200", SimpleNamespace(local=lambda *args: calls.append(args) or "ok"))
    payload = {"budget": launcher.training_budget_plan(79)}
    function = launcher.train_h200_budgeted.get_raw_f()
    assert function({"config": "test"}, payload, 200) == "ok"
    assert calls == [({"config": "test"}, payload, False)]
    with pytest.raises(RuntimeError, match="expired"):
        function({}, payload, 99)
    with pytest.raises(ValueError, match="exceeds"):
        function({}, payload, 1e9)
    with pytest.raises(ValueError, match="inconsistent"):
        function({}, {"budget": {"budget_usd": 79}}, 200)
    assert len(calls) == 1


def test_main_dry_run_never_launches_and_writes_budget_receipt(tmp_path, monkeypatch, capsys):
    import json

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run crossed a paid boundary")

    monkeypatch.setattr(launcher, "train_h200_budgeted", SimpleNamespace(spawn=forbidden))
    path = tmp_path / "plan.json"
    launcher.main(budget_usd=79, receipt_path=str(path))
    result = json.loads(path.read_text())
    assert result["dry_run"] and not result["paid_gpu_requested"]
    assert result["budget"]["budget_usd"] == 79
    with pytest.raises(FileExistsError):
        launcher.main(budget_usd=79, receipt_path=str(path))
    with pytest.raises(ValueError, match="do not resume"):
        launcher.main(budget_usd=79, resume_latest=True)
