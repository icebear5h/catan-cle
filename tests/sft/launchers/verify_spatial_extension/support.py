"""Shared helpers for offline verification of spatial extension receipts and downloads."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest

from sft.scripts.eval import verify_spatial_extension as verify

RUN_NAME = "verify-extension-fixture"


PARENT = verify.LOCAL_RUN_ROOT / verify.PARENT_RUN


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(verify.json_bytes(value))


def running_result(
    plan: dict[str, object], *, status: str = "running", phase: str = "training"
) -> dict[str, object]:
    return {"status": status, "phase": phase, "config": plan["config"],
            "source_sha256": plan["source_sha256"],
            "coordinator_call_id": plan["coordinator_call_id"],
            "stages": {"preflight": {"status": "completed", "call_id": "fc-offline-preflight"},
                       "training": {"status": "running", "call_id": "fc-offline-training"}}}


def volume_transport(
    monkeypatch: pytest.MonkeyPatch,
    payloads: dict[str, bytes],
    *,
    fail_at: str | None = None,
) -> list[object]:
    """The sole allowed transport consumes whole iterators serially, with pacing."""
    events: list[object] = []
    active = False

    def read_file(path: str) -> Iterator[bytes]:
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

    def from_name(name: str, *, create_if_missing: bool) -> SimpleNamespace:
        assert name == "catan-sft-runs" and create_if_missing is False
        events.append(("volume", name))
        return SimpleNamespace(read_file=read_file)

    def sleep(seconds: float) -> None:
        assert not active and seconds == 1
        events.append(("sleep", seconds))

    monkeypatch.setattr(modal.Volume, "from_name", from_name)
    monkeypatch.setattr(verify.time, "sleep", sleep)
    return events


def remote_result(plan: dict[str, object]) -> str:
    return f"catan-vision-sft/pipelines/{plan['run_name']}/result.json"
