"""Fixtures shared by the continuation and extension launcher suites."""

from dataclasses import asdict
from pathlib import Path

import modal
import pytest

from evals.catan_board_bench.tokens import semantic_recognition_token_inventory
from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.scripts.train.train_trl_catan_vision import TrainConfig

from .support import row, training_rows, write_json, write_rows


@pytest.fixture(autouse=True)
def no_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("offline test crossed a Modal or upload boundary")

    for cls, names in ((modal.Function, ("spawn", "remote")),
                       (modal.Volume, ("reload", "commit", "batch_upload")),
                       (modal.App, ("run",))):
        for name in names:
            monkeypatch.setattr(cls, name, forbidden)
    monkeypatch.setattr(launcher, "upload_training_bundle", forbidden)
    monkeypatch.setattr(launcher, "upload_eval_jsonl", forbidden)


@pytest.fixture
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    root = tmp_path / "runs"
    local = tmp_path / "receipts"
    root.mkdir()
    local.mkdir()
    monkeypatch.setattr(launcher, "RUN_ROOT", root)
    monkeypatch.setattr(launcher, "LOCAL_RUN_ROOT", local)
    monkeypatch.setattr(launcher, "PARENT_CHECKPOINT", str(root / "parent/checkpoints/checkpoint-128"))
    monkeypatch.setattr(launcher, "source_hashes", lambda: {"scorer.py": "version-one"})
    images = tmp_path / "images"
    images.mkdir()
    (images / "image.png").write_bytes(b"unit-test-image-bytes")
    train_path = tmp_path / "train.jsonl"
    write_rows(train_path, training_rows())
    panels = {}
    tasks = {"spatial": "directions", "fullboard": "full_board_readout", "node_tiles": "node_tiles",
             "paths": "shortest_node_path", "local": "local_node_tiles", "production": "dice_production"}
    for label, (count, batch, budget) in launcher.PANEL_BUDGETS.items():
        path = tmp_path / f"{label}.jsonl"
        write_rows(path, [row(f"{label}-{i}", tasks[label]) for i in range(count)])
        panels[label] = {"eval_jsonl": str(path), "image_root": str(images), "batch_size": batch,
                         "max_new_tokens": budget, "identity": launcher.dataset_identity(str(path), str(images))}
    inventory = tmp_path / "tokens.json"
    write_json(inventory, semantic_recognition_token_inventory())
    config = asdict(TrainConfig(train_jsonl=str(train_path), image_root=str(images), token_inventory=str(inventory),
                               output_dir=str(root / "test-run"), eval_jsonl=panels["fullboard"]["eval_jsonl"],
                               eval_image_root=str(images), per_device_eval_batch_size=2,
                               initial_bundle=launcher.PARENT_CHECKPOINT, **launcher.FIXED_CONFIG))
    return {"run_name": "test-run", "config": config, "parent_config": {**config, "seed": 44},
            "config_sha256": launcher.digest(config),
            "train_identity": launcher.dataset_identity(str(train_path), str(images)),
            "panels": panels, "mixture": launcher.validate_mixture(training_rows()),
            "source_sha256": launcher.source_hashes(), "reservation_id": "reservation-test",
            "saved_baselines": {"spatial": {}, "fullboard": {}}, "legacy_pilot_sha256": "legacy"}
