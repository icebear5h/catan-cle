"""Shared fixtures for offline verification of spatial extension receipts and downloads."""

import copy
import socket
from dataclasses import asdict, replace
from pathlib import Path

import modal
import pytest

from sft.launchers.spatial import modal_spatial_extension as extension
from sft.launchers.spatial.modal_spatial_continuation import FIXED_CONFIG
from sft.scripts.eval import verify_spatial_extension as verify
from sft.scripts.train.train_trl_catan_vision import TrainConfig

from .support import PARENT, RUN_NAME, running_result, write_json


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("offline test crossed a remote/model/training boundary")

    for cls, methods in (
        (modal.Function, ("remote", "spawn", "from_name")),
        (modal.FunctionCall, ("from_id", "get", "cancel")),
        (modal.Volume, ("from_name", "read_file", "iterdir", "reload", "commit", "batch_upload")),
        (modal.App, ("run",)),
        (socket.socket, ("connect", "connect_ex")),
    ):
        for method in methods:
            monkeypatch.setattr(cls, method, forbidden)
    monkeypatch.setattr(verify.evaluator, "load_model", forbidden)
    monkeypatch.setattr(verify.evaluator, "run_eval_job", forbidden)
    monkeypatch.setattr(extension, "run_training", forbidden)
    monkeypatch.setattr(extension.AutoTokenizer, "from_pretrained", forbidden)


@pytest.fixture
def launch(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Portable transport fixture; never claims these identities are real eval data."""
    parent = TrainConfig(**FIXED_CONFIG, seed=44, train_jsonl="/data/train.jsonl",
                         image_root="/data/images", eval_image_root="/data/images",
                         token_inventory="/data/tokens.json", eval_jsonl="/data/eval.jsonl",
                         per_device_eval_batch_size=2,
                         output_dir=f"/runs/catan-vision-sft/{verify.PARENT_RUN}")
    config = asdict(replace(parent, **extension.CONFIG_CHANGES,
                           initial_bundle=verify.PARENT_CHECKPOINT,
                           output_dir=f"/runs/catan-vision-sft/{RUN_NAME}"))
    extension.validate_config(config, RUN_NAME, asdict(parent))
    panels = {label: {"identity": {"rows": count}, "batch_size": batch,
                      "max_new_tokens": budget} for label, (count, batch, budget)
              in verify.PANEL_BUDGETS.items()}
    plan = {"schema": "catan_spatial_extension_launch/v1", "run_name": RUN_NAME,
            "config": config, "config_sha256": verify.digest(config),
            "parent_config": asdict(parent), "parent_config_sha256": verify.digest(asdict(parent)),
            "parent_checkpoint": verify.PARENT_CHECKPOINT, "panels": panels,
            "saved_baselines": {label: {} for label in panels},
            "source_sha256": {key: verify.sha256_file(verify.PROJECT_ROOT / key)
                              for key in verify.SCORER_FILES},
            "policy": {"checkpoint_steps": list(verify.CHECKPOINT_STEPS)},
            "coordinator_call_id": "fc-offline-coordinator"}
    run = tmp_path / RUN_NAME
    write_json(run / "launch.json", plan)
    return run, plan


@pytest.fixture(scope="module")
def derived_parent_256() -> tuple[dict[str, bytes], bytes]:
    """EXPLICIT TEST-ONLY derivation: parent generations are NOT extension results.

    Keep all 430 real responses/scores, input identities and parent eval-set IDs.
    Retarget checkpoint paths and create an eight-step history solely to exercise
    the verifier's checkpoint-256 contract without launching inference/training.
    """
    if not (PARENT / "post/result.json").exists():
        pytest.skip("retained parent generations are local artifacts")
    plan = extension.build_plan(extension.DEFAULT_INPUTS, RUN_NAME)
    plan["coordinator_call_id"] = "fc-offline-coordinator"
    post = copy.deepcopy(verify.read_json(PARENT / "post/result.json"))
    parent = verify.read_json(PARENT / "result.json")
    checkpoint = verify.expected_checkpoint(plan)
    post.update(checkpoint=checkpoint, source_sha256=plan["source_sha256"])
    post["checkpoint_audit"]["checkpoint"] = checkpoint
    # This history is explicit fixture data, never a runtime fallback.
    history = [{"step": step, "eval_loss": 1 / step} for step in verify.CHECKPOINT_STEPS]
    state = {"global_step": 256, "log_history": history}
    state_bytes = verify.json_bytes(state)
    post["checkpoint_audit"]["files_sha256"]["trainer_state.json"] = verify.hashlib.sha256(state_bytes).hexdigest()
    files = {}
    for label, panel in post["panels"].items():
        panel["records_path"] = f"/runs/catan-vision-sft/pipelines/{RUN_NAME}/post/{label}/records.jsonl"
        panel["scorer_sha256"] = verify.digest(plan["source_sha256"])
        summary = panel["summary"]
        summary["adapter_dir"] = summary["adapter_evidence"]["adapter_dir"] = checkpoint
        summary["adapter_evidence"]["visual_state"]["path"] = f"{checkpoint}/visual_model.safetensors"
        files[f"post/{label}-summary.json"] = verify.json_bytes(summary)
        files[f"post/{label}-records.jsonl"] = (PARENT / f"post/{label}-records.jsonl").read_bytes()
    worker = {"status": "completed", "config": plan["config"], "checkpoint": checkpoint,
              "source_sha256": plan["source_sha256"], "checkpoint_audit": post["checkpoint_audit"],
              "training": parent["stages"]["training"]["result"]["training"],
              "teacher_forced_history": history,
              "checkpoints": {str(step): {
                  "checkpoint": f"{plan['config']['output_dir']}/checkpoints/checkpoint-{step}",
                  "trainer_state_sha256": post["checkpoint_audit"]["files_sha256"]["trainer_state.json"]}
                  for step in verify.CHECKPOINT_STEPS}}
    result = running_result(plan, status="completed", phase="completed")
    result["stages"]["training"].update(status="completed", result=worker)
    result["stages"]["post"] = {"status": "completed", "call_id": "fc-offline-post", "result": post}
    files.update({"launch.json": verify.json_bytes(plan), "result.json": verify.json_bytes(result),
                  "post/result.json": verify.json_bytes(post)})
    return files, state_bytes


@pytest.fixture
def retained_run(
    derived_parent_256: tuple[dict[str, bytes], bytes], tmp_path: Path
) -> tuple[Path, dict[str, object]]:
    files, _ = derived_parent_256
    run = tmp_path / RUN_NAME
    for name, data in files.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return run, verify.read_json(run / "launch.json")
