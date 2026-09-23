"""Finalize run evidence after the Miles driver has saved data-source state."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from sft.json_types import JsonDict, as_dict, as_str, load_json_dict

from .artifacts import audit_hf_export, audit_native
from .contracts import MILES_COMMIT, TARGET_SUFFIXES, require
from .storage import file_hash, write_receipt


def _count(value: object, label: str, minimum: int = 1) -> int:
    if type(value) is not int or not isinstance(value, int) or value < minimum:
        raise ValueError(f"invalid {label}")
    return value


def _finite(value: object, label: str) -> float:
    if type(value) not in (float, int) or not isinstance(value, (float, int)):
        raise ValueError(f"invalid {label}")
    require(math.isfinite(value), f"nonfinite {label}")
    return float(value)


def _steps(directory: Path, initial: JsonDict) -> list[JsonDict]:
    records: list[JsonDict] = []
    expected: set[str] = set()
    world = _count(initial["world_size"], "world size")
    rollouts = _count(initial["num_rollout"], "rollouts")
    steps = _count(initial["steps_per_rollout"], "steps per rollout")
    for rank in range(world):
        peer = load_json_dict(directory / f"initialize-rank-{rank}.json")
        require(peer == {**initial, "rank": rank}, "rank initialization identities disagree")
        updated = False
        for rollout_id in range(rollouts):
            for step_id in range(steps):
                stem = f"rollout-{rollout_id}-step-{step_id}-rank-{rank}"
                expected.update((f"{stem}-before.json", f"{stem}-after.json"))
                before = load_json_dict(directory / f"{stem}-before.json")
                after = load_json_dict(directory / f"{stem}-after.json")
                for record in (before, after):
                    require(record.get("rollout_id") == rollout_id and record.get("step_id") == step_id
                            and record.get("rank") == rank, "step receipt identity mismatch")
                require(all(after.get(k) == v for k, v in before.items()), "before/after witness mismatch")
                require(after.get("successful") is True and after.get("loss_timing") == "forward_before_update",
                        "step did not successfully finish")
                _count(after.get("gradient_tensors"), "gradient tensor count")
                _count(after.get("nonzero_gradient_tensors"), "nonzero gradient tensor count")
                require(after.get("nonzero_gradient_families") == sorted(TARGET_SUFFIXES),
                        "missing target-family gradient witnesses")
                require(_finite(after.get("gradient_l2"), "gradient norm") > 0, "empty gradient witness")
                _finite(after.get("grad_norm"), "step grad norm")
                _finite(after.get("optimizer_grad_norm"), "optimizer grad norm")
                losses = as_dict(after.get("losses"))
                require(bool(losses), "missing real loss metrics")
                for name, value in losses.items():
                    _finite(value, name)
                if after.get("parameters_changed") is True:
                    require(bool(after.get("changed_tensors"))
                            and _finite(after.get("update_l2"), "update norm") > 0, "empty update witness")
                    updated = True
                records.append(after)
        require(updated, f"rank {rank} never witnessed an actual adapter update")
    require({p.name for p in directory.glob("rollout-*.json")} == expected,
            "missing/extra/unfinished step receipts")
    return records


def finalize_run(directory: Path) -> JsonDict:
    """Call after successful driver exit; this is the only writer of run.json.

    Requires every configured step/rank, finite gradients/losses, actual adapter
    updates, matching final native/HF artifacts, and the real persisted cursor.
    A post-save hook alone deliberately cannot satisfy this contract.
    """
    initial = load_json_dict(directory / "initialize-rank-0.json")
    require(initial.get("miles_commit") == MILES_COMMIT and initial.get("rank") == 0,
            "wrong initialized Miles revision/rank")
    require(file_hash(Path(as_str(initial["prompt_data"]))) == initial["input_sha256"],
            "prepared dataset changed during the run")
    steps = _steps(directory, initial)
    rollouts = _count(initial["num_rollout"], "rollouts")
    world = _count(initial["world_size"], "world size")
    final_id = rollouts - 1
    checkpoint = load_json_dict(directory / f"checkpoint-{final_id}.json")
    require(checkpoint.get("status") == "model_saved_cursor_pending" and checkpoint.get("complete") is False
            and checkpoint.get("rollout_id") == final_id, "invalid post-save receipt")
    root = Path(as_str(initial["save"]))
    native_path = root / f"iter_{final_id:07d}"
    hf_path = Path(as_str(initial["save_hf"]).format(rollout_id=final_id))
    require(str(native_path.resolve()) == checkpoint["checkpoint_dir"]
            and str(hf_path.resolve()) == checkpoint["hf_checkpoint_dir"], "final checkpoint paths disagree")
    cursor = root / "rollout" / f"global_dataset_state_dict_{final_id}.pt"
    require(cursor.is_file(), "data-source state has not yet been saved")
    saved: object = torch.load(cursor, map_location="cpu", weights_only=True)
    require(isinstance(saved, dict), "invalid saved data-source state")
    if not isinstance(saved, dict):
        raise ValueError(str(cursor))
    expected_samples = rollouts * _count(initial["rollout_batch_size"], "rollout batch size")
    require(saved.get("sample_index") == saved.get("sample_group_index") == expected_samples,
            "data-source cursor does not match the completed fresh run")
    _count(saved.get("sample_offset"), "sample offset", 0)
    _count(saved.get("epoch_id"), "epoch ID", 0)
    scopes = [load_json_dict(directory / f"scope-rank-{rank}.json") for rank in range(world)]
    require(audit_native(native_path, final_id, scopes) == checkpoint["native"],
            "native artifacts changed after post-save audit")
    require(audit_hf_export(Path(as_str(initial["hf_checkpoint"])), hf_path) == checkpoint["hf"],
            "HF export changed after post-save audit")
    receipt: JsonDict = {
        "schema": "catan_miles_sft_run/v1", "complete": True, "miles_commit": MILES_COMMIT,
        "final_rollout_id": final_id, "world_size": world,
        "optimizer_steps": len(steps) // world, "rank_step_receipts": len(steps),
        "input_sha256": initial["input_sha256"], "data_source_state_sha256": file_hash(cursor),
        "data_source_state": str(cursor.resolve()), "checkpoint": checkpoint,
        "first_step_losses": steps[0]["losses"], "last_step_losses": steps[(len(steps) // world) - 1]["losses"],
    }
    write_receipt(directory / "run.json", receipt)
    return receipt
