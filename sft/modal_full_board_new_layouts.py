"""Bounded pass over every new layout, with older examples mixed into each batch."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from sft.board_state_readout import select_validation
from sft.modal_full_board_continue import app, continue_cycle
from sft.modal_full_board_pilot import preflight, upload_training_bundle
from sft.scripts.train_trl_catan_vision import TrainConfig, iter_jsonl, sha256_file, write_json_atomic

ROOT = Path("artifacts/generated/board_recognition/full_board_diverse_v1")
PARENT = Path("artifacts/runs/sft/full-board-epoch2-20260907")


def mix_layouts(rows: list[dict], old_ids: set[str], *, new_layouts: int = 256, seed: int = 44) -> tuple[list[dict], dict]:
    fresh = [r for r in rows if r["row_id"] not in old_ids]
    older = [r for r in rows if r["row_id"] in old_ids]
    if len({r["layout_id"] for r in fresh}) != new_layouts:
        raise ValueError("unexpected number of fresh layouts")
    new = select_validation(fresh, count=new_layouts * 3, seed=seed)
    old = select_validation(older, count=new_layouts, seed=seed)
    for layout in {r["layout_id"] for r in new}:
        if Counter(r["density_bin"] for r in new if r["layout_id"] == layout) != {"dense": 1, "sparse": 1, "setup": 1}:
            raise ValueError("each new layout must contribute dense, sparse and setup examples")
    rng = random.Random(seed)
    rng.shuffle(new)
    rng.shuffle(old)
    mixed = []
    # Six new plus two older examples per effective batch of eight.
    for start in range(0, len(old), 2):
        batch = new[start * 3:(start + 2) * 3] + old[start:start + 2]
        rng.shuffle(batch)
        mixed.extend(batch)
    if len({r["row_id"] for r in mixed}) != len(mixed):
        raise ValueError("duplicate examples in training mixture")
    return mixed, {"new_boards": len(new), "older_boards": len(old), "new_layouts": new_layouts,
                   "new_density": dict(Counter(r["density_bin"] for r in new)),
                   "older_density": dict(Counter(r["density_bin"] for r in old)), "seed": seed}


@app.local_entrypoint()
def new_layout_pass(execute: bool = False, run_name: str = "full-board-new-layouts-20260907"):
    if not run_name.replace("-", "").isalnum():
        raise ValueError("invalid run name")
    receipt_dir = Path("artifacts/runs/sft") / run_name
    if (receipt_dir / "launch.json").exists():
        raise FileExistsError(receipt_dir / "launch.json")
    build = json.loads((ROOT / "build.json").read_text())
    if build["status"] != "completed" or not build["fixed_evaluation_bytes_unchanged"]:
        raise ValueError("diversified dataset is not ready")
    inputs = json.loads((ROOT / "dataset_inputs.json").read_text())
    if sha256_file(Path(inputs["train_jsonl"])) != build["export"]["files"]["train"]["sha256"]:
        raise ValueError("diversified training data changed")
    old_source = Path("artifacts/generated/board_recognition/replay_v1/full_board_readout_v1/stage1/train.jsonl")
    old_ids = {r["row_id"] for _, r in iter_jsonl(old_source)}
    rows = [r for _, r in iter_jsonl(Path(inputs["train_jsonl"]))]
    mixed, mixture = mix_layouts(rows, old_ids)
    result = json.loads((PARENT / "result.json").read_text())
    if result["status"] != "completed":
        raise ValueError("parent cycle is incomplete")
    parent = TrainConfig(**result["config"])
    baseline = result["evaluation"]
    # Use byte-identical evaluation rows and order from the preceding run.
    evaluation = PARENT / "validation64.jsonl"
    previous = json.loads((PARENT / "launch.json").read_text())
    if sha256_file(evaluation) != previous["eval_sha256"]:
        raise ValueError("parent validation file changed")
    eval_rows = [r for _, r in iter_jsonl(evaluation)]
    if len(eval_rows) != 64 or {r["layout_id"] for r in mixed} & {r["layout_id"] for r in eval_rows}:
        raise ValueError("validation count or layout isolation failed")
    plan = {**mixture, "steps": 128, "training_boards": len(mixed),
            "parent_checkpoint": baseline["checkpoint"], "baseline_board_exact": baseline["variants"]["heldout"]["board_exact"],
            "eval_boards": 64, "shuffled_image_test": False,
            "gpu": "one H200 stage at a time", "gpu_timeout_per_stage_seconds": 3600}
    if not execute:
        print(json.dumps(plan, indent=2))
        return
    receipt_dir.mkdir(parents=True, exist_ok=True)
    train_path = receipt_dir / "train.jsonl"
    if train_path.exists():
        raise FileExistsError(train_path)
    train_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in mixed))
    paths = upload_training_bundle(
        train_path, Path(inputs["image_root"]), Path(inputs["token_inventory"]),
        remote_dir=f"catan-vision-sft/datasets/{run_name}", require_curriculum=False,
        eval_jsonl=evaluation, eval_image_root=Path(inputs["image_root"]),
    )
    config = replace(parent, train_jsonl=paths[0], eval_jsonl=paths[1], image_root=paths[2],
                     eval_image_root=paths[2], token_inventory=paths[3],
                     output_dir=f"/runs/catan-vision-sft/{run_name}",
                     initial_bundle=baseline["checkpoint"], token_init="keep", seed=44)
    config.validate()
    audit = preflight.remote(asdict(config))
    receipt = {"plan": plan, "config": asdict(config), "preflight": audit,
               "train_sha256": sha256_file(train_path), "eval_sha256": sha256_file(evaluation),
               "parent_result_sha256": sha256_file(PARENT / "result.json"),
               "source_sha256": {p: sha256_file(Path(p)) for p in (
                   "sft/modal_full_board_new_layouts.py", "sft/modal_full_board_continue.py",
                   "sft/modal_full_board_pilot.py", "sft/scripts/train_trl_catan_vision.py")}}
    call = continue_cycle.spawn(asdict(config), asdict(parent), run_name, saved_baseline=baseline)
    receipt["call_id"] = call.object_id
    write_json_atomic(receipt_dir / "launch.json", receipt)
    print(json.dumps({"receipt": str(receipt_dir / "launch.json"), "call_id": call.object_id, "plan": plan}, indent=2))
