"""Bounded pass over every new layout, with older examples mixed into each batch."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from sft.board_state_readout import select_validation
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str, load_json_dict
from sft.launchers._json import at, at_dict, at_str
from sft.launchers._train_config import train_config
from sft.launchers.full_board.modal_full_board_continue import continue_cycle
from sft.launchers.full_board.modal_full_board_pilot import app, preflight
from sft.launchers.modal_catan_vision_sft import upload_training_bundle
from sft.scripts.train.train_trl_catan_vision import (
    iter_jsonl,
    sha256_file,
    write_json_atomic,
)

ROOT = Path("artifacts/generated/board_recognition/full_board_diverse_v1")
PARENT = Path("artifacts/runs/sft/full-board-epoch2-20260907")


def mix_layouts(rows: list[JsonDict], old_ids: set[str], *, new_layouts: int = 256,
                seed: int = 44) -> tuple[list[JsonDict], JsonLikeDict]:
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
    mixed: list[JsonDict] = []
    # Six new plus two older examples per effective batch of eight.
    for start in range(0, len(old), 2):
        batch = new[start * 3:(start + 2) * 3] + old[start:start + 2]
        rng.shuffle(batch)
        mixed.extend(batch)
    if len({r["row_id"] for r in mixed}) != len(mixed):
        raise ValueError("duplicate examples in training mixture")
    return mixed, {"new_boards": len(new), "older_boards": len(old), "new_layouts": new_layouts,
                   "new_density": dict(Counter(as_str(r["density_bin"]) for r in new)),
                   "older_density": dict(Counter(as_str(r["density_bin"]) for r in old)), "seed": seed}


@app.local_entrypoint()
def new_layout_pass(
    execute: bool = False,
    run_name: str = "full-board-new-layouts-20260907",
) -> None:
    if not run_name.replace("-", "").isalnum():
        raise ValueError("invalid run name")
    receipt_dir = Path("artifacts/runs/sft") / run_name
    if (receipt_dir / "launch.json").exists():
        raise FileExistsError(receipt_dir / "launch.json")
    build = load_json_dict(ROOT / "build.json")
    if build["status"] != "completed" or not build["fixed_evaluation_bytes_unchanged"]:
        raise ValueError("diversified dataset is not ready")
    inputs = load_json_dict(ROOT / "dataset_inputs.json")
    train_jsonl = Path(at_str(inputs, "train_jsonl"))
    if sha256_file(train_jsonl) != at(build, "export", "files", "train", "sha256"):
        raise ValueError("diversified training data changed")
    old_source = Path("artifacts/generated/board_recognition/replay_v1/full_board_readout_v1/stage1/train.jsonl")
    old_ids = {as_str(r["row_id"]) for _, r in iter_jsonl(old_source)}
    rows = [r for _, r in iter_jsonl(train_jsonl)]
    mixed, mixture = mix_layouts(rows, old_ids)
    result = load_json_dict(PARENT / "result.json")
    if result["status"] != "completed":
        raise ValueError("parent cycle is incomplete")
    parent = train_config(as_dict(result["config"]))
    baseline = as_dict(result["evaluation"])
    # Use byte-identical evaluation rows and order from the preceding run.
    evaluation = PARENT / "validation64.jsonl"
    previous = load_json_dict(PARENT / "launch.json")
    if sha256_file(evaluation) != previous["eval_sha256"]:
        raise ValueError("parent validation file changed")
    eval_rows = [r for _, r in iter_jsonl(evaluation)]
    if len(eval_rows) != 64 or {r["layout_id"] for r in mixed} & {r["layout_id"] for r in eval_rows}:
        raise ValueError("validation count or layout isolation failed")
    plan: JsonLikeDict = {**mixture, "steps": 128, "training_boards": len(mixed),
            "parent_checkpoint": baseline["checkpoint"],
            "baseline_board_exact": at_dict(baseline, "variants", "heldout")["board_exact"],
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
    image_root = Path(at_str(inputs, "image_root"))
    paths = upload_training_bundle(
        train_path, image_root, Path(at_str(inputs, "token_inventory")),
        remote_dir=f"catan-vision-sft/datasets/{run_name}", require_curriculum=False,
        eval_jsonl=evaluation, eval_image_root=image_root,
    )
    config = replace(parent, train_jsonl=paths[0], eval_jsonl=paths[1], image_root=paths[2],
                     eval_image_root=paths[2], token_inventory=paths[3],
                     output_dir=f"/runs/catan-vision-sft/{run_name}",
                     initial_bundle=as_str(baseline["checkpoint"]), token_init="keep", seed=44)
    config.validate()
    audit = preflight.remote(asdict(config))
    receipt: JsonLikeDict = {"plan": plan, "config": asdict(config), "preflight": audit,
               "train_sha256": sha256_file(train_path), "eval_sha256": sha256_file(evaluation),
               "parent_result_sha256": sha256_file(PARENT / "result.json"),
               "source_sha256": {p: sha256_file(Path(p)) for p in (
                    "sft/launchers/full_board/modal_full_board_new_layouts.py", "sft/launchers/full_board/modal_full_board_continue.py",
                    "sft/launchers/full_board/modal_full_board_pilot.py",
                    "sft/scripts/train/train_trl_catan_vision/__init__.py",
                    "sft/scripts/train/train_trl_catan_vision/_common.py",
                    "sft/scripts/train/train_trl_catan_vision/_config.py",
                    "sft/scripts/train/train_trl_catan_vision/_text_data.py",
                    "sft/scripts/train/train_trl_catan_vision/_vision_data.py",
                    "sft/scripts/train/train_trl_catan_vision/_datasets.py",
                    "sft/scripts/train/train_trl_catan_vision/_visual.py",
                    "sft/scripts/train/train_trl_catan_vision/_structure.py",
                    "sft/scripts/train/train_trl_catan_vision/_model_tokens.py",
                    "sft/scripts/train/train_trl_catan_vision/_frozen.py",
                    "sft/scripts/train/train_trl_catan_vision/_bundles.py",
                    "sft/scripts/train/train_trl_catan_vision/_optim.py",
                    "sft/scripts/train/train_trl_catan_vision/_trainer.py",
                    "sft/scripts/train/train_trl_catan_vision/_sft.py",
                    "sft/scripts/train/train_trl_catan_vision/_run.py")}}
    call = continue_cycle.spawn(asdict(config), asdict(parent), run_name, saved_baseline=baseline)
    receipt["call_id"] = call.object_id
    write_json_atomic(receipt_dir / "launch.json", receipt)
    print(json.dumps({"receipt": str(receipt_dir / "launch.json"), "call_id": call.object_id, "plan": plan}, indent=2))
