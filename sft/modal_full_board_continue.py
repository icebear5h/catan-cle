"""One bounded full-board continuation with identical full-validation pre/post checks."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from pathlib import Path

from sft.board_state_readout import select_validation
from sft.modal_full_board_pilot import (
    DATASET, ROOT, VOLUMES, app, evaluate, preflight, sft_runs, train, training_image,
    upload_training_bundle,
)
from sft.scripts.train_trl_catan_vision import TrainConfig, iter_jsonl, sha256_file, write_json_atomic


@app.function(image=training_image, cpu=(0.25, 0.25), memory=(2048, 2048),
              timeout=11100, retries=0, scaledown_window=2, volumes=VOLUMES)
def continue_cycle(payload: dict, parent_payload: dict, run_name: str, saved_baseline: dict | None = None) -> dict:
    destination = Path("/runs/catan-vision-sft/pipelines") / run_name / "result.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = {"status": "running", "phase": "baseline", "config": payload}

    def persist():
        sft_runs.reload()
        write_json_atomic(destination, result)
        sft_runs.commit()

    persist()
    try:
        if saved_baseline is not None:
            if saved_baseline["status"] != "completed" or saved_baseline["checkpoint"] != payload["initial_bundle"]:
                raise ValueError("saved baseline must evaluate the exact parent checkpoint")
            result["baseline"] = saved_baseline
            result["baseline_reused"] = True
        else:
            result["baseline"] = evaluate.remote(parent_payload, 128, output_tag=run_name,
                                                  all_layout_controls=True)
        result["phase"] = "training"
        persist()
        result["training"] = train.remote(payload)
        result["phase"] = "evaluation"
        persist()
        result["evaluation"] = evaluate.remote(payload, payload["max_steps"],
                                                output_tag="full64", all_layout_controls=True)
        before = result["baseline"]["variants"]["heldout"]
        after = result["evaluation"]["variants"]["heldout"]
        result["comparison"] = {
            "boards": after["boards"],
            "board_exact_before": before["board_exact"], "board_exact_after": after["board_exact"],
            "occupied_layout_macro_before": before["occupied_layout_macro_accuracy"],
            "occupied_layout_macro_after": after["occupied_layout_macro_accuracy"],
            "by_layout_before": before["by_layout"], "by_layout_after": after["by_layout"],
        }
        result.update(status="completed", phase="completed")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        persist()
    return result


@app.local_entrypoint()
def continue_pipeline(execute: bool = False, run_name: str = "full-board-epoch2-20260907"):
    if not run_name.replace("-", "").isalnum():
        raise ValueError("invalid run name")
    receipt_dir = Path("artifacts/runs/sft") / run_name
    if (receipt_dir / "launch.json").exists():
        raise FileExistsError(receipt_dir / "launch.json")
    previous = json.loads(Path("artifacts/runs/sft/full-board-fresh-20260907/launch.json").read_text())
    parent = TrainConfig(**previous["config"])
    train_rows = [r for _, r in iter_jsonl(DATASET / "stage1/train.jsonl")]
    validation = [r for _, r in iter_jsonl(DATASET / "stage1/validation.jsonl")]
    eval_rows = select_validation(validation, count=len(validation), seed=43)
    if len(train_rows) != 1024 or len(eval_rows) != 64:
        raise ValueError("unexpected dataset size")
    if {r["layout_id"] for r in train_rows} & {r["layout_id"] for r in eval_rows}:
        raise ValueError("train/validation layout overlap")
    random.Random(43).shuffle(train_rows)
    counts = {}
    for row in eval_rows:
        cell = counts.setdefault(row["layout_id"], {"boards": 0, "densities": {}})
        cell["boards"] += 1
        density = row["density_bin"]
        cell["densities"][density] = cell["densities"].get(density, 0) + 1
    if len(counts) != 5 or any(set(c["densities"]) != {"dense", "sparse", "setup", "empty"} for c in counts.values()):
        raise ValueError("validation must cover every density in all five layouts")
    plan = {"parent": parent.output_dir + "/checkpoints/checkpoint-128", "additional_steps": 128,
            "train_boards": len(train_rows), "validation": counts, "shuffle_seed": 43,
            "stages": ["full64 baseline plus controls", "one more epoch", "full64 evaluation plus controls"],
            "gpu": "one H200 stage at a time", "gpu_timeout_per_stage_seconds": 3600,
            "optimizer": "new optimizer and warmup; retained model and atlas weights"}
    if not execute:
        print(json.dumps(plan, indent=2))
        return
    receipt_dir.mkdir(parents=True, exist_ok=True)
    train_path, eval_path = receipt_dir / "train-epoch2.jsonl", receipt_dir / "validation64.jsonl"
    for path, rows in ((train_path, train_rows), (eval_path, eval_rows)):
        if path.exists():
            raise FileExistsError(path)
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    paths = upload_training_bundle(
        train_path, DATASET / "images", ROOT / "ms_swift_bidirectional_v1/trainable_tokens.json",
        remote_dir=f"catan-vision-sft/datasets/{run_name}", require_curriculum=False,
        eval_jsonl=eval_path, eval_image_root=DATASET / "images",
    )
    config = replace(parent, train_jsonl=paths[0], eval_jsonl=paths[1], image_root=paths[2],
                     eval_image_root=paths[2], token_inventory=paths[3],
                     output_dir=f"/runs/catan-vision-sft/{run_name}",
                     initial_bundle=plan["parent"], token_init="keep", seed=43)
    config.validate()
    baseline = replace(parent, eval_jsonl=paths[1], eval_image_root=paths[2], token_inventory=paths[3])
    audit = preflight.remote(asdict(config))
    receipt = {"plan": plan, "config": asdict(config), "baseline_config": asdict(baseline),
               "preflight": audit, "train_sha256": sha256_file(train_path),
               "eval_sha256": sha256_file(eval_path),
               "source_sha256": {p: sha256_file(Path(p)) for p in (
                   "sft/modal_full_board_continue.py", "sft/modal_full_board_pilot.py",
                   "sft/board_state_readout.py", "sft/scripts/train_trl_catan_vision.py")}}
    call = continue_cycle.spawn(asdict(config), asdict(baseline), run_name)
    receipt["call_id"] = call.object_id
    write_json_atomic(receipt_dir / "launch.json", receipt)
    print(json.dumps({"call_id": call.object_id, "receipt": str(receipt_dir / "launch.json"), "plan": plan}, indent=2))
