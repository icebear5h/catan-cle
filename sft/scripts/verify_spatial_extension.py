"""Read extension receipts from Modal volumes, or independently verify them offline.

python -B -m sft.scripts.verify_spatial_extension --download --status-only
python -B -m sft.scripts.verify_spatial_extension --download
python -B -m sft.scripts.verify_spatial_extension
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import modal

from sft.modal_spatial_continuation import (
    LOCAL_RUN_ROOT, NEW_PANEL_TASKS, OLD_PANELS, PANEL_BUDGETS, PROJECT_ROOT,
    check_identity, dataset_identity, digest, evaluation_conditions, read_json,
)
from sft.modal_spatial_extension import (
    CHECKPOINT_STEPS, DEFAULT_RUN_NAME, PARENT_CHECKPOINT, PARENT_RUN, validate_config,
)
from sft.scripts import eval_qwen_vl_adapter as evaluator
from sft.scripts.train_trl_catan_vision import sha256_file


# The scorer and its aggregate/topology dependencies, rather than unrelated source.
SCORER_FILES = (
    "sft/scripts/eval_qwen_vl_adapter.py", "sft/spatial_tasks.py",
    "sft/board_state_readout.py", "sft/behavior_diagnostics.py",
    "evals/catan_board_bench/tokens.py", "cle/game_engine/board_tokens.py",
    "cle/game_engine/models/map.py", "cle/game_engine/models/enums.py",
    "cle/game_engine/models/player.py",
)
CHECKPOINT_FILES = (
    "adapter_config.json", "adapter_model.safetensors", "visual_model.safetensors",
    "trainable_parameters.json", "training_config.json", "tokenizer_config.json",
    "tokenizer.json", "trainer_state.json",
)
STATE_PATH = "checkpoints/checkpoint-256/trainer_state.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def local_path(run: Path, relative: str) -> Path:
    """Never follow a receipt/output symlink into a historical run."""
    path = run / relative
    require(path.is_relative_to(run) and ".." not in Path(relative).parts,
            f"path escapes run: {relative}")
    for part in (path, *path.parents):
        if part == run:
            break
        require(not part.is_symlink(), f"symlink receipt path: {part}")
    return path


def atomic_write(run: Path, relative: str, data: bytes) -> None:
    path = local_path(run, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, local_path(run, relative))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_launch(run_dir: Path) -> tuple[Path, dict]:
    run = run_dir.expanduser().resolve(strict=True)
    launch = read_json(local_path(run, "launch.json"))
    require(launch["schema"] == "catan_spatial_extension_launch/v1"
            and launch["run_name"] == run.name and run.name != PARENT_RUN,
            "not an extension launch directory")
    validate_config(launch["config"], run.name, launch["parent_config"])
    require(launch["config_sha256"] == digest(launch["config"])
            and launch["parent_config_sha256"] == digest(launch["parent_config"])
            and launch["parent_checkpoint"] == PARENT_CHECKPOINT,
            "launch config/parent identity differs")
    require(set(launch["panels"]) == set(launch["saved_baselines"]) == set(PANEL_BUDGETS),
            "launch requires all six panels")
    require(launch["policy"]["checkpoint_steps"] == list(CHECKPOINT_STEPS),
            "launch checkpoint schedule differs")
    for label, (count, batch, budget) in PANEL_BUDGETS.items():
        panel = launch["panels"][label]
        require((panel["identity"]["rows"], panel["batch_size"], panel["max_new_tokens"])
                == (count, batch, budget), f"{label}: launch budget differs")
    return run, launch


def expected_checkpoint(launch: dict) -> str:
    return f"{launch['config']['output_dir']}/checkpoints/checkpoint-{launch['config']['max_steps']}"


def validate_result(result: dict, launch: dict) -> None:
    require(result["config"] == launch["config"]
            and result["source_sha256"] == launch["source_sha256"],
            "coordinator config/source differs from launch")
    require(result["status"] in ("running", "completed", "failed")
            and result["phase"] in ("preflight", "training", "post", "completed")
            and isinstance(result["coordinator_call_id"], str) and result["coordinator_call_id"]
            and set(result["stages"]) <= {"preflight", "training", "post"},
            "invalid coordinator status/stages")
    require(all(isinstance(entry, dict)
                and entry.get("status") in ("starting", "running", "completed", "failed")
                for entry in result["stages"].values()), "invalid stage receipt")
    if launch.get("coordinator_call_id"):
        require(result["coordinator_call_id"] == launch["coordinator_call_id"],
                "coordinator call ID differs from launch")


def status_report(result: dict) -> dict:
    report = {key: result[key] for key in ("status", "phase", "coordinator_call_id")}
    report["stages"] = {
        name: {key: entry[key] for key in ("status", "call_id", "error") if key in entry}
        for name, entry in result["stages"].items()
    }
    worker = result["stages"].get("training", {}).get("result", {})
    metrics = {group: {key: value for key, value in worker["training"][group].items()
                       if key in ("epoch", "train_loss", "train_runtime", "eval_loss",
                                  "answer_token_accuracy", "answer_row_exact", "eval_mean_token_accuracy")}
               for group in ("metrics", "eval_metrics") if group in worker.get("training", {})}
    if "teacher_forced_history" in worker:
        metrics["teacher_forced_history"] = [
            {key: row[key] for key in ("step", "eval_loss") if key in row}
            for row in worker["teacher_forced_history"]]
    if metrics:
        report["training"] = metrics
    if "error" in result:
        report["error"] = result["error"]
    return report


def completed_post(result: dict, post: dict, launch: dict) -> None:
    require(result["status"] == result["phase"] == "completed"
            and all(result["stages"][key]["status"] == "completed"
                    for key in ("preflight", "training", "post")),
            "pipeline has not completed all three stages")
    require(post == result["stages"]["post"]["result"], "post receipt differs from coordinator")
    require(post["status"] == "completed" and set(post["panels"]) == set(PANEL_BUDGETS),
            "post requires all six completed panels")
    require(post["checkpoint"] == expected_checkpoint(launch)
            and post["source_sha256"] == launch["source_sha256"],
            "post checkpoint/source differs from launch")


def full_history(history: list[dict]) -> bool:
    return (isinstance(history, list) and all(isinstance(row, dict) for row in history)
            and [row.get("step") for row in history] == list(CHECKPOINT_STEPS))


def download_receipts(run: Path, launch: dict, *, status_only: bool) -> dict:
    """Only direct serial read_file calls; no listing, remote functions or GPU work."""
    volume = modal.Volume.from_name("catan-sft-runs", create_if_missing=False)
    remote = f"catan-vision-sft/pipelines/{launch['run_name']}"
    first = True

    def read(path: str) -> bytes:
        nonlocal first
        if not first:
            time.sleep(1)
        first = False
        return b"".join(volume.read_file(path))

    data = read(f"{remote}/result.json")
    result = json.loads(data)
    validate_result(result, launch)
    atomic_write(run, "result.json", data)
    if status_only or result["status"] != "completed":
        return result
    data = read(f"{remote}/post/result.json")
    post = json.loads(data)
    completed_post(result, post, launch)
    atomic_write(run, "post/result.json", data)
    for label in PANEL_BUDGETS:
        saved = post["panels"][label]
        for name in ("summary.json", "records.jsonl"):
            data = read(f"{remote}/post/{label}/{name}")
            if name == "summary.json":
                require(json.loads(data) == saved["summary"], f"{label}: summary receipt differs")
            else:
                require(hashlib.sha256(data).hexdigest() == saved["records_sha256"],
                        f"{label}: records hash differs")
            atomic_write(run, f"post/{label}-{name}", data)
    worker = result["stages"]["training"]["result"]
    if not full_history(worker.get("teacher_forced_history", [])):
        data = read(f"catan-vision-sft/{launch['run_name']}/{STATE_PATH}")
        require(hashlib.sha256(data).hexdigest()
                == post["checkpoint_audit"]["files_sha256"]["trainer_state.json"],
                "trainer_state hash differs")
        json.loads(data)
        atomic_write(run, STATE_PATH, data)
    return result


def verify_training(run: Path, launch: dict, result: dict, audit: dict) -> list[dict]:
    worker = result["stages"]["training"]["result"]
    require(worker["status"] == "completed" and worker["config"] == launch["config"]
            and worker["source_sha256"] == launch["source_sha256"]
            and worker["checkpoint"] == audit["checkpoint"] == expected_checkpoint(launch)
            and worker["checkpoint_audit"] == audit, "training/checkpoint audit differs")
    require(audit["visual_dtypes"] == {"F32": 333}, "checkpoint must contain 333 FP32 tensors")
    hashes = audit["files_sha256"]
    require(set(hashes) == set(CHECKPOINT_FILES)
            and all(isinstance(value, str) and len(value) == 64
                    and all(c in "0123456789abcdef" for c in value) for value in hashes.values()),
            "incomplete checkpoint file hashes")
    for key in ("tokens", "token_ids"):
        require(audit["semantic_tokens"][key] == launch["parent_audit"]["semantic_tokens"][key],
                f"checkpoint atlas {key} differs from parent")
    history = worker.get("teacher_forced_history")
    state_path = local_path(run, STATE_PATH)
    if state_path.exists():
        require(sha256_file(state_path) == hashes["trainer_state.json"], "trainer_state hash differs")
        state = read_json(state_path)
        require(state["global_step"] == launch["config"]["max_steps"], "trainer_state step differs")
        saved_history = [row for row in state["log_history"] if "eval_loss" in row]
        require(history is None or history == saved_history, "embedded history differs from trainer_state")
        history = saved_history
    require(history is not None and full_history(history), "all eight teacher-forced evaluations required")
    require(all(isinstance(row.get("eval_loss"), (int, float))
                and not isinstance(row["eval_loss"], bool) and math.isfinite(row["eval_loss"])
                for row in history), "non-finite or missing teacher-forced loss")
    if "checkpoints" in worker:
        require(set(worker["checkpoints"]) == {str(step) for step in CHECKPOINT_STEPS},
                "training checkpoint schedule differs")
        for step, saved in worker["checkpoints"].items():
            require(saved["checkpoint"] == f"{launch['config']['output_dir']}/checkpoints/checkpoint-{step}",
                    "training checkpoint path differs")
        require(worker["checkpoints"]["256"]["trainer_state_sha256"] == hashes["trainer_state.json"],
                "final training checkpoint hash differs")
    return history


def verify_panel(run: Path, launch: dict, post: dict, label: str) -> dict:
    panel, saved = launch["panels"][label], post["panels"][label]
    baseline = launch["saved_baselines"][label]["summary"]
    source = (OLD_PANELS[label] if label in OLD_PANELS
              else launch["dataset_inputs"]["new_panels"][NEW_PANEL_TASKS[label]])
    identity = dataset_identity(source["eval_jsonl"], source["image_root"])
    require(identity == panel["local_identity"], f"{label}: local input bytes/pixels/order differ")
    check_identity(identity, panel["identity"])
    require(saved["identity"] == panel["identity"], f"{label}: uploaded input identity differs")
    require(saved["conditions"] == evaluation_conditions(label)
            and saved["scorer_sha256"] == digest(launch["source_sha256"]),
            f"{label}: conditions/scorer differ")
    require(saved["records_path"] == f"/runs/catan-vision-sft/pipelines/{run.name}/post/{label}/records.jsonl",
            f"{label}: records path differs")
    path = local_path(run, f"post/{label}-records.jsonl")
    records_hash = sha256_file(path)
    require(records_hash == saved["records_sha256"], f"{label}: records hash differs")
    records = [row for _, row in evaluator.iter_jsonl(path)]
    summary = read_json(local_path(run, f"post/{label}-summary.json"))
    require(summary == saved["summary"], f"{label}: summary receipt differs")
    require(baseline["eval_source_sha256"] == identity["sha256"]
            and baseline["eval_jsonl"] == panel["eval_jsonl"]
            and baseline["image_root"] == panel["image_root"] and baseline["eval_set_id"],
            f"{label}: pinned parent input metadata differs")
    audit = post["checkpoint_audit"]
    evidence = summary["adapter_evidence"]
    visual = evidence["visual_state"]
    require(evidence["adapter_loaded"] is True and evidence["adapter_dir"] == post["checkpoint"]
            and visual["loaded"] is True and visual["tensors"] == visual["expected_tensors"] == 333
            and visual["path"] == f"{post['checkpoint']}/visual_model.safetensors"
            and visual["sha256"] == audit["files_sha256"]["visual_model.safetensors"],
            f"{label}: checkpoint visual/hash identity differs")
    require(evidence["visual_precision"] == {
        "base_load_dtype": "torch.bfloat16", "loaded_dtypes": {"torch.float32": 333},
        "promoted_before_restore": True, "source_dtypes": {"F32": 333}},
        f"{label}: visual precision differs")
    require(summary["precision"]["preserve_visual_fp32"] is True
            and summary["reasoning_enabled"] is summary["candidate_scoring"] is False,
            f"{label}: precision/generation flags differ")
    for key in ("tokens", "token_ids"):
        require(evidence["semantic_tokens"][key] == audit["semantic_tokens"][key],
                f"{label}: loaded atlas {key} differ")
    rows = [row for _, row in evaluator.iter_jsonl(Path(source["eval_jsonl"]))]
    # Generation stably partitions the frozen input into short, then long answers.
    ordered = sorted(rows, key=evaluator.is_long_answer)
    count, batch, budget = PANEL_BUDGETS[label]
    require(len(records) == len(rows) == count, f"{label}: incomplete record count")
    for index, (row, record) in enumerate(zip(ordered, records, strict=True), 1):
        require(record["id"] == (row.get("id") or row["row_id"]) and record["index"] == index,
                f"{label}: record order/ID differs at {index}")
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        metadata.update(eval_set_id=baseline["eval_set_id"], eval_source_sha256=identity["sha256"])
        require(record["metadata"] == metadata and record["expected"] == evaluator.expected_text(row),
                f"{label}: target/metadata differs at {index}")
        require(isinstance(record["response"], str) and record.get("candidate_score") is None,
                f"{label}: missing response or unexpected candidate scoring at {index}")
        score = evaluator.score_response(record["expected"], record["response"], metadata=metadata)
        require(score == record["score"], f"{label}: saved score differs at {index}")
        record["score"] = score
    computed = evaluator.summarize(records)
    expected = {**computed, "generated_at": summary["generated_at"],
                "model_id": launch["config"]["model_id"], "adapter_dir": post["checkpoint"],
                "eval_jsonl": panel["eval_jsonl"], "image_root": panel["image_root"],
                "token_inventory": launch["config"]["token_inventory"],
                "bits": 16, "batch_size": batch, "max_new_tokens": budget,
                "long_max_new_tokens": budget, "adapter_evidence": evidence,
                "reasoning_enabled": False, "image_variant": "original", "occlusion_margin": 0.03,
                "candidate_scoring": False, "rows_skipped_without_spatial_target": 0,
                "eval_set_id": baseline["eval_set_id"], "eval_source_sha256": identity["sha256"],
                "precision": {"preserve_visual_fp32": True,
                              "generation_autocast": {"device_type": "cuda", "dtype": "torch.bfloat16"}}}
    differences = sorted(key for key in summary.keys() | expected.keys()
                         if key not in summary or key not in expected or summary[key] != expected[key])
    require(not differences, f"{label}: summary aggregates/settings differ: {', '.join(differences)}")
    return {"rows": count, "correct": computed["correct"], "exact_accuracy": computed["exact_accuracy"],
            "records_sha256": records_hash, "input_identity": identity}


def verify_local(run: Path, launch: dict) -> dict:
    result = read_json(local_path(run, "result.json"))
    validate_result(result, launch)
    post = read_json(local_path(run, "post/result.json"))
    completed_post(result, post, launch)
    for relative in SCORER_FILES:
        require(sha256_file(PROJECT_ROOT / relative) == launch["source_sha256"][relative],
                f"scorer source changed: {relative}")
    history = verify_training(run, launch, result, post["checkpoint_audit"])
    panels = {label: verify_panel(run, launch, post, label) for label in PANEL_BUDGETS}
    report = {"status": "PASS", "stage": "post", "run_name": run.name,
              "checkpoint": post["checkpoint"], "panels": panels,
              "teacher_forced_history": history,
              "scorer_sha256": {key: launch["source_sha256"][key] for key in SCORER_FILES},
              "receipts_sha256": {key: sha256_file(local_path(run, key))
                                  for key in ("launch.json", "result.json", "post/result.json")}}
    atomic_write(run, "post-verification.json", json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=LOCAL_RUN_ROOT / DEFAULT_RUN_NAME)
    parser.add_argument("--download", action="store_true", help="read receipts from Modal volumes")
    parser.add_argument("--status-only", action="store_true", help="show coordinator receipt; skip verification")
    args = parser.parse_args(argv)
    try:
        run, launch = load_launch(args.run_dir)
        result = (download_receipts(run, launch, status_only=args.status_only) if args.download
                  else read_json(local_path(run, "result.json")))
        validate_result(result, launch)
        if args.status_only or result["status"] != "completed":
            print(json.dumps(status_report(result), sort_keys=True))
            return 1 if result["status"] == "failed" else 0 if args.status_only else 2
        report = verify_local(run, launch)
        counts = " ".join(f"{label}={panel['correct']}/{panel['rows']}"
                          for label, panel in report["panels"].items())
        print(f"PASS checkpoint-256 {counts}\n{run / 'post-verification.json'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, modal.exception.Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
