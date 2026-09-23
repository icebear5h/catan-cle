from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from sft.board import coordinate_comparison
from sft.json_types import (
    JsonDict,
    as_dict,
    as_int,
    as_list,
    as_str,
)
from sft.launchers.modal_catan_vision_sft import hf_cache, sft_runs
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    _message_pair,
    assert_runtime_versions,
    encode_text_pair,
    load_token_inventory,
    sha256_file,
    write_json_atomic,
)

from ._config import (
    BATCH,
    CACHE,
    COMPARISON_GPU_DEADLINE,
    CONTEXT,
    GPU_DEADLINE,
    NEW_TOKENS,
    VOLUMES,
    app,
    eval_image,
    hf_secret,
    reload_volumes,
)
from ._snapshots import adapter_preflight, base_preflight, file_manifest, snapshot, volume_bundle
from ._validation import digest, error_record, now, read_json, verify_inputs


@app.function(
    image=eval_image, volumes=VOLUMES, secrets=[hf_secret],
    cpu=(4.0, 4.0), memory=(16 * 1024, 16 * 1024), timeout=1200,
    startup_timeout=300, retries=0, max_containers=1, scaledown_window=2,
)
def prepare_cpu(launch: JsonDict) -> JsonDict:
    reload_volumes()
    receipt_path = Path(as_str(launch["prep_dir"])) / "prepare.json"
    if receipt_path.parent.exists() or Path(as_str(launch["output_dir"])).exists():
        raise FileExistsError("remote run name already used; choose a fresh --run-name")
    receipt_path.parent.mkdir(parents=True, exist_ok=False)
    receipt: JsonDict = {"status": "preparing", "launch_sha256": digest(launch),
                         "started_at": now()}
    write_json_atomic(receipt_path, receipt)
    sft_runs.commit()
    try:
        rows = verify_inputs(launch)
        receipt["dependencies"] = assert_runtime_versions()
        inventory = load_token_inventory(
            Path(as_str(launch["data_dir"])) / "trainable_tokens.json")
        if launch["adapter_dir"]:
            bundle, adapter_files = volume_bundle(as_str(launch["adapter_dir"]))
        else:
            bundle, adapter_files = snapshot(as_str(launch["hf_repo"]),
                                             as_str(launch["hf_revision"]),
                                             subdir=as_str(launch["checkpoint_subdir"]),
                                             adapter=True)
        tokenizer, checkpoint = adapter_preflight(bundle, inventory, as_str(launch["model_id"]),
                                                  as_str(launch["model_revision"]))
        lengths: list[JsonDict] = []
        for line, row in enumerate(rows, 1):
            prompt, answer = _message_pair(row, line_number=line, input_mode="text")
            pair = encode_text_pair(tokenizer, prompt, answer, max_sequence_length=CONTEXT)
            prompt_tokens = as_list(pair["labels"]).count(-100)
            completion_tokens = len(as_list(pair["input_ids"])) - prompt_tokens
            if prompt_tokens + NEW_TOKENS > CONTEXT or completion_tokens > NEW_TOKENS:
                raise ValueError(f"row {line} exceeds the 4096 context / 512 completion budget")
            lengths.append({"id": row.get("id") or row.get("row_id"),
                            "prompt": prompt_tokens, "gold_completion": completion_tokens})
        prepared_fields: JsonDict = {
            "checkpoint_dir": str(bundle), "checkpoint": checkpoint,
            "token_lengths": list(lengths),
            "max_prompt_tokens": max(as_int(item["prompt"]) for item in lengths),
            "max_gold_tokens": max(as_int(item["gold_completion"]) for item in lengths),
            "checkpoint_files": file_manifest(bundle, adapter_files),
        }
        receipt.update(prepared_fields)
        write_json_atomic(receipt_path, receipt)
        hf_cache.commit()
        sft_runs.commit()
        base, base_files = snapshot(as_str(launch["model_id"]),
                                    as_str(launch["model_revision"]), adapter=False)
        receipt["base_audit"] = base_preflight(base, base_files, bundle, inventory, checkpoint)
        base_fields: JsonDict = {"base_snapshot": str(base),
                                 "base_files": file_manifest(base, base_files),
                                 "status": "prepared"}
        receipt.update(base_fields)
    except BaseException as exc:
        failure: JsonDict = {"status": "failed", "error": error_record(exc)}
        receipt.update(failure)
    finally:
        receipt["ended_at"] = now()
        write_json_atomic(receipt_path, receipt)
        try:
            hf_cache.commit()
        finally:
            sft_runs.commit()
    return receipt


def eval_command(launch: JsonDict, prepared: JsonDict) -> list[str]:
    data_dir = as_str(launch["data_dir"])
    return [
        sys.executable, "-m", "sft.scripts.eval.eval_qwen_vl_adapter",
        "--eval-jsonl", data_dir + "/review.jsonl",
        "--token-inventory", data_dir + "/trainable_tokens.json",
        "--adapter-dir", as_str(prepared["checkpoint_dir"]),
        "--output-dir", as_str(launch["output_dir"]),
        "--model-id", as_str(as_dict(prepared["checkpoint"])["saved_model_id"]),
        "--model-revision", as_str(launch["model_revision"]),
        "--bits", "16", "--input-mode", "text", "--max-sequence-length", str(CONTEXT),
        "--batch-size", str(BATCH), "--long-batch-size", str(BATCH),
        "--max-new-tokens", str(NEW_TOKENS), "--long-max-new-tokens", str(NEW_TOKENS),
        "--no-candidate-scoring", "--disable-flash-attn2",
    ]


def verify_outputs(output: Path, launch: JsonDict) -> JsonDict:
    records = [as_dict(row) for _, row in evaluator.iter_jsonl(output / "records.jsonl")]
    inputs = as_dict(launch["inputs"])
    comparison = inputs.get("schema") == coordinate_comparison.SCHEMA
    count = as_int(inputs["rows"])
    schema = coordinate_comparison.SCHEMA if comparison else evaluator.BOARD_FLUENCY_SCHEMA
    ids = [record.get("id") for record in records]
    if len(ids) != count or len(set(ids)) != count or set(ids) != set(as_list(inputs["ids"])):
        raise ValueError(f"evaluation did not return exactly the expected {count} IDs")
    if any(not isinstance(record.get("response"), str)
           or as_dict(record.get("score", {})).get("scoring") != schema
           or type(as_dict(record.get("score", {})).get("correct")) is not bool
           for record in records):
        raise ValueError("evaluation contains incomplete/unscored records")
    summary = read_json(output / "summary.json")
    pinned_snapshot = str(Path(CACHE)
                          / ("models--" + as_str(launch["model_id"]).replace("/", "--"))
                          / "snapshots" / as_str(launch["model_revision"]))
    if summary.get("model_id") not in (launch["model_id"], pinned_snapshot):
        raise ValueError("summary base model differs from the pinned model/revision")
    if summary.get("rows") != count or summary.get("attempted") != count:
        raise ValueError("summary is incomplete")
    for key, expected in {
        "model_revision": launch["model_revision"],
        "input_mode": "text", "bits": 16, "max_sequence_length": CONTEXT,
        "candidate_scoring": False, "batch_size": BATCH,
        "max_new_tokens": NEW_TOKENS, "long_max_new_tokens": NEW_TOKENS,
    }.items():
        if summary.get(key) != expected:
            raise ValueError(f"summary inference condition differs: {key}")
    for dimension in (("family", "operation", "representation") if comparison else ("family", "operation")):
        actual = {key: as_dict(value)["total"]
                  for key, value in as_dict(summary[f"by_{dimension}"]).items()}
        if actual != inputs[f"by_{dimension}"]:
            raise ValueError(f"summary by_{dimension} differs from the validated input")
    if comparison:
        paired = coordinate_comparison.paired_summary(records)
        if summary.get("coordinate_comparison") != paired or paired["pairs"] != inputs["pairs"]:
            raise ValueError("paired summary differs from rescored raw predictions")
        for record in records:
            rescored = coordinate_comparison.score_coordinate_comparison(
                as_str(record["expected"]), as_str(record["response"]),
                as_dict(record["metadata"]),
            )
            if rescored != record["score"]:
                raise ValueError("stored comparison score differs from raw response")
        if sum(bool(as_dict(r["score"])["correct"]) for r in records) != summary["correct"]:
            raise ValueError("summary correct count differs from raw responses")
    return {"rows": count, "exact_accuracy": summary["exact_accuracy"],
            "files": file_manifest(output, ["records.jsonl", "summary.json"])}


@app.function(
    image=eval_image, volumes=VOLUMES, gpu="H200", cpu=(8.0, 8.0),
    memory=(64 * 1024, 64 * 1024), timeout=900, startup_timeout=300,
    retries=0, max_containers=1, scaledown_window=2,
)
def eval_h200(launch: JsonDict, prepared: JsonDict) -> JsonDict:
    return _eval_bounded(launch, prepared, GPU_DEADLINE)


@app.function(
    image=eval_image, volumes=VOLUMES, gpu="H200", cpu=(8.0, 8.0),
    memory=(64 * 1024, 64 * 1024), timeout=1800, startup_timeout=300,
    retries=0, max_containers=1, scaledown_window=2,
)
def eval_coordinate_h200(launch: JsonDict, prepared: JsonDict) -> JsonDict:
    if as_dict(launch["inputs"]).get("schema") != coordinate_comparison.SCHEMA:
        raise ValueError("coordinate worker requires the admitted matched comparison")
    return _eval_bounded(launch, prepared, COMPARISON_GPU_DEADLINE)


def _eval_bounded(launch: JsonDict, prepared: JsonDict,
                  execution_seconds: int) -> JsonDict:
    deadline = time.monotonic() + execution_seconds
    reload_volumes()
    output = Path(as_str(launch["output_dir"]))
    # A committed marker precedes model loading; repeated/redelivered calls fail closed.
    output.mkdir(parents=True, exist_ok=False)
    run: JsonDict = {"status": "running", "started_at": now(), "launch": launch,
                     "prepare_sha256": digest(prepared),
                     "command": list(eval_command(launch, prepared))}
    write_json_atomic(output / "launch.json", launch)
    write_json_atomic(output / "run.json", run)
    sft_runs.commit()
    try:
        if (launch["prepare_only"] or prepared.get("status") != "prepared"
                or prepared.get("launch_sha256") != digest(launch)):
            raise ValueError("CPU preparation did not succeed for this exact launch")
        if read_json(Path(as_str(launch["prep_dir"])) / "prepare.json") != prepared:
            raise ValueError("CPU preparation receipt changed")
        verify_inputs(launch)
        for root_key, files_key in (("checkpoint_dir", "checkpoint_files"), ("base_snapshot", "base_files")):
            for name, entry in as_dict(prepared[files_key]).items():
                evidence = as_dict(entry)
                path = Path(as_str(prepared[root_key])) / name
                if not path.is_file() or path.stat().st_size != evidence["bytes"]:
                    raise ValueError(f"prepared cache file missing or incomplete: {path}")
                if (launch["adapter_dir"] and root_key == "checkpoint_dir"
                        and sha256_file(path) != evidence["sha256"]):
                    raise ValueError(f"saved checkpoint changed after CPU preparation: {path}")
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "PYTHONUNBUFFERED": "1"}
        with (output / "evaluator.log").open("wb") as log:
            subprocess.run([as_str(part) for part in as_list(run["command"])],
                           env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=max(0.001, deadline - time.monotonic()))
        completed: JsonDict = {"status": "completed",
                               "result": verify_outputs(output, launch)}
        run.update(completed)
    except BaseException as exc:
        run_failure: JsonDict = {"status": "failed", "error": error_record(exc)}
        run.update(run_failure)
    finally:
        run["ended_at"] = now()
        write_json_atomic(output / "run.json", run)
        sft_runs.commit()
    return {"status": run["status"], "output_dir": str(output), "error": run.get("error")}
