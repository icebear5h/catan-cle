"""One bounded, text-only eval of the unchanged 200-row board-fluency review.

CLI: MODAL_PROFILE=tetracorp .venv/bin/python -m modal run -m sft.modal_board_fluency_eval \
    --hf-repo "$HF_REPO" --hf-revision "$HF_SHA" --run-name "$RUN_NAME" --prepare-only
Omit --prepare-only for CPU preparation followed by exactly one H200 call.
Alternatively, use --adapter-dir /runs/catan-vision-sft/.../checkpoints/checkpoint-128
instead of the HF adapter arguments. CATAN_HF_SECRET_NAME selects the HF secret.
Every invocation needs a fresh run name; preparation reuses the shared HF cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from sft.lora_expansion import expected_adapter_shapes, standard_lora_rank
from sft.modal_catan_vision_sft import (
    HF_SECRET_NAME, hf_cache, sft_data, sft_runs, training_base_image,
)
from sft.scripts import eval_qwen_vl_adapter as evaluator
from sft.scripts.train_trl_catan_vision import (
    RUN_CONFIG_FILE,
    TRAINABLE_SCOPE_FILE,
    VISUAL_STATE_FILE,
    _message_pair,
    assert_runtime_versions,
    discover_components,
    encode_text_pair,
    load_checkpoint_text_tokenizer,
    load_token_inventory,
    language_linear_targets,
    sha256_file,
    write_json_atomic,
)


MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DEFAULT_REVIEW = "artifacts/generated/sft/symbolic_board_fluency_review_v1/review.jsonl"
DEFAULT_INVENTORY = "artifacts/generated/sft/symbolic_board_v2/trainable_tokens.json"
ORIGINAL_INVENTORY = (
    "artifacts/generated/board_recognition/replay_v1/"
    "ms_swift_bidirectional_v1/trainable_tokens.json"
)
ROWS = 200
CONTEXT = 4096
NEW_TOKENS = 512
BATCH = 16
GPU_DEADLINE = 840
CACHE = "/cache/huggingface/hub"
VOLUMES = {"/cache": hf_cache, "/data": sft_data, "/runs": sft_runs}
REQUIRED_BUNDLE = {
    "adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json",
    "tokenizer.json", RUN_CONFIG_FILE, TRAINABLE_SCOPE_FILE, VISUAL_STATE_FILE,
}
INFERENCE_SIDECARS = {
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json",
    "merges.txt", "chat_template.jinja", "chat_template.json", "processor_config.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
}

app = modal.App("catan-board-fluency-eval")
# The symbolic scorer imports the board data package, whose source-contract
# helpers also import the viewer's detached ServerState definition.
eval_image = training_base_image.pip_install("jsonschema==4.25.1", "pydantic==2.12.3").add_local_python_source(
    "cle", "evals", "sft", "data_pipeline", "playground",
)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

with eval_image.imports():
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig
    from peft.tuners.tuners_utils import check_target_module_exists
    from safetensors import safe_open
    from transformers import AddedToken, AutoConfig, AutoModelForMultimodalLM, AutoTokenizer


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def error_record(exc: BaseException) -> dict:
    message = str(exc)
    if token := os.environ.get("HF_TOKEN"):
        message = message.replace(token, "[REDACTED]")
    message = re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", message)
    return {"type": type(exc).__name__, "message": message[:1500]}


def validate_cli(hf_repo: str, hf_revision: str, model_id: str,
                 model_revision: str, run_name: str, checkpoint_subdir: str,
                 adapter_dir: str = "") -> None:
    if adapter_dir:
        if hf_repo or hf_revision or checkpoint_subdir:
            raise ValueError("--adapter-dir is mutually exclusive with HF adapter arguments")
        path = PurePosixPath(adapter_dir)
        if (not path.is_relative_to("/runs/catan-vision-sft") or len(path.parts) <= 3
                or ".." in path.parts or str(path) != adapter_dir):
            raise ValueError("--adapter-dir must be an exact absolute path below /runs/catan-vision-sft")
    elif not hf_repo or not hf_revision:
        raise ValueError("provide --adapter-dir OR both --hf-repo and --hf-revision")
    revisions = [("model-revision", model_revision)]
    if not adapter_dir:
        revisions.append(("hf-revision", hf_revision))
    for label, value in revisions:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError(f"--{label} must be an immutable lowercase 40-character commit SHA")
    for value in ([model_id] if adapter_dir else [hf_repo, model_id]):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("model repositories must be explicit owner/repo IDs")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", run_name):
        raise ValueError("--run-name must be unique and contain only letters, digits, _ or -")
    if checkpoint_subdir and (
        PurePosixPath(checkpoint_subdir).is_absolute()
        or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part)
               for part in checkpoint_subdir.split("/"))
    ):
        raise ValueError("--checkpoint-subdir must be a relative directory without traversal")


def inspect_inputs(review: Path, inventory: Path) -> tuple[list[dict], dict]:
    """Validate stored gold through the actual scorer, without regenerating answers."""
    load_token_inventory(inventory)
    rows = [row for _, row in evaluator.iter_jsonl(review)]
    ids, families, operations = [], Counter(), Counter()
    for line, row in enumerate(rows, 1):
        row_id = row.get("id") or row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"row {line} needs a nonempty string ID")
        _, answer = _message_pair(row, line_number=line, input_mode="text")
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        score = evaluator.score_response(answer, answer, metadata=metadata)
        if score["scoring"] != evaluator.BOARD_FLUENCY_SCHEMA or score["correct"] is not True:
            raise ValueError(f"unscorable review/v1 stored gold: {row_id}")
        ids.append(row_id)
        families[metadata["family"]] += 1
        operations[metadata["operation"]] += 1
    if len(ids) != ROWS or len(set(ids)) != ROWS:
        raise ValueError(f"expected exactly {ROWS} unique review IDs, found {len(ids)} rows")
    return rows, {
        "rows": ROWS, "ids": ids, "review_sha256": sha256_file(review),
        "inventory_sha256": sha256_file(inventory), "by_family": dict(families),
        "by_operation": dict(operations),
    }


def verify_inputs(launch: dict) -> list[dict]:
    root = Path(launch["data_dir"])
    if read_json(root / "launch.json") != launch:
        raise ValueError("uploaded launch receipt differs")
    rows, contract = inspect_inputs(root / "review.jsonl", root / "trainable_tokens.json")
    if contract != launch["inputs"]:
        raise ValueError("uploaded JSONL or inventory differs from the local validation")
    return rows


def snapshot(repo: str, revision: str, *, subdir: str = "", adapter: bool) -> tuple[Path, list[str]]:
    """Download only enumerated inference files; existing immutable blobs are reused."""
    prefix = subdir + "/" if subdir else ""
    files = {
        name.removeprefix(prefix)
        for name in HfApi().list_repo_files(repo, revision=revision)
        if name.startswith(prefix)
    }
    if adapter:
        missing = REQUIRED_BUNDLE - files
        if missing or any(name.startswith("frozen_adapter/") for name in files):
            raise ValueError(
                "Expected a standalone PEFT inference bundle at the selected subdirectory; "
                f"missing={sorted(missing)}. Merged-only models and frozen-parent bundles "
                "are incompatible with this evaluator."
            )
        selected = REQUIRED_BUNDLE | (files & INFERENCE_SIDECARS)
    else:
        weights = {name for name in files if re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", name)}
        if "config.json" not in files or not weights:
            raise ValueError("base repository requires native config and safetensors inference weights")
        selected = weights | (files & INFERENCE_SIDECARS)
        if "model.safetensors.index.json" in files:
            selected.add("model.safetensors.index.json")
        elif weights != {"model.safetensors"}:
            raise ValueError("sharded base weights require model.safetensors.index.json")
    selected |= {name for name in files if re.fullmatch(r"chat_templates/[^/]+\.jinja", name)}
    root = Path(snapshot_download(
        repo_id=repo, revision=revision, cache_dir=CACHE,
        allow_patterns=sorted(prefix + name for name in selected),
        max_workers=8, force_download=False,
    ))
    if root.name != revision:
        raise ValueError("HF snapshot did not resolve to the requested immutable revision")
    root = root / subdir if subdir else root
    for name in selected:
        if not (root / name).is_file() or (root / name).stat().st_size == 0:
            raise FileNotFoundError(f"incomplete inference file: {root / name}")
    return root, sorted(selected)


def tensor_headers(path: Path) -> dict:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: {"shape": handle.get_slice(key).get_shape(),
                      "dtype": handle.get_slice(key).get_dtype()} for key in handle.keys()}


def volume_bundle(adapter_dir: str) -> tuple[Path, list[str]]:
    """Inspect the exact saved bundle in place; never select or modify a checkpoint."""
    bundle = Path(adapter_dir)
    if not bundle.is_dir():
        raise FileNotFoundError(f"saved adapter directory does not exist: {bundle}")
    files = {name for name in REQUIRED_BUNDLE | INFERENCE_SIDECARS if (bundle / name).is_file()}
    missing = REQUIRED_BUNDLE - files
    if missing or (bundle / "frozen_adapter").exists():
        raise ValueError(f"incompatible saved PEFT bundle: missing={sorted(missing)}; frozen parents unsupported")
    files |= {str(path.relative_to(bundle)) for path in (bundle / "chat_templates").glob("*.jinja")
              if path.is_file()}
    for name in files:
        if (bundle / name).stat().st_size == 0:
            raise ValueError(f"empty inference file: {bundle / name}")
    return bundle, sorted(files)


def adapter_preflight(bundle: Path, inventory: dict, model_id: str, model_revision: str) -> tuple[Any, dict]:
    """Check the evaluator's sidefile contract before downloading the 27B base."""
    saved = read_json(bundle / "adapter_config.json")
    parent = read_json(bundle / RUN_CONFIG_FILE)
    rank = standard_lora_rank(saved, parent)
    pinned_snapshot = str(Path(CACHE) / ("models--" + model_id.replace("/", "--")) / "snapshots" / model_revision)
    scope = read_json(bundle / TRAINABLE_SCOPE_FILE)
    tokenizer = load_checkpoint_text_tokenizer(bundle, inventory["tokens"])
    semantic = scope["semantic_tokens"]
    indices = saved.get("trainable_token_indices")
    if (
        not isinstance(saved.get("target_modules"), (list, str))
        or not isinstance(indices, dict) or len(indices) != 2
        or any(ids != semantic["token_ids"] for ids in indices.values())
        or parent.get("model_id") not in (model_id, pinned_snapshot) or scope.get("errors")
        or (bundle / "frozen_adapter").exists() or (bundle / "frozen_bundle.json").exists()
    ):
        raise ValueError(f"incompatible language rank-{rank} LoRA / atlas rows / base-model sidefiles")
    headers = tensor_headers(bundle / "adapter_model.safetensors")
    rows = [value["shape"] for key, value in headers.items() if "trainable_tokens_delta" in key]
    if len(rows) != 2 or any(len(shape) != 2 or shape[0] != 154 for shape in rows):
        raise ValueError("adapter weights must contain both 154-row atlas tensors")
    a = {name.removesuffix(".lora_A.weight"): value["shape"]
         for name, value in headers.items() if name.endswith(".lora_A.weight")}
    b = {name.removesuffix(".lora_B.weight"): value["shape"]
         for name, value in headers.items() if name.endswith(".lora_B.weight")}
    if (not a or a.keys() != b.keys()
            or any(len(shape) != 2 or shape[0] != rank for shape in a.values())
            or any(len(shape) != 2 or shape[1] != rank for shape in b.values())):
        raise ValueError(f"adapter weights must contain paired rank-{rank} LoRA tensors")
    return tokenizer, {"adapter_config": saved, "semantic_tokens": semantic,
                       "lora_rank": rank, "lora_alpha": saved["lora_alpha"],
                       "saved_model_id": parent["model_id"],
                       "training_profile": parent["profile"], "adapter_tensors": len(headers),
                       "tokenizer_size": len(tokenizer)}


def base_preflight(base: Path, files: list[str], bundle: Path, inventory: dict,
                   checkpoint: dict) -> dict:
    """Inspect tensor headers and configuration only; never instantiate the base model."""
    config = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    text_config = getattr(config, "text_config", config)
    if getattr(text_config, "max_position_embeddings", CONTEXT) < CONTEXT:
        raise ValueError("base model context is smaller than 4096")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    tokenizer.add_tokens([AddedToken(token, normalized=False, special=False) for token in inventory["tokens"]])
    if [tokenizer.encode(token, add_special_tokens=False) for token in inventory["tokens"]] != [
        [token_id] for token_id in checkpoint["semantic_tokens"]["token_ids"]
    ]:
        raise ValueError("pinned base tokenizer and checkpoint atlas token IDs differ")
    headers, locations = {}, {}
    for name in files:
        if name.endswith(".safetensors"):
            shard = tensor_headers(base / name)
            if headers.keys() & shard.keys():
                raise ValueError("duplicate tensors across base weight files")
            headers.update(shard)
            locations.update(dict.fromkeys(shard, name))
    index_path = base / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = read_json(index_path)["weight_map"]
        if weight_map != locations:
            raise ValueError("base weight index is incomplete")
    adapter_config = LoraConfig.from_pretrained(bundle, local_files_only=True)
    # Check the actual inference architecture without allocating model weights.
    # The HF files also contain training-only MTP tensors that are not modules
    # in AutoModelForMultimodalLM and must not become false-positive LoRA targets.
    with torch.device("meta"):
        architecture = AutoModelForMultimodalLM.from_config(
            config, dtype=torch.bfloat16, attn_implementation="sdpa",
        )
    intended = set(language_linear_targets(architecture, discover_components(architecture)))
    modules = {name for name, _ in architecture.named_modules() if name}
    matched = {name for name in modules if check_target_module_exists(adapter_config, name)}
    del architecture
    if not intended or intended != matched:
        raise ValueError("LoRA targets must match exactly the pinned base's language linear layers")
    rank = standard_lora_rank(checkpoint["adapter_config"], read_json(bundle / RUN_CONFIG_FILE))
    row_modules = checkpoint["adapter_config"]["trainable_token_indices"]
    if set(row_modules) != {"model.language_model.embed_tokens", "lm_head"}:
        raise ValueError("expected both Qwen language input and output atlas rows")
    token_widths = {}
    for module in row_modules:
        shape = headers.get(module + ".weight", {}).get("shape", [])
        if len(shape) != 2:
            raise ValueError("unsupported atlas embedding/head module")
        token_widths[module] = shape[1]
    expected = expected_adapter_shapes(
        {module: headers[module + ".weight"]["shape"] for module in intended}, token_widths, rank,
    )
    # Match prepare_semantic_tokens: resize to the saved tokenizer, padded to 128 rows.
    current_vocab = headers["model.language_model.embed_tokens.weight"]["shape"][0]
    runtime_vocab = ((max(checkpoint["tokenizer_size"], current_vocab) + 127) // 128) * 128
    ids = checkpoint["semantic_tokens"]["token_ids"]
    if min(ids) < 0 or max(ids) >= runtime_vocab:
        raise ValueError("atlas token IDs exceed the evaluator's resized embedding/head slots")
    actual = {name: value["shape"] for name, value in tensor_headers(bundle / "adapter_model.safetensors").items()}
    if actual != expected:
        raise ValueError("adapter tensor names/shapes do not match complete language LoRA plus atlas rows")
    visual = tensor_headers(bundle / VISUAL_STATE_FILE)
    expected_visual = {f"base_model.model.{name}": value["shape"] for name, value in headers.items()
                       if name.startswith("model.visual.")}
    if not expected_visual or {name: value["shape"] for name, value in visual.items()} != expected_visual:
        raise ValueError("saved visual sidefile is incomplete or incompatible with the pinned base")
    return {"model_type": config.model_type, "base_tensors": len(headers),
            "runtime_vocab_size": runtime_vocab,
            "language_lora_modules": len(intended), "visual_tensors": len(visual),
            "visual_dtypes": dict(Counter(value["dtype"] for value in visual.values()))}


def file_manifest(root: Path, names: list[str]) -> dict:
    return {name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
            for name in names}


@app.function(
    image=eval_image, volumes=VOLUMES, secrets=[hf_secret],
    cpu=(4.0, 4.0), memory=(16 * 1024, 16 * 1024), timeout=1200,
    startup_timeout=300, retries=0, max_containers=1, scaledown_window=2,
)
def prepare_cpu(launch: dict) -> dict:
    for volume in VOLUMES.values():
        volume.reload()
    receipt_path = Path(launch["prep_dir"]) / "prepare.json"
    if receipt_path.parent.exists() or Path(launch["output_dir"]).exists():
        raise FileExistsError("remote run name already used; choose a fresh --run-name")
    receipt_path.parent.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "preparing", "launch_sha256": digest(launch), "started_at": now()}
    write_json_atomic(receipt_path, receipt)
    sft_runs.commit()
    try:
        rows = verify_inputs(launch)
        receipt["dependencies"] = assert_runtime_versions()
        inventory = load_token_inventory(Path(launch["data_dir"]) / "trainable_tokens.json")
        if launch["adapter_dir"]:
            bundle, adapter_files = volume_bundle(launch["adapter_dir"])
        else:
            bundle, adapter_files = snapshot(launch["hf_repo"], launch["hf_revision"],
                                             subdir=launch["checkpoint_subdir"], adapter=True)
        tokenizer, checkpoint = adapter_preflight(bundle, inventory, launch["model_id"], launch["model_revision"])
        lengths = []
        for line, row in enumerate(rows, 1):
            prompt, answer = _message_pair(row, line_number=line, input_mode="text")
            pair = encode_text_pair(tokenizer, prompt, answer, max_sequence_length=CONTEXT)
            prompt_tokens = pair["labels"].count(-100)
            completion_tokens = len(pair["input_ids"]) - prompt_tokens
            if prompt_tokens + NEW_TOKENS > CONTEXT or completion_tokens > NEW_TOKENS:
                raise ValueError(f"row {line} exceeds the 4096 context / 512 completion budget")
            lengths.append({"id": row.get("id") or row.get("row_id"),
                            "prompt": prompt_tokens, "gold_completion": completion_tokens})
        receipt.update(checkpoint_dir=str(bundle), checkpoint=checkpoint, token_lengths=lengths,
                       max_prompt_tokens=max(item["prompt"] for item in lengths),
                       max_gold_tokens=max(item["gold_completion"] for item in lengths),
                       checkpoint_files=file_manifest(bundle, adapter_files))
        write_json_atomic(receipt_path, receipt)
        hf_cache.commit()
        sft_runs.commit()
        base, base_files = snapshot(launch["model_id"], launch["model_revision"], adapter=False)
        receipt["base_audit"] = base_preflight(base, base_files, bundle, inventory, checkpoint)
        receipt.update(base_snapshot=str(base), base_files=file_manifest(base, base_files), status="prepared")
    except BaseException as exc:
        receipt.update(status="failed", error=error_record(exc))
    finally:
        receipt["ended_at"] = now()
        write_json_atomic(receipt_path, receipt)
        try:
            hf_cache.commit()
        finally:
            sft_runs.commit()
    return receipt


def eval_command(launch: dict, prepared: dict) -> list[str]:
    return [
        sys.executable, "-m", "sft.scripts.eval_qwen_vl_adapter",
        "--eval-jsonl", launch["data_dir"] + "/review.jsonl",
        "--token-inventory", launch["data_dir"] + "/trainable_tokens.json",
        "--adapter-dir", prepared["checkpoint_dir"], "--output-dir", launch["output_dir"],
        "--model-id", prepared["checkpoint"]["saved_model_id"], "--model-revision", launch["model_revision"],
        "--bits", "16", "--input-mode", "text", "--max-sequence-length", str(CONTEXT),
        "--batch-size", str(BATCH), "--long-batch-size", str(BATCH),
        "--max-new-tokens", str(NEW_TOKENS), "--long-max-new-tokens", str(NEW_TOKENS),
        "--no-candidate-scoring", "--disable-flash-attn2",
    ]


def verify_outputs(output: Path, launch: dict) -> dict:
    records = [row for _, row in evaluator.iter_jsonl(output / "records.jsonl")]
    ids = [record.get("id") for record in records]
    if len(ids) != ROWS or len(set(ids)) != ROWS or set(ids) != set(launch["inputs"]["ids"]):
        raise ValueError("evaluation did not return exactly the expected 200 IDs")
    if any(not isinstance(record.get("response"), str)
           or record.get("score", {}).get("scoring") != evaluator.BOARD_FLUENCY_SCHEMA
           or type(record.get("score", {}).get("correct")) is not bool for record in records):
        raise ValueError("evaluation contains incomplete/unscored records")
    summary = read_json(output / "summary.json")
    pinned_snapshot = str(Path(CACHE) / ("models--" + launch["model_id"].replace("/", "--"))
                          / "snapshots" / launch["model_revision"])
    if summary.get("model_id") not in (launch["model_id"], pinned_snapshot):
        raise ValueError("summary base model differs from the pinned model/revision")
    if summary.get("rows") != ROWS or summary.get("attempted") != ROWS:
        raise ValueError("summary is incomplete")
    for key, expected in {
        "model_revision": launch["model_revision"],
        "input_mode": "text", "bits": 16, "max_sequence_length": CONTEXT,
        "candidate_scoring": False, "batch_size": BATCH,
        "max_new_tokens": NEW_TOKENS, "long_max_new_tokens": NEW_TOKENS,
    }.items():
        if summary.get(key) != expected:
            raise ValueError(f"summary inference condition differs: {key}")
    for dimension in ("family", "operation"):
        if {key: value["total"] for key, value in summary[f"by_{dimension}"].items()} != launch["inputs"][f"by_{dimension}"]:
            raise ValueError(f"summary by_{dimension} differs from the validated input")
    return {"rows": ROWS, "exact_accuracy": summary["exact_accuracy"],
            "files": file_manifest(output, ["records.jsonl", "summary.json"])}


@app.function(
    image=eval_image, volumes=VOLUMES, gpu="H200", cpu=(8.0, 8.0),
    memory=(64 * 1024, 64 * 1024), timeout=900, startup_timeout=300,
    retries=0, max_containers=1, scaledown_window=2,
)
def eval_h200(launch: dict, prepared: dict) -> dict:
    deadline = time.monotonic() + GPU_DEADLINE
    for volume in VOLUMES.values():
        volume.reload()
    output = Path(launch["output_dir"])
    # A committed marker precedes model loading; repeated/redelivered calls fail closed.
    output.mkdir(parents=True, exist_ok=False)
    run = {"status": "running", "started_at": now(), "launch": launch,
           "prepare_sha256": digest(prepared), "command": eval_command(launch, prepared)}
    write_json_atomic(output / "launch.json", launch)
    write_json_atomic(output / "run.json", run)
    sft_runs.commit()
    try:
        if (launch["prepare_only"] or prepared.get("status") != "prepared"
                or prepared.get("launch_sha256") != digest(launch)):
            raise ValueError("CPU preparation did not succeed for this exact launch")
        if read_json(Path(launch["prep_dir"]) / "prepare.json") != prepared:
            raise ValueError("CPU preparation receipt changed")
        verify_inputs(launch)
        for root_key, files_key in (("checkpoint_dir", "checkpoint_files"), ("base_snapshot", "base_files")):
            for name, evidence in prepared[files_key].items():
                path = Path(prepared[root_key]) / name
                if not path.is_file() or path.stat().st_size != evidence["bytes"]:
                    raise ValueError(f"prepared cache file missing or incomplete: {path}")
                if (launch["adapter_dir"] and root_key == "checkpoint_dir"
                        and sha256_file(path) != evidence["sha256"]):
                    raise ValueError(f"saved checkpoint changed after CPU preparation: {path}")
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "PYTHONUNBUFFERED": "1"}
        with (output / "evaluator.log").open("wb") as log:
            subprocess.run(run["command"], env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=max(0.001, deadline - time.monotonic()))
        run.update(status="completed", result=verify_outputs(output, launch))
    except BaseException as exc:
        run.update(status="failed", error=error_record(exc))
    finally:
        run["ended_at"] = now()
        write_json_atomic(output / "run.json", run)
        sft_runs.commit()
    return {"status": run["status"], "output_dir": str(output), "error": run.get("error")}


def download_file(remote: str, local: Path) -> bool:
    """Volume.read_file yields byte chunks, not one bytes object."""
    try:
        chunks = iter(sft_runs.read_file(remote))
        first = next(chunks, b"")
    except FileNotFoundError:
        return False
    with local.open("xb") as handle:
        handle.write(first)
        for chunk in chunks:
            handle.write(chunk)
    return True


@app.local_entrypoint()
def main(
    run_name: str,
    hf_repo: str = "",
    hf_revision: str = "",
    adapter_dir: str = "",
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    checkpoint_subdir: str = "",
    eval_jsonl: str = DEFAULT_REVIEW,
    token_inventory: str = "",
    prepare_only: bool = False,
) -> None:
    """Prepare immutable HF caches, then evaluate once; --prepare-only uses CPU only."""
    validate_cli(hf_repo, hf_revision, model_id, model_revision, run_name, checkpoint_subdir, adapter_dir)
    review = Path(eval_jsonl).expanduser().resolve()
    inventory = Path(token_inventory or DEFAULT_INVENTORY).expanduser()
    if not token_inventory and not inventory.is_file():
        inventory = Path(ORIGINAL_INVENTORY)
    inventory = inventory.resolve()
    _, contract = inspect_inputs(review, inventory)
    review_bytes, inventory_bytes = review.read_bytes(), inventory.read_bytes()
    if (hashlib.sha256(review_bytes).hexdigest() != contract["review_sha256"]
            or hashlib.sha256(inventory_bytes).hexdigest() != contract["inventory_sha256"]):
        raise ValueError("local inputs changed during validation")
    local = Path("artifacts/runs/sft") / run_name
    local.mkdir(parents=True, exist_ok=False)
    launch = {
        "schema": "catan_board_fluency_eval_launch/v1", "created_at": now(),
        "run_name": run_name, "hf_repo": hf_repo, "hf_revision": hf_revision,
        "adapter_dir": adapter_dir, "hf_secret_name": HF_SECRET_NAME,
        "checkpoint_subdir": checkpoint_subdir, "model_id": model_id,
        "model_revision": model_revision, "prepare_only": prepare_only, "inputs": contract,
        "local_review": str(review), "local_inventory": str(inventory),
        "data_dir": f"/data/board-fluency-eval/{run_name}-{uuid.uuid4().hex}",
        "prep_dir": f"/runs/board-fluency-prep/{run_name}",
        "output_dir": f"/runs/board-fluency-eval/{run_name}",
        "source_sha256": {str(Path(module.__file__).name): sha256_file(Path(module.__file__))
                          for module in (sys.modules[__name__], evaluator)},
        "limits": {"gpu": "H200", "cpu": 8, "memory_gib": 64, "execution_seconds": 900,
                   "startup_seconds": 300, "inner_seconds": GPU_DEADLINE, "retries": 0,
                   "max_containers": 1, "cpu_prepare_seconds": 1200,
                   "context": CONTEXT, "new_tokens": NEW_TOKENS, "batch_size": BATCH},
    }
    write_json_atomic(local / "launch.json", launch)
    state = {"status": "uploading", "launch_sha256": digest(launch), "gpu_calls_requested": 0}
    write_json_atomic(local / "orchestration.json", state)
    try:
        remote = launch["data_dir"].removeprefix("/data")
        with sft_data.batch_upload(force=False) as batch:
            batch.put_file(io.BytesIO(review_bytes), remote + "/review.jsonl")
            batch.put_file(io.BytesIO(inventory_bytes), remote + "/trainable_tokens.json")
            batch.put_file(local / "launch.json", remote + "/launch.json")
        state["status"] = "preparing"
        write_json_atomic(local / "orchestration.json", state)
        prepared = prepare_cpu.remote(launch)
        write_json_atomic(local / "prepare.json", prepared)
        if prepared["status"] != "prepared":
            raise RuntimeError(f"CPU preparation failed: {prepared.get('error')}")
        print(json.dumps({"status": "prepared", "checkpoint": prepared["checkpoint_dir"],
                          "base": prepared["base_snapshot"], "max_prompt_tokens": prepared["max_prompt_tokens"],
                          "max_gold_tokens": prepared["max_gold_tokens"], "local_dir": str(local)}, indent=2))
        if prepare_only:
            state["status"] = "prepared_only"
            return
        state.update(status="gpu_requested", gpu_calls_requested=1)
        write_json_atomic(local / "orchestration.json", state)
        result = eval_h200.remote(launch, prepared)
        state.update(status=result["status"], result=result)
        if result["status"] != "completed":
            raise RuntimeError(f"GPU eval failed; no retry: {result.get('error')}")
        state["status"] = "downloading"
    except BaseException as exc:
        state.update(status="failed", error=error_record(exc))
        raise
    finally:
        download_errors = {}
        targets = {"prepare.json": launch["prep_dir"] + "/prepare.json"}
        if state["gpu_calls_requested"]:
            targets.update({name: launch["output_dir"] + "/" + name
                            for name in ("records.jsonl", "summary.json", "run.json")})
            if state["status"] == "failed":
                targets["evaluator.log"] = launch["output_dir"] + "/evaluator.log"
        for name, remote_path in targets.items():
            if (local / name).exists():
                continue
            try:
                if not download_file(remote_path.removeprefix("/runs"), local / name):
                    download_errors[name] = "not present on remote volume"
            except Exception as exc:
                download_errors[name] = error_record(exc)
        state.update(ended_at=now(), download_errors=download_errors)
        write_json_atomic(local / "orchestration.json", state)
    try:
        if state["download_errors"]:
            raise RuntimeError(f"artifact download incomplete; do not rerun GPU: {state['download_errors']}")
        verification = verify_outputs(local, launch)
        run = read_json(local / "run.json")
        if (run["status"] != "completed" or run["launch"] != launch
                or run["prepare_sha256"] != digest(prepared) or run["result"] != verification):
            raise ValueError("downloaded run receipt or artifact hashes differ")
        state["status"] = "completed"
    except Exception as exc:
        state.update(status="artifact_verification_failed", error=error_record(exc))
        raise
    finally:
        write_json_atomic(local / "orchestration.json", state)
    print(json.dumps({"status": "completed", "local_dir": str(local), **verification}, indent=2))
