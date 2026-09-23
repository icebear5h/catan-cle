from __future__ import annotations

import json
import signal
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import FrameType

from sft.json_types import (
    JsonDict,
    as_dict,
    as_float,
    as_list,
    as_str,
    loads_json,
)
from sft.launchers._train_config import train_config
from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.lora_expansion import expand_lora_bundle
from sft.paths import PROJECT_ROOT
from sft.scripts.builders.build_board_fluency_dataset import validate_dataset
from sft.scripts.train import train_trl_catan_vision as trainer
from sft.scripts.train.train_trl_catan_vision import (
    _message_pair,
    encode_text_pair,
    load_token_inventory,
    sha256_file,
)

from ._config import (
    ABSOLUTE_SECONDS,
    ATOL,
    BASE,
    BASELINE_ROOT,
    BASELINE_SHA256,
    COORDINATOR_SECONDS,
    DATASET,
    PARENT,
    REVIEW,
    REVIEW_SHA256,
    RTOL,
    STAGE_SECONDS,
    STARTUP,
)
from ._data import (
    budget_plan,
    configuration,
    inspect_data,
    progress,
    require_dependencies,
    rows_at,
    source_hashes,
)


def build_plan(run_name: str, budget_usd: float) -> JsonDict:
    config = configuration(run_name)
    budget = budget_plan(budget_usd)
    require_dependencies()
    config.validate()
    if (PROJECT_ROOT / "artifacts/runs/sft" / run_name).exists():
        raise FileExistsError("local run name already used")
    # The data agent owns semantic recomputation/provenance admission. Preserve
    # its real report, then pin the exact bytes the launcher actually uploads.
    validation = validate_dataset(DATASET)
    if validation.get("valid") is not True or validation.get("admitted_for_training") is not True:
        raise ValueError("dataset validator did not admit this corpus for SFT")
    inputs = inspect_data(DATASET, REVIEW)
    root = str(Path(config.output_dir).parent)
    # Serialized immediately below, so any JSON-encodable report may be embedded.
    plan: dict[str, object] = {"schema": "catan_board_fluency_sft_launch/v1", "run_name": run_name,
            "created_at": shared.now(), "reservation_id": uuid.uuid4().hex,
            "root": root, "data_dir": str(Path(config.train_jsonl).parent),
            "parent": PARENT, "base": BASE, "model_revision": shared.MODEL_REVISION,
            "profile": "icebear5h", "hf_secret_name": "huggingface-secret-2",
            "config": asdict(config), "budget": budget, "inputs": inputs,
            "dataset_validation": validation, "source_sha256": source_hashes(),
            "limits": {"stage_seconds": STAGE_SECONDS, "startup_seconds": STARTUP,
                       "coordinator_seconds": COORDINATOR_SECONDS,
                       "absolute_seconds": ABSOLUTE_SECONDS, "retries": 0,
                       "max_containers": 1, "scaledown_seconds": 2},
             "gate_tolerance": {"atol": ATOL, "rtol": RTOL},
             "prevalidation_reference": {"root": BASELINE_ROOT, "records_sha256": BASELINE_SHA256,
                                         "completed_rows": 144, "requested_rows": 190},
             "review_baseline": {"correct": 24, "rows": 200,
                                 "source": "user-approved retained baseline", "review_sha256": REVIEW_SHA256}}
    # Audits contain integer histogram keys. Normalize once before both RPC and
    # persistence so the remote object is identical to its JSON receipt.
    return as_dict(loads_json(json.dumps(plan, allow_nan=False)))


def verify_plan(plan: JsonDict) -> None:
    if plan["config"] != asdict(configuration(as_str(plan["run_name"]))):
        raise ValueError("configuration differs from the approved continuation")
    if plan["budget"] != budget_plan(as_float(as_dict(plan["budget"])["approved_usd"])):
        raise ValueError("budget/rate guard differs")
    if (plan["parent"], plan["base"], plan["model_revision"]) != (PARENT, BASE, shared.MODEL_REVISION):
        raise ValueError("parent or pinned base differs")
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("source files changed after local admission")
    require_dependencies()
    train_config(as_dict(plan["config"])).validate()


def verify_uploaded(plan: JsonDict, *, semantic: bool = False) -> None:
    root = Path(as_str(plan["data_dir"]))
    if shared.read_json(root / "launch.json") != plan:
        raise ValueError("uploaded launch receipt differs")
    check_manifest(root, as_dict(as_dict(plan["inputs"])["files"]))
    if sha256_file(root / "review.jsonl") != REVIEW_SHA256:
        raise ValueError("uploaded review changed")
    if semantic and inspect_data(root, root / "review.jsonl") != plan["inputs"]:
        raise ValueError("uploaded data/inventory differs from local admission")


def check_manifest(root: Path, expected: JsonDict, *, hashes: bool = True) -> None:
    for name, item in expected.items():
        evidence = as_dict(item)
        path = root / name
        if not path.is_file() or path.stat().st_size != evidence["bytes"]:
            raise ValueError(f"prepared file missing/incomplete: {path}")
        if hashes and "sha256" not in evidence and path.suffix != ".safetensors":
            raise ValueError(f"metadata file lacks a content hash: {path}")
        if hashes and "sha256" in evidence and sha256_file(path) != evidence["sha256"]:
            raise ValueError(f"prepared file changed: {path}")


def cached_base_files() -> list[str]:
    """Reuse the evaluator's header audit without calling its downloading snapshot()."""
    base = Path(BASE)
    if not base.is_dir() or base.name != shared.MODEL_REVISION:
        raise FileNotFoundError("the exact approved base snapshot must already be cached")
    index = base / "model.safetensors.index.json"
    weights = ({as_str(name) for name in as_dict(shared.read_json(index)["weight_map"]).values()}
               if index.is_file() else {"model.safetensors"})
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in weights):
        raise ValueError("unexpected cached weight shard path")
    names = weights | {name for name in shared.INFERENCE_SIDECARS if (base / name).is_file()}
    if index.is_file():
        names.add(index.name)
    if "config.json" not in names or not weights:
        raise ValueError("incomplete native base cache")
    if any(not (base / name).is_file() or not (base / name).stat().st_size for name in names):
        raise FileNotFoundError("missing base cache shard/sidefile; downloads are disabled")
    return sorted(names)


def write_rows(path: Path, rows: list[JsonDict]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_work(plan: JsonDict, deadline: float) -> JsonDict:
    verify_uploaded(plan, semantic=True)
    progress("CPU: validating pinned cache, parent and all native text boundaries")
    plan_config = as_dict(plan["config"])
    data_dir = Path(as_str(plan["data_dir"]))
    inventory = load_token_inventory(Path(as_str(plan_config["token_inventory"])))
    parent, parent_files = shared.volume_bundle(PARENT)
    tokenizer, checkpoint = shared.adapter_preflight(parent, inventory, shared.MODEL_ID, shared.MODEL_REVISION)
    parent_config = as_dict(checkpoint["adapter_config"])
    if (parent_config["r"], parent_config["lora_alpha"]) != (8, 16):
        raise ValueError("expected the preserved rank8/alpha16 September 12 checkpoint")
    if checkpoint["saved_model_id"] != BASE:
        raise ValueError("parent must reference the exact approved cache path")
    base_files = cached_base_files()
    base_audit = shared.base_preflight(Path(BASE), base_files, parent, inventory, checkpoint)
    lengths: JsonDict = {}
    for split in ("train", "validation", "test", "validation_eval", "review"):
        items = rows_at(data_dir / f"{split}.jsonl")
        maximum, max_prompt, max_answer = 0, 0, 0
        for row in items:
            prompt, answer = _message_pair(row, line_number=0, input_mode="text")
            pair = encode_text_pair(tokenizer, prompt, answer, max_sequence_length=4096)
            input_ids = as_list(pair["input_ids"])
            prefix = as_list(pair["labels"]).count(-100)
            completion = len(input_ids) - prefix
            if split in ("validation_eval", "review") and (prefix + 512 > 4096 or completion > 512):
                raise ValueError(f"{split} exceeds the greedy512 context allowance")
            maximum, max_prompt, max_answer = max(maximum, len(input_ids)), max(max_prompt, prefix), max(max_answer, completion)
        lengths[split] = {"rows": len(items), "max_sequence": maximum,
                          "max_prompt": max_prompt, "max_completion": max_answer}
        progress(f"CPU: {split} boundaries checked ({len(items)} rows)")
        check_deadline(deadline)
    directory = Path(as_str(plan["root"])) / "prepare"
    panel = {as_str(r.get("id") or r["row_id"]): r
             for r in rows_at(data_dir / "validation_eval.jsonl")}
    write_rows(directory / "teacher120.jsonl",
               [panel[as_str(i)] for i in as_list(as_dict(plan["inputs"])["teacher_ids"])])
    parent_hashes = shared.file_manifest(parent, parent_files)
    expanded = Path(as_str(plan_config["initial_bundle"]))
    if expanded.exists():
        raise FileExistsError("expanded bundle is immutable; choose a fresh run name")
    progress("CPU: expanding the preserved adapter to rank16/alpha32")
    conversion = expand_lora_bundle(parent, expanded, rank=16, alpha=32, seed=44)
    expanded, expanded_files = shared.volume_bundle(str(expanded))
    _, expanded_audit = shared.adapter_preflight(expanded, inventory, shared.MODEL_ID, shared.MODEL_REVISION)
    shared.base_preflight(Path(BASE), base_files, expanded, inventory, expanded_audit)
    expanded_config = as_dict(expanded_audit["adapter_config"])
    if (expanded_config["r"], expanded_config["lora_alpha"]) != (16, 32):
        raise ValueError("converter did not produce rank16/alpha32")
    if (as_dict(expanded_audit["semantic_tokens"])["token_ids"]
            != as_dict(checkpoint["semantic_tokens"])["token_ids"]):
        raise ValueError("conversion changed atlas token IDs")
    check_manifest(parent, parent_hashes)
    expanded_hashes = shared.file_manifest(expanded, expanded_files + ["lora_expansion.json"])
    if expanded_hashes[trainer.VISUAL_STATE_FILE] != parent_hashes[trainer.VISUAL_STATE_FILE]:
        raise ValueError("conversion changed the preserved visual sidefile")
    return {"checkpoint": checkpoint, "base_audit": base_audit, "conversion": conversion,
            "parent_files": parent_hashes, "expanded_files": expanded_hashes,
             "base_files": {name: {"bytes": (Path(BASE) / name).stat().st_size,
                                    **({"sha256": sha256_file(Path(BASE) / name)}
                                       if not name.endswith(".safetensors") else {})}
                            for name in base_files},
            "teacher_sha256": sha256_file(directory / "teacher120.jsonl"), "token_lengths": lengths}


def check_deadline(deadline: float) -> None:
    if time.time() >= deadline:
        raise TimeoutError("fixed stage deadline reached; no automatic extension")


@contextmanager
def deadline_alarm(deadline: float) -> Iterator[None]:
    def expired(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError("inner stage deadline reached; preserving completed artifacts")

    check_deadline(deadline)
    started = time.time()
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    old_handler = signal.signal(signal.SIGALRM, expired)
    allowance = deadline - started
    signal.setitimer(signal.ITIMER_REAL, min(allowance, old_timer[0]) if old_timer[0] else allowance)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        remaining = old_timer[0] - (time.time() - started)
        if old_timer[0] and remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])
