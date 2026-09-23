from __future__ import annotations

import inspect
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

from sft.json_types import JsonDict, JsonValue, as_dict, as_int, as_list, as_str
from sft.launchers.board_fluency import modal_board_fluency_eval as shared
from sft.paths import PROJECT_ROOT
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train import train_trl_catan_vision as trainer
from sft.scripts.train.train_trl_catan_vision import (
    TrainConfig,
    _message_pair,
    load_token_inventory,
    sha256_file,
)

from ._config import (
    BASE,
    CONTROL_CPU,
    COORDINATOR_SECONDS,
    DATA_FILES,
    PARENT,
    PRIOR_ATTEMPTS_USD,
    REVIEW_SHA256,
    SOURCE_FILES,
    STAGE_SECONDS,
    STARTUP,
)


def progress(message: str) -> None:
    print(f"[board-fluency {shared.now()}] {message}", flush=True)


def budget_plan(budget_usd: float) -> JsonDict:
    if not math.isfinite(budget_usd) or not 0 < budget_usd <= 15:
        raise ValueError("--budget-usd must be positive and at most the approved $15")
    gpu_rate = 0.001261 + 16 * 0.0000131 + 128 * 0.00000222
    gpu_seconds = sum(STAGE_SECONDS[s] + STARTUP for s in ("gate", "train", "posteval"))
    gpu_max = gpu_seconds * gpu_rate
    cpu_max = (STAGE_SECONDS["prepare"] + STARTUP) * (4 * 0.0000131 + 16 * 0.00000222)
    coordinator_max = (COORDINATOR_SECONDS + STARTUP) * (CONTROL_CPU * 0.0000131 + 2 * 0.00000222)
    # Includes idle scaledown, cancellation RPC/termination grace and reservation.
    reserve = 0.75
    upper = gpu_max + cpu_max + coordinator_max + reserve + PRIOR_ATTEMPTS_USD
    if upper > budget_usd:
        raise ValueError(f"bounded compute envelope ${upper:.6f} exceeds ${budget_usd:.2f}")
    return {"approved_usd": budget_usd, "rates_checked": "2026-09-15",
            "h200_per_second": 0.001261, "cpu_core_per_second": 0.0000131,
            "memory_gib_per_second": 0.00000222, "gpu_stage_per_second": gpu_rate,
            "gpu_seconds_including_startups": gpu_seconds, "gpu_upper_usd": gpu_max,
            "prepare_upper_usd": cpu_max, "coordinator_upper_usd": coordinator_max,
             "termination_and_control_reserve_usd": reserve, "compute_upper_usd": upper,
             "prior_attempts_allowance_usd": PRIOR_ATTEMPTS_USD,
            "excluded": "unrelated jobs, storage, image storage/build charges and subscriptions",
            "policy": "fixed checked rates; no automatic retry, extension or replacement stage"}


def configuration(run_name: str) -> TrainConfig:
    shared.validate_cli("", "", shared.MODEL_ID, shared.MODEL_REVISION, run_name, "", PARENT)
    root = f"/runs/catan-vision-sft/{run_name}"
    if root == str(Path(PARENT).parents[1]):
        raise ValueError("run name collides with the preserved parent")
    data = f"/data/board-fluency-sft/{run_name}"
    return TrainConfig(
        train_jsonl=data + "/train.jsonl", token_inventory=data + "/trainable_tokens.json",
        output_dir=root + "/training", eval_jsonl=root + "/prepare/teacher120.jsonl",
        model_id=BASE, profile="vision_tokens_lora", input_mode="text",
        max_sequence_length=4096, initial_bundle=root + "/expanded-r16", token_init="keep",
        max_steps=128, per_device_train_batch_size=4, gradient_accumulation_steps=2,
        per_device_eval_batch_size=4, learning_rate=1e-4, language_lora_learning_rate=5e-5,
        lora_rank=16, lora_alpha=32, lora_dropout=0.05, save_steps=32, eval_steps=32,
        save_total_limit=4, dataloader_num_workers=0, seed=44, require_curriculum=False,
        resume_from_checkpoint=None, publish_to_hub=False,
    )


def require_dependencies() -> None:
    callback = inspect.signature(trainer.run_training).parameters.get("extra_callbacks")
    if callback is None or callback.kind != inspect.Parameter.KEYWORD_ONLY:
        raise RuntimeError("run_training(config, *, extra_callbacks=None) is required")


def source_hashes() -> JsonDict:
    return {name: sha256_file(PROJECT_ROOT / name) for name in SOURCE_FILES}


def rows_at(path: Path) -> list[JsonDict]:
    return [as_dict(row) for _, row in evaluator.iter_jsonl(path)]


def row_contract(rows: list[JsonDict]) -> JsonDict:
    ids: list[JsonValue] = []
    operations: Counter[str] = Counter()
    families: Counter[str] = Counter()
    states: set[str] = set()
    for line, row in enumerate(rows, 1):
        row_id = row.get("id") or row.get("row_id")
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(f"missing row ID at line {line}")
        _, gold = _message_pair(row, line_number=line, input_mode="text")
        metadata = evaluator.evaluation_metadata(row, image_variant="original")
        score = evaluator.score_response(gold, gold, metadata=metadata)
        if score.get("correct") is not True or score.get("scoring") != metadata.get("schema"):
            raise ValueError(f"stored gold did not use the fluency schema scorer: {row_id}")
        state = as_dict(row.get("metadata", {})).get("state_sha256")
        if not isinstance(state, str) or len(state) != 64:
            raise ValueError(f"missing canonical state identity: {row_id}")
        states.add(state)
        ids.append(row_id)
        operations[as_str(metadata["operation"])] += 1
        families[as_str(metadata["family"])] += 1
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty split or duplicate row IDs")
    return {"rows": len(rows), "ids": ids, "state_sha256": list[JsonValue](sorted(states)),
            "by_operation": dict[str, JsonValue](operations),
            "by_family": dict[str, JsonValue](families)}


def teacher_ids(rows: list[JsonDict]) -> list[str]:
    """First 120 in deterministic operation-balanced order; never sample per eval."""
    groups: defaultdict[str, deque[str]] = defaultdict(deque)
    for row in rows:
        groups[as_str(as_dict(row["metadata"])["operation"])].append(
            as_str(row.get("id") or row["row_id"]))
    chosen: list[str] = []
    counts: Counter[str] = Counter()
    while len(chosen) < 120:
        for operation in sorted(groups):
            if groups[operation] and len(chosen) < 120:
                chosen.append(groups[operation].popleft())
                counts[operation] += 1
        if not any(groups.values()) and len(chosen) < 120:
            raise ValueError("validation_eval needs at least 120 rows")
    # Two pip operations have only five independent heldout layouts each.
    # Saturate those real examples rather than duplicate them to pad a quota.
    unsaturated = [counts[op] for op, remaining in groups.items() if remaining]
    if (not unsaturated or max(unsaturated) - min(unsaturated) > 1
            or set(counts) != set(groups)):
        raise ValueError("validation_eval cannot supply a balanced fixed teacher-forced panel")
    return chosen


def inspect_data(root: Path, review: Path) -> JsonDict:
    inventory = load_token_inventory(root / "trainable_tokens.json")
    if sha256_file(review) != REVIEW_SHA256:
        raise ValueError("the unchanged 200-row review hash differs")
    panels = {name: row_contract(rows_at(root / f"{name}.jsonl"))
              for name in ("train", "validation", "test", "validation_eval")}
    panels["review"] = row_contract(rows_at(review))
    rows = {name: as_int(panel["rows"]) for name, panel in panels.items()}
    states = {name: {as_str(state) for state in as_list(panel["state_sha256"])}
              for name, panel in panels.items()}
    if (rows["train"] != 3200 or len(as_dict(panels["train"]["by_operation"])) != 20
            or rows["validation_eval"] != 190 or rows["review"] != 200):
        raise ValueError("expected train3200/20 operations, validation_eval190 and review200")
    for name in ("validation", "test"):
        if not 300 <= rows[name] <= 450:
            raise ValueError(f"{name} differs from the approved approximately 370 rows")
    for name in ("train", "validation", "test", "validation_eval"):
        if states[name] & states["review"]:
            raise ValueError(f"review state leaked into {name}")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if states[a] & states[b]:
            raise ValueError(f"state leakage between {a} and {b}")
    validation = {r.get("id") or r["row_id"]: r for r in rows_at(root / "validation.jsonl")}
    eval_rows = rows_at(root / "validation_eval.jsonl")
    if any(validation.get(r.get("id") or r["row_id"]) != r for r in eval_rows):
        raise ValueError("validation_eval must contain unchanged validation rows")
    return {"panels": dict[str, JsonValue](panels),
            "teacher_ids": list[JsonValue](teacher_ids(eval_rows)),
            "files": shared.file_manifest(root, list(DATA_FILES)),
            "review_sha256": REVIEW_SHA256, "atlas_tokens": inventory["tokens"]}
