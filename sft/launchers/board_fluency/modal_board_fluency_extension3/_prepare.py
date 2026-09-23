from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from sft.json_types import JsonDict, as_dict, as_list, as_str
from sft.launchers.board_fluency import modal_board_fluency_extension as ext1
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_sft import (
    BASE,
    CheckpointCallback,
    check_deadline,
    check_manifest,
    rows_at,
    shared,
    trainer,
    visual_file_digest,
)
from sft.scripts.train.train_trl_catan_vision import TrainConfig, load_token_inventory, sha256_file

from ._baselines import inspect_original
from ._config import DATA_DIR, FROZEN_DIGEST, PARENT, PARENT_ROOT, R06_ROOT, TEACHER_SHA256
from ._planning import read_parent
from ._types import dig, json_dict


def verify_initialization(config: TrainConfig, prepared: JsonDict,
                          report: JsonDict) -> None:
    if Path(as_str(report["path"])).resolve(strict=True) != Path(PARENT).resolve(strict=True):
        raise ValueError("trainer did not load the trained r02 checkpoint-256")
    for key, name in {"adapter_sha256": "adapter_model.safetensors", "visual_sha256": trainer.VISUAL_STATE_FILE,
                      "parent_training_config_sha256": trainer.RUN_CONFIG_FILE}.items():
        if report[key] != as_dict(as_dict(prepared["parent_files"])[name])["sha256"]:
            raise ValueError(f"actual initial-bundle content differs: {key}")
    expected = {"input_mode": "text", "source_input_mode": "text", "lora_rank": 16, "lora_alpha": 32,
                "optimizer_state_restored": False, "scheduler_state_restored": False, "rng_state_restored": False}
    if any(report[key] != value for key, value in expected.items()) or config.resume_from_checkpoint is not None:
        raise ValueError("extension requires the trained text bundle and fresh optimizer/schedule/RNG")


def check_schedule_log(state: TrainerState, logs: dict[str, float] | None) -> None:
    if not logs or "learning_rate" not in logs:
        return
    value = logs["learning_rate"]
    if state.global_step == 1:
        if value != 0.0:
            raise ValueError("warmup schedule must start at exactly 0.0 on update 1")
    elif not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError("warmup schedule allows zero at update 1 only")


def check_schedule_history(trainer_state: JsonDict) -> None:
    history = as_list(trainer_state.get("log_history", []))
    seen_first = False
    for item in history:
        entry = as_dict(item)
        if "learning_rate" not in entry or entry.get("step") is None:
            continue
        step, value = entry["step"], entry["learning_rate"]
        if step == 1:
            if value != 0.0:
                raise ValueError("warmup schedule must start at exactly 0.0 on update 1")
            seen_first = True
        elif not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
            raise ValueError("warmup schedule allows zero at update 1 only")
    if not seen_first:
        raise ValueError("warmup schedule missing update-1 zero")


class Extension3Callback(CheckpointCallback):
    def __init__(
        self,
        config: TrainConfig,
        deadline: float,
        prepared: JsonDict,
    ) -> None:
        super().__init__(config, deadline)
        self.prepared = prepared

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        **kwargs: object,
    ) -> None:
        super().on_train_begin(args, state, control, model, optimizer, **kwargs)
        # `args` is the remote TRL SFTConfig; the locally installed TRL does not declare
        # these sampling fields, so read them dynamically (still AttributeError if absent).
        if (args.world_size != 1 or getattr(args, "train_sampling_strategy") != "sequential"
                or getattr(args, "shuffle_dataset")
                or args.max_steps != 512 or args.per_device_train_batch_size != 4
                or args.gradient_accumulation_steps != 2 or args.save_total_limit != 16):
            raise ValueError("actual trainer sampling/update/checkpoint retention differs")
        output = Path(self.config.output_dir)
        verify_initialization(self.config, self.prepared,
                              shared.read_json(output / trainer.INITIAL_BUNDLE_FILE))
        dataset = shared.read_json(output / trainer.DATASET_REPORT_FILE)
        for split, path, rows, expected in (
            ("train", self.config.train_jsonl, 4096,
             as_dict(self.prepared["suffix"])["sha256"]),
            ("eval", self.config.eval_jsonl, 120, self.prepared["teacher_sha256"]),
        ):
            actual = as_dict(as_dict(dataset)[split])
            if (Path(as_str(actual["source"])).resolve(strict=True)
                    != Path(as_str(path)).resolve(strict=True)
                    or actual["source_sha256"] != expected or actual["rows"] != rows
                    or actual["input_mode"] != "text" or actual["truncation"]):
                raise ValueError("actual trainer dataset differs from admitted sequential data")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs: object,
    ) -> None:
        super().on_log(args, state, control, logs, **kwargs)
        check_schedule_log(state, logs)


def prepare_work(plan: JsonDict, deadline: float) -> JsonDict:
    parent = read_parent(Path(PARENT_ROOT))
    prior = as_dict(parent["prepare"]["result"])
    plan_config = as_dict(plan["config"])
    inventory = load_token_inventory(as_str(plan_config["token_inventory"]))
    bundle, _ = shared.volume_bundle(PARENT)
    parent_files = ext1.full_checkpoint_manifest(bundle)
    tokenizer, checkpoint = shared.adapter_preflight(bundle, inventory, shared.MODEL_ID, shared.MODEL_REVISION)
    saved_config = shared.read_json(bundle / trainer.RUN_CONFIG_FILE)
    semantic = as_dict(as_dict(as_dict(as_dict(
        parent["train"]["result"])["training"])["reload_validation"])["semantic_tokens"])
    if (saved_config != as_dict(parent["launch"])["config"]
            or saved_config["input_mode"] != "text"
            or checkpoint["saved_model_id"] != BASE or checkpoint["lora_rank"] != 16
            or checkpoint["lora_alpha"] != 32
            or as_dict(checkpoint["adapter_config"])["lora_dropout"] != 0.05
            or as_dict(checkpoint["semantic_tokens"])["token_ids"] != semantic["token_ids"]
            or as_dict(checkpoint["semantic_tokens"])["tokens"] != semantic["tokens"]):
        raise ValueError("trained parent rank/alpha/dropout/input/base/atlas identity differs")
    base_files = original.cached_base_files()
    if set(base_files) != set(as_dict(prior["base_files"])):
        raise ValueError("cached native base file set changed")
    check_manifest(Path(BASE), as_dict(prior["base_files"]))
    base_audit = shared.base_preflight(Path(BASE), base_files, bundle, inventory, checkpoint)
    if base_audit != prior["base_audit"]:
        raise ValueError("cached base metadata/shards/header/architecture audit changed")
    if shared.read_json(bundle / "trainer_state.json")["global_step"] != 256:
        raise ValueError("trained parent global step differs")
    frozen = visual_file_digest(bundle / trainer.VISUAL_STATE_FILE)
    if (frozen != as_dict(parent["train"]["result"])["frozen_visual_file_tensor_sha256"]
            or frozen != FROZEN_DIGEST):
        raise ValueError("parent frozen visual tensors differ from r02 training")
    suffix, identity, baselines = inspect_original(parent, Path(PARENT_ROOT), Path(DATA_DIR),
                                                  Path(DATA_DIR) / "review.jsonl",
                                                  Path(R06_ROOT) / "prepare/teacher120.jsonl")
    if identity != plan["suffix"] or baselines != plan["saved_baselines"]:
        raise ValueError("CPU suffix/baseline admission differs")
    lengths: JsonDict = {"train_suffix": json_dict(ext1.text_boundaries(
        tokenizer, [as_dict(json.loads(line)) for line in suffix.splitlines()],
        generation=False, deadline=deadline))}
    for panel, path in {"teacher120": as_str(plan_config["eval_jsonl"]),
                        "review": DATA_DIR + "/review.jsonl", "validation_eval": DATA_DIR + "/validation_eval.jsonl"}.items():
        lengths[panel] = json_dict(ext1.text_boundaries(tokenizer, rows_at(Path(path)),
                                                        generation=panel != "teacher120", deadline=deadline))
    del tokenizer
    check_deadline(deadline)
    destination = Path(as_str(plan_config["train_jsonl"]))
    with destination.open("xb") as handle:
        handle.write(suffix)
        handle.flush()
        os.fsync(handle.fileno())
    suffix_files = shared.file_manifest(destination.parent, [destination.name])
    if suffix_files[destination.name] != {key: identity[key] for key in ("bytes", "sha256")}:
        raise ValueError("exclusive raw-line suffix write differs")
    check_manifest(bundle, parent_files)
    return {"checkpoint": checkpoint, "parent_files": parent_files,
            "parent_manifest_provenance": "full current r02 checkpoint manifest established by this CPU prepare; no historical r06 adapter hash claimed",
            "frozen_visual_file_tensor_sha256": frozen, "base_files": prior["base_files"],
            "base_audit": base_audit, "base_manifest_provenance": "r02 metadata hashes and shard sizes, rechecked tensor headers/index",
            "suffix_files": suffix_files, "suffix": identity, "baselines": baselines,
            "teacher_sha256": TEACHER_SHA256,
            "token_lengths": lengths,
            "prior_token_lengths": prior["token_lengths"],
            "token_boundary_provenance": "actual native trained-parent tokenizer, exact original raw rows"}


def prepared_for(plan: JsonDict) -> JsonDict:
    prepared = ext1.completed_stage(plan, "prepare")
    check_manifest(Path(PARENT), as_dict(prepared["parent_files"]))
    check_manifest(Path(BASE), as_dict(prepared["base_files"]))
    check_manifest(Path(as_str(dig(plan, "config", "train_jsonl"))).parent, as_dict(prepared["suffix_files"]))
    if (prepared["suffix"] != plan["suffix"] or prepared["baselines"] != plan["saved_baselines"]
            or sha256_file(Path(as_str(dig(plan, "config", "eval_jsonl")))) != prepared["teacher_sha256"]):
        raise ValueError("prepared suffix/teacher/baseline identity changed")
    return prepared
