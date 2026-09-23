from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

import modal
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str
from sft.launchers._train_config import train_config
from sft.launchers.board_fluency import modal_board_fluency_sft as original
from sft.launchers.board_fluency.modal_board_fluency_eval import reload_volumes
from sft.launchers.board_fluency.modal_board_fluency_sft import (
    OFFLINE,
    CheckpointCallback,
    check_deadline,
    check_manifest,
    deadline_alarm,
    eval_panel,
    load_eval,
    progress,
    shared,
    trainer,
    visual_file_digest,
)
from sft.scripts.train.train_trl_catan_vision import TrainConfig, write_json_atomic

from ._baselines import verify_plan
from ._config import CHECKPOINT_STEPS, PARENT, RESOURCES, STAGE_SECONDS
from ._planning import source_hashes
from ._prepare import (
    completed_stage,
    full_checkpoint_manifest,
    prepare_work,
    prepared_for,
    verify_initialization,
)
from ._types import at


class ExtensionCallback(CheckpointCallback):
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
        super().on_train_begin(args, state, control, model=model, optimizer=optimizer, **kwargs)
        # The runtime args are TRL's SFTConfig from the Modal image; the locally
        # installed TRL/Transformers stubs predate these two sampler fields.
        if (args.world_size != 1 or getattr(args, "train_sampling_strategy") != "sequential"
                or getattr(args, "shuffle_dataset")
                or args.max_steps != 128 or args.per_device_train_batch_size != 4
                or args.gradient_accumulation_steps != 2 or args.save_total_limit != 4):
            raise ValueError("actual trainer sampling/update/checkpoint retention differs")
        output = Path(self.config.output_dir)
        verify_initialization(self.config, self.prepared, shared.read_json(output / trainer.INITIAL_BUNDLE_FILE))
        dataset = shared.read_json(output / trainer.DATASET_REPORT_FILE)
        for split, path, rows, expected in (
            ("train", self.config.train_jsonl, 2176, at(self.prepared, "suffix", "sha256")),
            ("eval", self.config.eval_jsonl, 120, self.prepared["teacher_sha256"]),
        ):
            actual = as_dict(dataset[split])
            if (path is None
                    or Path(as_str(actual["source"])).resolve(strict=True) != Path(path).resolve(strict=True)
                    or actual["source_sha256"] != expected or actual["rows"] != rows
                    or actual["input_mode"] != "text" or actual["truncation"]):
                raise ValueError("actual trainer dataset differs from admitted sequential data")


def train_work(plan: JsonDict, deadline: float) -> JsonLikeDict:
    prepared = prepared_for(plan)
    config = train_config(as_dict(plan["config"]))
    if Path(config.output_dir).exists():
        raise FileExistsError("training output exists; retries/resumption are not authorized")
    callback = ExtensionCallback(config, deadline, prepared)
    progress("extension: trained r06 checkpoint, fresh optimizer, rows 1025–2048, 128 updates")
    result = trainer.run_training(config, extra_callbacks=[callback])
    if result["status"] != "completed" or callback.saved_steps != CHECKPOINT_STEPS:
        raise RuntimeError("extension did not commit exactly checkpoints 32/64/96/128")
    verify_initialization(config, prepared, as_dict(result["initial_bundle"]))
    for step in CHECKPOINT_STEPS:
        saved = Path(config.output_dir) / "checkpoints" / f"checkpoint-{step}"
        shared.volume_bundle(str(saved))
        if shared.read_json(saved / "trainer_state.json")["global_step"] != step:
            raise ValueError("retained checkpoint global step differs")
    final = Path(config.output_dir) / "checkpoints/checkpoint-128"
    frozen = visual_file_digest(final / trainer.VISUAL_STATE_FILE)
    if frozen != prepared["frozen_visual_file_tensor_sha256"]:
        raise ValueError("extension changed the parent's canonical frozen visual tensors")
    check_manifest(Path(PARENT), as_dict(prepared["parent_files"]))
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("source manifest changed during training")
    return {"training": result, "final_checkpoint": str(final), "committed_steps": callback.saved_steps,
            "final_checkpoint_files": full_checkpoint_manifest(final),
            "frozen_visual_file_tensor_sha256": frozen, "additional_updates": 128, "cumulative_updates": 256,
            "consumed": at(plan, "suffix", "consumed"), "cumulative_unique_examples": 2048,
            "cumulative_corpus_epoch": 0.64, "consumption_basis": "verified single-process sequential sampler, exact suffix hash, 128 completed batch8 updates"}


def posteval_work(plan: JsonDict, deadline: float) -> JsonLikeDict:
    prepared = prepared_for(plan)
    training = completed_stage(plan, "train")
    checkpoint = as_str(at(plan, "config", "output_dir")) + "/checkpoints/checkpoint-128"
    if training["final_checkpoint"] != checkpoint:
        raise ValueError("post-evaluation checkpoint differs")
    check_manifest(Path(checkpoint), as_dict(training["final_checkpoint_files"]))
    model, tokenizer, evidence = load_eval(plan, checkpoint)
    panels: JsonLikeDict = {}
    for panel in ("review", "validation_eval"):
        check_deadline(deadline)
        panels[panel] = eval_panel(plan, model, tokenizer, evidence, checkpoint, panel,
                                  Path(as_str(plan["root"])) / "posteval" / panel)
    check_manifest(Path(PARENT), as_dict(prepared["parent_files"]))
    return {"checkpoint": checkpoint, "panels": panels, "saved_baselines": prepared["baselines"]}


def worker(plan: JsonDict, stage: str, deadline: float) -> None:
    os.environ.update(OFFLINE)
    torch.set_num_threads(RESOURCES["prepare" if stage == "prepare" else "gpu"]["cpu"])
    directory = Path(as_str(plan["root"])) / stage
    receipt: JsonLikeDict = {"status": "running", "stage": stage, "launch_sha256": shared.digest(plan),
               "started_at": shared.now(), "inner_deadline_unix": deadline}
    try:
        write_json_atomic(directory / "result.json", receipt)
        original.sft_runs.commit()
        with deadline_alarm(deadline):
            if shared.read_json(directory.parent / "launch.json") != plan:
                raise ValueError("extension reservation changed")
            verify_plan(plan)
            receipt["dependencies"] = trainer.assert_runtime_versions()
            function = {"prepare": prepare_work, "train": train_work, "posteval": posteval_work}[stage]
            receipt.update(result=function(plan, deadline), status="completed")
            if source_hashes() != plan["source_sha256"]:
                raise ValueError("source manifest changed during stage")
    except BaseException as exc:
        receipt.update(status="failed", error=shared.error_record(exc))
        raise
    finally:
        receipt["ended_at"] = shared.now()
        write_json_atomic(directory / "result.json", receipt)
        original.sft_runs.commit()


def bounded_stage(plan: JsonDict, stage: str, deadline: float) -> JsonDict:
    hard_deadline = min(deadline - 30, time.time() + STAGE_SECONDS[stage] - 30)
    inner_deadline = hard_deadline - 45
    check_deadline(inner_deadline)
    reload_volumes()
    root = as_str(plan["root"])
    directory = Path(root) / stage
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-B", "-m", "sft.launchers.board_fluency.modal_board_fluency_extension", "--worker", stage,
               "--plan", root + "/launch.json", "--deadline", str(inner_deadline)]
    started_at = shared.now()
    wrapper: JsonLikeDict = {"status": "running", "stage": stage, "launch_sha256": shared.digest(plan),
               "started_at": started_at, "command": command,
               "call_id": modal.current_function_call_id(), "hard_deadline_unix": hard_deadline}
    process, error = None, None
    log_path = Path("/tmp") / f"board-fluency-extension-{plan['reservation_id']}-{stage}.log"
    try:
        write_json_atomic(directory / "wrapper.json", wrapper)
        original.sft_runs.commit()
        with log_path.open("x", buffering=1) as log:
            process = subprocess.Popen(command, env={**os.environ, **OFFLINE}, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            relay_errors: list[JsonDict] = []
            stdout = process.stdout
            if stdout is None:
                raise RuntimeError("worker stdout pipe was not opened")

            def forward_logs() -> None:
                try:
                    for line in stdout:
                        log.write(line)
                        print(line, end="", flush=True)
                except BaseException as exc:
                    relay_errors.append(shared.error_record(exc))

            relay = threading.Thread(target=forward_logs, daemon=True)
            try:
                relay.start()
                code = process.wait(timeout=max(0.001, hard_deadline - time.time()))
                if code:
                    raise RuntimeError(f"{stage} worker exited {code}; see {directory}/worker.log")
            finally:
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                if relay.ident is not None:
                    relay.join(timeout=5)
            if relay.is_alive() or relay_errors:
                raise RuntimeError(f"worker log relay incomplete: {relay_errors}")
        original.sft_runs.reload()
        result = shared.read_json(directory / "result.json")
        if (result["status"] != "completed" or result["stage"] != stage
                or result["launch_sha256"] != shared.digest(plan)):
            raise RuntimeError(f"{stage} worker did not complete this launch")
        return result
    except BaseException as exc:
        error = shared.error_record(exc)
        raise
    finally:
        try:
            original.sft_runs.reload()
            if log_path.is_file():
                shutil.copyfile(log_path, directory / "worker.log")
        except BaseException as exc:
            error = shared.error_record(exc)
            raise
        finally:
            wrapper.update(status="failed" if error else "completed", error=error, ended_at=shared.now())
            if error:
                path = directory / "result.json"
                receipt: JsonDict = shared.read_json(path) if path.is_file() else {
                    "stage": stage, "launch_sha256": shared.digest(plan), "started_at": started_at}
                if receipt.get("status") != "completed":
                    receipt.update(status="failed", error=error, ended_at=shared.now())
                    write_json_atomic(path, receipt)
            write_json_atomic(directory / "wrapper.json", wrapper)
            original.sft_runs.commit()
