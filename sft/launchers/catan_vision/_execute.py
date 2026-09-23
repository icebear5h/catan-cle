"""Upload the pinned bundle and cross the paid H200 boundary."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from sft.json_types import JsonDict, JsonLikeDict
from sft.launchers import modal_catan_vision_sft as launcher
from sft.launchers.catan_vision._bundle import upload_training_bundle
from sft.launchers.catan_vision._config import BUDGET_STARTUP_SECONDS, BUDGET_TIMEOUT_SECONDS, app
from sft.launchers.catan_vision._receipts import _write_json_atomic
from sft.scripts.train.train_trl_catan_vision import TrainConfig


def execute_training_run(
    *,
    config: TrainConfig,
    launch: JsonLikeDict,
    plan: JsonLikeDict,
    budget: JsonLikeDict | None,
    train_jsonl: str,
    image_root: str,
    token_inventory: str,
    eval_jsonl: str | None,
    eval_image_root: str | None,
    remote_dir: str,
    remote_train: str,
    remote_eval: str | None,
    remote_images: str,
    remote_tokens: str,
    factors_local: Path | None,
    output_dir: str,
    local_contract: JsonDict | None,
    local_eval_contract: JsonDict | None,
    require_curriculum: bool,
    resume_latest: bool,
    spawn_training: bool,
    receipt: Path | None,
) -> None:
    """Upload the pinned bundle, then cross the paid H200 boundary."""

    (
        uploaded_train,
        uploaded_eval,
        uploaded_images,
        uploaded_tokens,
        uploaded_contract,
        uploaded_eval_contract,
    ) = upload_training_bundle(
        Path(train_jsonl),
        Path(image_root),
        Path(token_inventory),
        remote_dir=remote_dir,
        require_curriculum=require_curriculum,
        eval_jsonl=Path(eval_jsonl) if eval_jsonl is not None else None,
        eval_image_root=Path(eval_image_root) if eval_image_root is not None else None,
        extra_files={"visual_delta_factors.safetensors": factors_local} if factors_local is not None else None,
    )
    if (uploaded_train, uploaded_eval, uploaded_images, uploaded_tokens) != (
        remote_train,
        remote_eval,
        remote_images,
        remote_tokens,
    ):
        raise RuntimeError("uploaded paths disagree with the launch plan")
    if uploaded_contract != local_contract:
        raise RuntimeError("dataset changed between planning and upload")
    if uploaded_eval_contract != local_eval_contract:
        raise RuntimeError("evaluation dataset changed between planning and upload")
    if budget is not None:
        deadline = time.time() + BUDGET_TIMEOUT_SECONDS + BUDGET_STARTUP_SECONDS
        call = launcher.train_h200_budgeted.spawn(asdict(config), launch, deadline)
        try:
            watcher = launcher.guard_training_budget.spawn(call.object_id, deadline, output_dir)
        except BaseException:
            call.cancel(terminate_containers=True)
            raise
        spawned: JsonLikeDict = {
            "status": "spawned", "function_call_id": call.object_id,
            "budget_guard_call_id": watcher.object_id, "deadline_unix": deadline,
            "identity": plan["identity"], "output_dir": output_dir,
            "modal_app_id": app.app_id,
        }
        if receipt is not None:
            _write_json_atomic(receipt, {**plan, **spawned})
        print(json.dumps(spawned, indent=2, sort_keys=True), flush=True)
        if not spawn_training:
            print(json.dumps(watcher.get(), indent=2, sort_keys=True))
        return
    if spawn_training:
        call = launcher.train_h200.spawn(asdict(config), launch, resume_latest)
        print(
            json.dumps(
                {
                    "status": "spawned",
                    "function_call_id": call.object_id,
                    "identity": plan["identity"],
                    "output_dir": output_dir,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = launcher.train_h200.remote(asdict(config), launch, resume_latest)
    print(json.dumps(result, indent=2, sort_keys=True))
