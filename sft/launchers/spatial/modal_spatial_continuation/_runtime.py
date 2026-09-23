"""Runtime verification and the CPU preflight function."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from transformers import AddedToken

from sft.json_types import JsonDict, as_dict, opt_str
from sft.launchers._hf import load_tokenizer
from sft.launchers._json import at, at_dict, at_int, at_list, at_str
from sft.launchers.full_board.modal_full_board_pilot import (
    SNAPSHOT,
    app,
)
from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.paths import PROJECT_ROOT
from sft.scripts.eval import eval_qwen_vl_adapter as evaluator
from sft.scripts.train.train_trl_catan_vision import (
    iter_jsonl,
    load_token_inventory,
    sha256_file,
)

from ._audits import completion_audit
from ._base import CPU_OPTIONS, PANEL_BUDGETS, PREFLIGHT_TIMEOUT
from ._plan import (
    check_identity,
    dataset_identity,
    digest,
    read_json,
    validate_config,
    validate_mixture,
)


def reload_volumes() -> None:
    for volume in (launcher.hf_cache, launcher.sft_data, launcher.sft_runs):
        volume.reload()


def verify_runtime(plan: JsonDict) -> None:
    validate_config(at_dict(plan, "config"), at_str(plan, "run_name"))
    if digest(plan["config"]) != plan["config_sha256"]:
        raise ValueError("configuration hash changed since launch planning")
    if set(at_dict(plan, "panels")) != set(PANEL_BUDGETS):
        raise ValueError("exactly the six approved evaluation panels are required")
    if set(at_dict(plan, "saved_baselines")) != {"spatial", "fullboard"}:
        raise ValueError("both saved parent baselines must be checked before training")
    for label, panel in at_dict(plan, "panels").items():
        count, batch, budget = PANEL_BUDGETS[label]
        if (at(panel, "identity", "rows"), at(panel, "batch_size"), at(panel, "max_new_tokens")) != (count, batch, budget):
            raise ValueError("unapproved evaluation panel size or budget")
    if launcher.source_hashes() != plan["source_sha256"]:
        raise ValueError("working-tree source hashes changed since launch planning")
    if "metadata" not in inspect.signature(evaluator.score_response).parameters:
        raise ValueError("task-aware score_response(..., metadata=...) is required before launch")


@app.function(**CPU_OPTIONS, cpu=(4.0, 4.0), memory=(8192, 8192), timeout=PREFLIGHT_TIMEOUT)
def continuation_preflight(plan: JsonDict) -> JsonDict:
    reload_volumes()
    verify_runtime(plan)
    config = at_dict(plan, "config")
    if Path(at_str(config, "output_dir")).exists():
        raise FileExistsError(config["output_dir"])
    if not Path(SNAPSHOT).is_dir():
        raise FileNotFoundError("pinned base snapshot is not cached")
    checkpoint = launcher.checkpoint_audit(Path(launcher.PARENT_CHECKPOINT), at_dict(plan, "parent_config"), parent=True)
    inventory = load_token_inventory(at_str(config, "token_inventory"))
    atlas_tokens = [str(t) for t in at_list(inventory, "atlas_tokens")]
    tokenizer = load_tokenizer(launcher.PARENT_CHECKPOINT)
    base = load_tokenizer(SNAPSHOT)
    base.add_tokens([AddedToken(t, normalized=False, special=False) for t in atlas_tokens])
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in atlas_tokens]
    expected = at_dict(checkpoint, "semantic_tokens")
    if (expected["tokens"] != inventory["atlas_tokens"] or ids != [[i] for i in at_list(expected, "token_ids")]
            or [base.encode(t, add_special_tokens=False) for t in atlas_tokens] != ids):
        raise ValueError("parent/base tokenizer atlas IDs differ")
    adapter = read_json(Path(launcher.PARENT_CHECKPOINT) / "adapter_config.json")
    if len(as_dict(adapter.get("trainable_token_indices", {}))) != 2 or any(v != expected["token_ids"] for v in at_dict(adapter, "trainable_token_indices").values()):
        raise ValueError("adapter must retain both input and output atlas row IDs")
    train_rows = [r for _, r in iter_jsonl(Path(at_str(config, "train_jsonl")))]
    mixture = validate_mixture(train_rows)
    if mixture != plan["mixture"]:
        raise ValueError("training step ownership changed")
    train_identity = dataset_identity(at_str(config, "train_jsonl"), opt_str(config["image_root"]))
    check_identity(train_identity, plan["train_identity"])
    panel_tokens: JsonDict = {}
    panel_identities: JsonDict = {}
    baselines: JsonDict = {}
    report: JsonDict = {"status": "completed", "checkpoint": checkpoint, "mixture": mixture,
                        "train_identity": train_identity, "train_tokens": completion_audit(train_rows, tokenizer),
                        "panel_tokens": panel_tokens, "panels": panel_identities, "baselines": baselines}
    for label, panel in at_dict(plan, "panels").items():
        eval_jsonl = at_str(panel, "eval_jsonl")
        identity = dataset_identity(eval_jsonl, opt_str(at(panel, "image_root")))
        check_identity(identity, at(panel, "identity"))
        rows = [r for _, r in iter_jsonl(Path(eval_jsonl))]
        panel_tokens[label] = completion_audit(rows, tokenizer, budget=at_int(panel, "max_new_tokens"))
        panel_identities[label] = identity
    report["teacher_eval_identity"] = dataset_identity(at_str(config, "eval_jsonl"), opt_str(config["eval_image_root"]))
    check_identity(report["teacher_eval_identity"], at(plan, "panels", "fullboard", "identity"))
    # Runtime source lives in launchers; compare its bytes to the unchanged historical hash.
    if sha256_file(PROJECT_ROOT / "sft/launchers/full_board/modal_full_board_pilot.py") != plan["legacy_pilot_sha256"]:
        raise ValueError("legacy fullboard FP32 restore/autocast recipe changed; review baseline before launch")
    for label, saved in at_dict(plan, "saved_baselines").items():
        baseline = launcher.matched_baseline(label, as_dict(saved), at_dict(plan, "panels", label), checkpoint)
        baselines[label] = baseline
        baseline["scorer_sha256"] = digest(plan["source_sha256"])
        baseline["conditions"] = evaluation_conditions(label)
    launcher.sft_runs.commit()
    return report


def panel_args(plan: JsonDict, checkpoint: str, label: str) -> argparse.Namespace:
    panel = at_dict(plan, "panels", label)
    _, batch, budget = PANEL_BUDGETS[label]
    if (panel["batch_size"], panel["max_new_tokens"]) != (batch, budget):
        raise ValueError("unapproved evaluation budget")
    return argparse.Namespace(image_root=panel["image_root"], limit=None, batch_size=batch,
                              long_batch_size=batch, max_new_tokens=budget, long_max_new_tokens=budget,
                              candidate_scoring=False, preserve_visual_fp32=True, occlusion_margin=0.03,
                              model_id=SNAPSHOT, adapter_dir=checkpoint, bits=16,
                              token_inventory=at(plan, "config", "token_inventory"),
                              disable_flash_attn2=True, enable_thinking=False, do_sample=False)


def evaluation_conditions(label: str) -> JsonDict:
    _, batch, budget = PANEL_BUDGETS[label]
    conditions: JsonDict = dict(image_variant="original", enable_thinking=False, do_sample=False,
                bits=16, candidate_scoring=False, preserve_visual_fp32=True,
                batch_size=batch, long_batch_size=batch,
                max_new_tokens=budget, long_max_new_tokens=budget)
    return conditions
