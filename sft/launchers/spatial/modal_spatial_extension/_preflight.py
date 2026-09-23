"""The CPU preflight that re-audits the parent before training."""

from __future__ import annotations

from pathlib import Path

from transformers import AddedToken

from sft.json_types import JsonDict, as_dict, as_str, opt_str
from sft.launchers._hf import load_tokenizer
from sft.launchers._json import at_dict, at_int, at_list, at_str
from sft.launchers.spatial import modal_spatial_extension as extension
from sft.launchers.spatial.modal_spatial_continuation import (
    CPU_OPTIONS,
    PREFLIGHT_TIMEOUT,
    SNAPSHOT,
    completion_audit,
    dataset_identity,
    digest,
    evaluation_conditions,
    matched_baseline,
    read_json,
    reload_volumes,
    validate_mixture,
)
from sft.scripts.train.train_trl_catan_vision import (
    iter_jsonl,
    load_token_inventory,
)

from ._base import app
from ._plan import audit_inputs, verify_files, verify_runtime


@app.function(**CPU_OPTIONS, cpu=(4.0, 4.0), memory=(8192, 8192), timeout=PREFLIGHT_TIMEOUT)
def extension_preflight(plan: JsonDict) -> JsonDict:
    reload_volumes()
    verify_runtime(plan)
    config = at_dict(plan, "config")
    if Path(at_str(config, "output_dir")).exists():
        raise FileExistsError(config["output_dir"])
    if not Path(SNAPSHOT).is_dir():
        raise FileNotFoundError("pinned base snapshot is not cached")
    verify_files(plan["remote_receipts_sha256"])
    checkpoint = extension.audit_parent(plan)
    identities = audit_inputs(plan)
    inventory = load_token_inventory(at_str(config, "token_inventory"))
    atlas_tokens = [as_str(t) for t in at_list(inventory, "atlas_tokens")]
    tokenizer = load_tokenizer(extension.PARENT_CHECKPOINT)
    base = load_tokenizer(SNAPSHOT)
    base.add_tokens([AddedToken(t, normalized=False, special=False) for t in atlas_tokens])
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in atlas_tokens]
    expected = at_dict(checkpoint, "semantic_tokens")
    if (expected["tokens"] != inventory["atlas_tokens"] or ids != [[i] for i in at_list(expected, "token_ids")]
            or [base.encode(t, add_special_tokens=False) for t in atlas_tokens] != ids):
        raise ValueError("parent/base tokenizer atlas IDs differ")
    indices = at_dict(read_json(Path(extension.PARENT_CHECKPOINT) / "adapter_config.json"), "trainable_token_indices")
    if len(indices) != 2 or any(value != expected["token_ids"] for value in indices.values()):
        raise ValueError("both input/output atlas row IDs must be retained")
    rows = [row for _, row in iter_jsonl(Path(at_str(config, "train_jsonl")))]
    if validate_mixture(rows) != plan["mixture"]:
        raise ValueError("training step ownership changed")
    panel_tokens: JsonDict = {}
    panel_identities: JsonDict = {}
    baselines: JsonDict = {}
    report: JsonDict = {"status": "completed", "checkpoint": extension.PARENT_CHECKPOINT,
                        "checkpoint_audit": checkpoint, "source_sha256": plan["source_sha256"], **identities,
                        "mixture": plan["mixture"], "train_tokens": completion_audit(rows, tokenizer),
                        "panel_tokens": panel_tokens, "panels": panel_identities, "baselines": baselines}
    for label, panel_value in at_dict(plan, "panels").items():
        panel = as_dict(panel_value)
        identity = dataset_identity(at_str(panel, "eval_jsonl"), opt_str(panel["image_root"]))
        if identity != panel["identity"]:
            raise ValueError(f"uploaded panel changed: {label}")
        rows = [row for _, row in iter_jsonl(Path(at_str(panel, "eval_jsonl")))]
        panel_tokens[label] = completion_audit(rows, tokenizer, budget=at_int(panel, "max_new_tokens"))
        panel_identities[label] = identity
        saved = at_dict(plan, "saved_baselines", label)
        baseline = matched_baseline(label, saved, panel, checkpoint,
                                    expected_checkpoint=extension.PARENT_CHECKPOINT,
                                    expected_correct=at_int(saved, "summary", "correct"), strict_summary=True)
        baselines[label] = {**baseline, "scorer_sha256": digest(plan["source_sha256"]),
                            "conditions": evaluation_conditions(label)}
    extension.sft_runs.commit()
    return report
