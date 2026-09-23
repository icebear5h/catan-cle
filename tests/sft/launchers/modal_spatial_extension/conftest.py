"""Shared fixtures for modal spatial extension plan, preflight, training, and coordination."""

import json
from pathlib import Path

import pytest
from modal_spatial_continuation.fixtures import no_remote as offline_boundaries
from modal_spatial_continuation.fixtures import plan as base_plan

from sft.board.spatial_tasks import node_tile_tokens, shortest_node_path
from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.launchers.spatial import modal_spatial_extension as extension

from .support import write_checkpoint, write_json, write_rows


@pytest.fixture(autouse=True)
def no_models_or_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("offline test attempted model loading or real training")

    monkeypatch.setattr(extension, "run_training", forbidden)
    monkeypatch.setattr(extension.evaluator, "load_model", forbidden)
    monkeypatch.setattr(extension.AutoTokenizer, "from_pretrained", forbidden)


@pytest.fixture
def receipts(
    base_plan: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    old = launcher
    monkeypatch.setattr(extension, "RUN_ROOT", old.RUN_ROOT)
    monkeypatch.setattr(extension, "LOCAL_RUN_ROOT", old.LOCAL_RUN_ROOT)
    checkpoint = old.RUN_ROOT / extension.PARENT_RUN / "checkpoints/checkpoint-128"
    parent_result = old.LOCAL_RUN_ROOT / extension.PARENT_RUN / "result.json"
    monkeypatch.setattr(extension, "PARENT_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr(extension, "PARENT_RESULT", parent_result)
    monkeypatch.setattr(extension, "source_hashes", lambda: {"scorer.py": "version-one"})
    parent_config = {**base_plan["parent_config"], "output_dir": str(checkpoint.parents[1])}
    write_checkpoint(checkpoint, parent_config, 128)
    audit = extension.checkpoint_audit(checkpoint, parent_config, parent=False)
    # The legacy transport fixtures deliberately omit solver targets. Supply
    # valid gold here because these tests run the actual task-aware scorer.
    tiles = node_tile_tokens("<N00>")
    examples = {
        "node_tiles": ({"node": "<N00>"}, " ".join(tiles)),
        "paths": ({"start": "<N00>", "end": "<N03>"}, " ".join(shortest_node_path("<N00>", "<N03>"))),
        "local": ({"node": "<N00>"}, json.dumps({t: {"resource": "wood", "number": 6} for t in tiles})),
        "production": ({"color": "RED", "roll": 8}, json.dumps(dict.fromkeys(("wood", "brick", "sheep", "wheat", "ore"), 0))),
    }
    for label, (target, answer) in examples.items():
        panel = base_plan["panels"][label]
        rows = [r for _, r in extension.iter_jsonl(Path(panel["eval_jsonl"]))]
        for row in rows:
            row["metadata"]["target"] = target
            row["messages"][-1]["content"] = answer
        write_rows(Path(panel["eval_jsonl"]), rows)
        panel["identity"] = extension.dataset_identity(panel["eval_jsonl"], panel["image_root"])
    old_panels = {label: {**base_plan["panels"][label], "sha256": base_plan["panels"][label]["identity"]["sha256"]}
                  for label in ("spatial", "fullboard")}
    monkeypatch.setattr(extension, "OLD_PANELS", old_panels)
    monkeypatch.setattr(old, "OLD_PANELS", old_panels)
    metadata = tmp_path / "metadata.json"
    write_json(metadata, {"schema": "catan_spatial_continuation/v1"})
    inputs = {"schema": "catan_spatial_continuation_inputs/v1", "metadata": str(metadata),
              **{k: base_plan["config"][k] for k in ("train_jsonl", "image_root", "token_inventory")},
              "new_panels": {task: {key: base_plan["panels"][label][key] for key in
                                    ("eval_jsonl", "image_root", "batch_size", "max_new_tokens")}
                             for label, task in extension.NEW_PANEL_TASKS.items()}}
    manifest = tmp_path / "dataset_inputs.json"
    write_json(manifest, inputs)
    panels = {}
    for label, panel in base_plan["panels"].items():
        rows = [r for _, r in extension.iter_jsonl(Path(panel["eval_jsonl"]))]
        records = []
        for row in rows:
            target = extension.evaluator.expected_text(row)
            meta = extension.evaluator.evaluation_metadata(row, image_variant="original")
            score = extension.evaluator.score_response(target, target, metadata=meta)
            records.append({"id": row["row_id"], "expected": target, "response": target,
                            "metadata": meta, "score": score, "candidate_score": None})
        summary = {**extension.evaluator.summarize(records), "adapter_dir": str(checkpoint),
                   "adapter_evidence": {"adapter_dir": str(checkpoint), "adapter_loaded": True,
                       "visual_state": {"sha256": audit["files_sha256"]["visual_model.safetensors"], "loaded": True},
                       "semantic_tokens": audit["semantic_tokens"]},
                   "model_id": extension.SNAPSHOT, "eval_jsonl": panel["eval_jsonl"],
                   "image_root": panel["image_root"], "image_variant": "original", "reasoning_enabled": False,
                   "candidate_scoring": False, "bits": 16, "batch_size": panel["batch_size"],
                   "max_new_tokens": panel["max_new_tokens"], "long_max_new_tokens": panel["max_new_tokens"],
                   "eval_source_sha256": panel["identity"]["sha256"], "precision": {"preserve_visual_fp32": True}}
        output = old.RUN_ROOT / "pipelines" / extension.PARENT_RUN / "post" / label
        write_json(output / "summary.json", summary)
        write_rows(output / "records.jsonl", records)
        write_json(parent_result.parent / f"post/{label}-summary.json", summary)
        write_rows(parent_result.parent / f"post/{label}-records.jsonl", records)
        panels[label] = {"summary": summary, "identity": panel["identity"],
                         "records_path": str(output / "records.jsonl"),
                         "records_sha256": extension.sha256_file(output / "records.jsonl"),
                         "scorer_sha256": extension.digest(extension.source_hashes()),
                         "conditions": extension.evaluation_conditions(label)}
    post = {"status": "completed", "checkpoint": str(checkpoint), "checkpoint_audit": audit,
            "panels": panels, "source_sha256": extension.source_hashes()}
    preflight = {"status": "completed", "train_identity": base_plan["train_identity"],
                 "teacher_eval_identity": base_plan["panels"]["fullboard"]["identity"]}
    parent = {"status": "completed", "phase": "completed", "config": parent_config,
              "stages": {"post": {"status": "completed", "result": post},
                         "training": {"status": "completed"},
                         "preflight": {"status": "completed", "result": preflight}}}
    write_json(parent_result, parent)
    write_json(parent_result.parent / "post/result.json", post)
    write_json(old.RUN_ROOT / "pipelines" / extension.PARENT_RUN / "result.json", parent)
    write_json(old.RUN_ROOT / "pipelines" / extension.PARENT_RUN / "post/result.json", post)
    write_json(parent_result.with_name("launch.json"), {
        "config": parent_config, "config_sha256": extension.digest(parent_config),
        "dataset_inputs": inputs, "train_identity": base_plan["train_identity"], "mixture": base_plan["mixture"],
        "panels": base_plan["panels"], "input_files_sha256": {str(path): extension.sha256_file(path)
            for path in (manifest, metadata, Path(inputs["token_inventory"]))}})
    return manifest


@pytest.fixture
def plan(receipts: Path) -> dict[str, object]:
    return extension.build_plan(receipts, "test-run")

__all__ = ["base_plan", "no_models_or_training", "offline_boundaries", "plan", "receipts"]
