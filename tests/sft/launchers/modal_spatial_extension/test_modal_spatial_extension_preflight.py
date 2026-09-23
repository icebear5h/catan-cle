"""CPU preflight rescoring and baseline audits."""

import copy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sft.launchers.spatial import modal_spatial_continuation as launcher
from sft.launchers.spatial import modal_spatial_extension as extension

from .support import preflight, write_checkpoint, write_json, write_rows


def test_cpu_preflight_really_rescores_every_saved_record_and_entire_summary(plan: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    score = extension.evaluator.score_response

    def track(expected: str, response: str, *, metadata: dict[str, object]) -> object:
        calls.append(metadata)
        return score(expected, response, metadata=metadata)

    monkeypatch.setattr(extension.evaluator, "score_response", track)
    result = preflight(plan, monkeypatch)
    assert len(calls) == sum(budget[0] for budget in extension.PANEL_BUDGETS.values()) == 430
    assert result["checkpoint_audit"] == plan["parent_audit"]
    assert set(result["baselines"]) == set(extension.PANEL_BUDGETS)
    for label, baseline in result["baselines"].items():
        assert baseline["reused"] and baseline["original_only"]
        assert baseline["scorer_sha256"] == extension.digest(plan["source_sha256"])
        for key, value in extension.evaluator.summarize([
            r for _, r in extension.iter_jsonl(Path(baseline["records_path"]))]).items():
            if key != "generated_at":
                assert baseline["summary"][key] == value


@pytest.mark.parametrize("label", list(extension.PANEL_BUDGETS))
@pytest.mark.parametrize("problem", ["records_hash", "metadata", "summary_dimension", "input", "original", "budget", "checkpoint"])
def test_any_baseline_mismatch_blocks_cpu_preflight(
    plan: dict[str, object], monkeypatch: pytest.MonkeyPatch, label: str, problem: str
) -> None:
    saved = plan["saved_baselines"][label]
    output = Path(saved["output_dir"])
    if problem in ("records_hash", "metadata"):
        records = [r for _, r in extension.iter_jsonl(output / "records.jsonl")]
        records[0]["metadata"]["eval_variant"] = "blank"
        write_rows(output / "records.jsonl", records)
        if problem == "metadata":
            saved["records_sha256"] = extension.sha256_file(output / "records.jsonl")
    elif problem == "input":
        path = Path(saved["summary"]["eval_jsonl"])
        path.write_text(path.read_text() + "\n")
    else:
        if problem == "summary_dimension":
            saved["summary"]["by_suite"] = {"fabricated": {"correct": 999}}
        elif problem == "original":
            saved["summary"]["image_variant"] = "blank"
        elif problem == "budget":
            saved["summary"]["long_max_new_tokens"] += 1
        else:
            saved["summary"]["adapter_dir"] = "older-checkpoint"
        write_json(output / "summary.json", saved["summary"])
        saved["summary_sha256"] = extension.sha256_file(output / "summary.json")
    with pytest.raises(ValueError):
        preflight(plan, monkeypatch)


@pytest.mark.parametrize("filename", ["adapter_model.safetensors", "tokenizer.json", "training_config.json", "trainer_state.json"])
def test_parent_full_hash_audit_blocks_each_changed_checkpoint_file(plan: dict[str, object], filename: str) -> None:
    path = Path(extension.PARENT_CHECKPOINT) / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hashes"):
        extension.audit_parent(plan)


def test_explicit_checkpoint_step_and_fp32_scope_do_not_weaken_legacy(plan: dict[str, object], tmp_path: Path) -> None:
    path = tmp_path / "checkpoint-256"
    write_checkpoint(path, plan["config"], 256)
    report = extension.checkpoint_audit(path, plan["config"], parent=False, expected_step=256)
    assert report["visual_dtypes"] == {"F32": 333}
    with pytest.raises(ValueError, match="global_step"):
        extension.checkpoint_audit(path, plan["config"], parent=False)
    save_file({f"visual.{i}": torch.zeros(1, dtype=torch.bfloat16) for i in range(333)}, path / "visual_model.safetensors")
    with pytest.raises(ValueError, match="333 FP32"):
        extension.checkpoint_audit(path, plan["config"], parent=False, expected_step=256)


@pytest.mark.parametrize("label", list(extension.PANEL_BUDGETS))
def test_retained_real_parent_responses_still_match_all_summary_fields(label: str) -> None:
    """Optional local integration: use actual saved generations, never inference."""
    root = launcher.PROJECT_ROOT / "artifacts/runs/sft" / extension.PARENT_RUN
    if not (root / "post/result.json").exists():
        pytest.skip("retained parent artifacts are not part of the repository")
    post = extension.read_json(root / "post/result.json")
    launch = extension.read_json(root / "launch.json")
    saved = post["panels"][label]
    summary = saved["summary"]
    local = (extension.OLD_PANELS[label] if label in extension.OLD_PANELS
             else launch["dataset_inputs"]["new_panels"][extension.NEW_PANEL_TASKS[label]])
    rows = {r.get("id") or r.get("row_id"): r for _, r in extension.iter_jsonl(Path(local["eval_jsonl"]))}
    records_path = root / f"post/{label}-records.jsonl"
    assert extension.sha256_file(records_path) == saved["records_sha256"]
    assert extension.read_json(root / f"post/{label}-summary.json") == summary
    records = [r for _, r in extension.iter_jsonl(records_path)]
    assert len(records) == len(rows) and {r["id"] for r in records} == set(rows)
    for record in records:
        row = copy.deepcopy(rows[record["id"]])
        row.setdefault("metadata", {}).update(eval_source_sha256=summary["eval_source_sha256"],
                                               eval_set_id=summary["eval_set_id"])
        metadata = extension.evaluator.evaluation_metadata(row, image_variant="original")
        assert metadata == record["metadata"]
        assert extension.evaluator.expected_text(row) == record["expected"]
        score = extension.evaluator.score_response(record["expected"], record["response"], metadata=metadata)
        assert score == record["score"]
        record["score"] = score
    rescored = extension.evaluator.summarize(records)
    assert all(value == summary[key] for key, value in rescored.items() if key != "generated_at")
