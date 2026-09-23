"""Offline rescoring and tampering rejection."""

import json
from pathlib import Path

import pytest

from sft.scripts.eval import verify_spatial_extension as verify

from .support import PARENT, RUN_NAME, remote_result, volume_transport, write_json


def test_all_real_parent_responses_rescore_offline_with_parent_metadata(retained_run: tuple[Path, dict[str, object]], capsys: pytest.CaptureFixture[str]) -> None:
    run, plan = retained_run
    originals = [PARENT / "launch.json", PARENT / "result.json", *PARENT.glob("post/*")]
    original_hashes = {path: verify.sha256_file(path) for path in originals if path.is_file()}
    assert verify.main(["--run-dir", str(run)]) == 0
    report = verify.read_json(run / "post-verification.json")
    assert report["status"] == "PASS" and report["checkpoint"].endswith("checkpoint-256")
    assert sum(panel["rows"] for panel in report["panels"].values()) == 430
    for label, panel in report["panels"].items():
        assert panel["correct"] == plan["saved_baselines"][label]["summary"]["correct"]
        assert plan["saved_baselines"][label]["summary"]["eval_set_id"] == f"{verify.PARENT_RUN}-{label}"
    assert {path: verify.sha256_file(path) for path in original_hashes} == original_hashes
    assert "PASS checkpoint-256" in capsys.readouterr().out


@pytest.mark.parametrize("with_state", [False, True])
def test_completed_download_is_serial_paced_and_verifies_every_panel(retained_run: tuple[Path, dict[str, object]], derived_parent_256: tuple[dict[str, bytes], bytes], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_state: bool) -> None:
    source_run, plan = retained_run
    run = tmp_path / "download" / RUN_NAME
    write_json(run / "launch.json", plan)
    result = verify.read_json(source_run / "result.json")
    if with_state:
        result["stages"]["training"]["result"].pop("teacher_forced_history")
    remote = remote_result(plan).removesuffix("/result.json")
    payloads = {f"{remote}/result.json": verify.json_bytes(result),
                f"{remote}/post/result.json": (source_run / "post/result.json").read_bytes()}
    for label in verify.PANEL_BUDGETS:
        for name in ("summary.json", "records.jsonl"):
            payloads[f"{remote}/post/{label}/{name}"] = (source_run / f"post/{label}-{name}").read_bytes()
    if with_state:
        payloads[f"catan-vision-sft/{RUN_NAME}/{verify.STATE_PATH}"] = derived_parent_256[1]
    events = volume_transport(monkeypatch, payloads)
    assert verify.main(["--run-dir", str(run), "--download"]) == 0
    assert [event[1] for event in events if event[0] == "read"] == list(payloads)
    assert events[1:] == [event for index, path in enumerate(payloads)
                         for event in ([("sleep", 1)] if index else []) + [("read", path)]]
    assert (run / verify.STATE_PATH).exists() is with_state
    assert verify.read_json(run / "post-verification.json")["status"] == "PASS"


@pytest.mark.parametrize("problem", ["records_hash", "score", "target", "metadata", "new_run_id", "order", "missing", "duplicate", "response", "candidate"])
def test_saved_record_tampering_is_rejected_even_with_updated_receipt_hash(retained_run: tuple[Path, dict[str, object]], problem: str) -> None:
    run, plan = retained_run
    post = verify.read_json(run / "post/result.json")
    path = run / "post/spatial-records.jsonl"
    records = [row for _, row in verify.evaluator.iter_jsonl(path)]
    if problem == "score":
        records[0]["score"]["response_normalized"] = "tampered"
    elif problem == "target":
        records[0]["expected"] = "tampered"
    elif problem in ("metadata", "records_hash"):
        records[0]["metadata"]["relationship"] = "tampered"
    elif problem == "new_run_id":
        records[0]["metadata"]["eval_set_id"] = f"{RUN_NAME}-spatial"
    elif problem == "order":
        records.reverse()
    elif problem == "missing":
        records.pop()
    elif problem == "duplicate":
        records[1] = records[0]
    elif problem == "response":
        records[0]["response"] = None
    else:
        records[0]["candidate_score"] = {"correct": True}
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    if problem != "records_hash":
        post["panels"]["spatial"]["records_sha256"] = verify.sha256_file(path)
    with pytest.raises(ValueError):
        verify.verify_panel(run, plan, post, "spatial")
    assert not (run / "post-verification.json").exists()


@pytest.mark.parametrize("problem", ["aggregate", "extra_aggregate", "budget", "bits", "reasoning", "autocast", "fp32", "visual_hash", "visual_path", "atlas", "checkpoint", "source", "eval_set_id", "conditions"])
def test_summary_settings_and_checkpoint_tampering_are_independently_rejected(retained_run: tuple[Path, dict[str, object]], problem: str) -> None:
    run, plan = retained_run
    post = verify.read_json(run / "post/result.json")
    panel = post["panels"]["fullboard"]
    summary = panel["summary"]
    if problem == "aggregate":
        summary["full_board"]["occupied_layout_macro_accuracy"] = -1
    elif problem == "extra_aggregate":
        summary["candidate_rows"] = 64
    elif problem == "budget":
        summary["long_max_new_tokens"] += 1
    elif problem == "bits":
        summary["bits"] = 4
    elif problem == "reasoning":
        summary["reasoning_enabled"] = True
    elif problem == "autocast":
        summary["precision"]["generation_autocast"]["dtype"] = "torch.float32"
    elif problem == "fp32":
        summary["adapter_evidence"]["visual_precision"]["loaded_dtypes"] = {"torch.float32": 332}
    elif problem in ("visual_hash", "visual_path"):
        summary["adapter_evidence"]["visual_state"]["sha256" if problem == "visual_hash" else "path"] = "wrong"
    elif problem == "atlas":
        summary["adapter_evidence"]["semantic_tokens"]["token_ids"][0] += 1
    elif problem == "checkpoint":
        summary["adapter_dir"] = verify.PARENT_CHECKPOINT
    elif problem == "source":
        summary["eval_source_sha256"] = "wrong"
    elif problem == "eval_set_id":
        summary["eval_set_id"] = f"{RUN_NAME}-fullboard"
    else:
        panel["conditions"]["do_sample"] = True
    write_json(run / "post/fullboard-summary.json", summary)
    with pytest.raises(ValueError):
        verify.verify_panel(run, plan, post, "fullboard")


@pytest.mark.parametrize("problem", ["row_order", "pixels", "bytes", "remote_identity"])
def test_original_ordered_input_and_pixels_are_pinned(retained_run: tuple[Path, dict[str, object]], tmp_path: Path, problem: str) -> None:
    run, plan = retained_run
    post = verify.read_json(run / "post/result.json")
    local = plan["dataset_inputs"]["new_panels"]["node_tiles"]
    original = Path(local["eval_jsonl"])
    copied = tmp_path / "source.jsonl"
    data = original.read_bytes()
    if problem == "row_order":
        data = b"\n".join(reversed(data.splitlines())) + b"\n"
    elif problem == "bytes":
        data += b"\n"
    copied.write_bytes(data)
    local["eval_jsonl"] = str(copied)
    if problem == "pixels":
        images = tmp_path / "images"
        images.mkdir()
        for _, row in verify.evaluator.iter_jsonl(copied):
            reference = verify.evaluator.image_reference(row)
            destination = images / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((Path(local["image_root"]) / reference).read_bytes())
        local["image_root"] = str(images)
        assert verify.dataset_identity(str(copied), str(images)) == plan["panels"]["node_tiles"]["local_identity"]
        destination.write_bytes(destination.read_bytes() + b"changed image bytes")
    elif problem == "remote_identity":
        post["panels"]["node_tiles"]["identity"]["sha256"] = "different-upload"
    with pytest.raises(ValueError):
        verify.verify_panel(run, plan, post, "node_tiles")
