from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ContentImage, ContentText

from evals.inspect_archives import (
    DEFAULT_STRICT_VISION_RUNS,
    _policy_available_actions,
    build_policy_archive,
    build_strict_vision_archive,
    is_valid_policy_selection,
    verify_inspect_archive_log,
    write_inspect_archive_log,
)
from scripts.import_catan_inspect_archives import (
    OWNER_FILE,
    prepare_output,
    record_owned_log,
    safe_suite_directory,
)
from scripts.render_catan_inspect_viz import strict_vision_rows


def test_strict_vision_archive_preserves_prompt_targets_and_scores() -> None:
    bundle = build_strict_vision_archive(DEFAULT_STRICT_VISION_RUNS["gemma4_31b"])

    assert bundle.model_id == "google/gemma-4-31b-it"
    assert str(bundle.model) == "archive/google/gemma-4-31b-it"
    assert bundle.source_records == 60
    assert bundle.imported_records == 60
    assert bundle.task.metadata["model_selection_report"].endswith(
        "local-27b-80b-models-for-catan.md"
    )
    assert bundle.expected_metrics == {
        "exact": 28,
        "json_valid": 60,
        "protocol_exact": 24,
        "requests": 60,
    }

    sample = bundle.task.dataset[0]
    assert sample.id == "ascii_q000_direction_to_tile"
    assert sample.target == '{"tile":"T04"}'
    assert isinstance(sample.input[0], ChatMessageSystem)
    assert isinstance(sample.input[1], ChatMessageUser)
    user_content = sample.input[1].content
    assert isinstance(user_content, list)
    assert isinstance(user_content[0], ContentImage)
    assert Path(user_content[0].image).is_file()
    assert isinstance(user_content[1], ContentText)
    assert "Question: Which tile is directly LEFT of T00?" in user_content[1].text
    assert sample.metadata["archive"]["score"]["correct"] is True
    assert sample.metadata["provenance"]["scorer_version"] == "strict_typed_json/v2"


def test_policy_archive_uses_original_provider_run_without_repair_overlays() -> None:
    bundle = build_policy_archive()

    assert bundle.model_id == "qwen/qwen3.8-27b"
    assert str(bundle.model) == "archive/qwen/qwen3.8-27b"
    assert bundle.source_records == 164
    assert bundle.imported_records == 111
    assert bundle.task.metadata["model_selection_report"].endswith(
        "local-27b-80b-models-for-catan.md"
    )
    assert bundle.expected_metrics == {
        "decision_count": 164,
        "response_count": 111,
        "exact_responses": 111,
        "exact_human_matches": 50,
        "nontrivial_responses": 89,
        "nontrivial_human_matches": 28,
        "parse_warnings": 35,
        "valid_selections": 111,
    }

    sample = bundle.task.dataset[0]
    assert sample.id == "242781000:0"
    assert "not a quality oracle" in sample.target
    assert sample.metadata["policy"]["model_action_index"] == 0
    assert sample.metadata["policy"]["human_action_index"] == 7
    assert sample.metadata["policy"]["agreement"] is False
    assert sample.metadata["available_actions"][7]["index"] == 7
    assert "three distinct resources: Sheep, Wood, and Wood" in sample.metadata[
        "visible_rationale"
    ]


def test_provider_free_inspect_log_round_trip(tmp_path: Path) -> None:
    bundle = build_policy_archive()
    log_path = write_inspect_archive_log(bundle, tmp_path, embed_images=False)
    verification = verify_inspect_archive_log(bundle, log_path)

    assert verification == {
        "status": "success",
        "model": "archive/qwen/qwen3.8-27b",
        "samples": 111,
        "scores": {
            "policy_selection_valid": 1.0,
            "policy_parse_clean": pytest.approx(76 / 111),
            "recorded_human_action_match": pytest.approx(50 / 111),
            "nontrivial_recorded_human_action_match": pytest.approx(28 / 89),
        },
        "source_scores_verified": True,
    }

    log = read_eval_log(str(log_path))
    assert log.eval.metadata["archive_variant"] == "original_provider_run"
    assert log.eval.metadata["interpretation"].endswith(
        "is not a policy-quality oracle."
    )
    first = log.samples[0]
    assert first.messages[-1].content.startswith("<goals>")
    assert first.scores["recorded_human_action_match"].value == "I"
    assert first.metadata["archive"]["provider"] is None
    assert first.metadata["archive"]["adapter_source"] == "action_diff"


def test_embedded_image_log_preserves_exact_sample_identity(tmp_path: Path) -> None:
    bundle = build_strict_vision_archive(
        DEFAULT_STRICT_VISION_RUNS["gemma4_31b"],
        limit=1,
    )
    log_path = write_inspect_archive_log(bundle, tmp_path, embed_images=True)
    verification = verify_inspect_archive_log(bundle, log_path)

    assert verification["samples"] == 1
    log = read_eval_log(str(log_path), resolve_attachments=True)
    user_content = log.samples[0].messages[1].content
    assert isinstance(user_content, list)
    assert isinstance(user_content[0], ContentImage)
    assert user_content[0].image.startswith("data:image/png;base64,")


def test_policy_selection_validates_exact_menu_membership() -> None:
    bundle = build_policy_archive(limit=1)
    sample = bundle.task.dataset[0]
    policy = dict(sample.metadata["policy"])
    actions = sample.metadata["available_actions"]

    assert is_valid_policy_selection(policy, actions)
    policy["model_action_index"] = 999
    assert not is_valid_policy_selection(policy, actions)
    policy["model_action_index"] = False
    assert not is_valid_policy_selection(policy, actions)
    policy["model_action_index"] = 0
    policy["model_action"] = "different action"
    assert not is_valid_policy_selection(policy, actions)


def test_policy_archive_rejects_missing_provider_legal_menu() -> None:
    with pytest.raises(ValueError, match="missing its exact legal menu"):
        _policy_available_actions({"decision_id": "game:1"}, {"result": {}})


def test_importer_replace_deletes_only_owned_logs(tmp_path: Path) -> None:
    output_dir = tmp_path / "inspect"
    prepare_output(output_dir, replace=False)
    owned = output_dir / "policy" / "owned.eval"
    owned.parent.mkdir()
    owned.write_text("owned")
    record_owned_log(output_dir, owned)

    unrelated = output_dir / "unrelated.eval"
    unrelated.write_text("unrelated")
    with pytest.raises(SystemExit, match="differ from the ownership manifest"):
        prepare_output(output_dir, replace=True)
    assert owned.exists()
    assert unrelated.exists()

    unrelated.unlink()
    prepare_output(output_dir, replace=True)
    assert not owned.exists()
    owner = json.loads((output_dir / OWNER_FILE).read_text())
    assert owner["generated_logs"] == []


def test_importer_refuses_index_only_ownership_claim(tmp_path: Path) -> None:
    output_dir = tmp_path / "index-only"
    output_dir.mkdir()
    fake_log = output_dir / "fake.eval"
    fake_log.write_text("not generated by importer")
    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "schema": "catan-inspect-archive/v1",
                "logs": [{"log_path": "fake.eval"}],
            }
        )
    )

    with pytest.raises(SystemExit, match="unowned Inspect output"):
        prepare_output(output_dir, replace=True)
    assert fake_log.exists()


def test_importer_rejects_symlinked_output_root(tmp_path: Path) -> None:
    real_output = tmp_path / "real"
    real_output.mkdir()
    linked_output = tmp_path / "linked"
    linked_output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(SystemExit, match="symlinked Inspect output"):
        prepare_output(linked_output, replace=False)


def test_importer_rejects_symlinked_suite_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    outside = tmp_path / "outside"
    output_dir.mkdir()
    outside.mkdir()
    (output_dir / "policy").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SystemExit, match="symlinked Inspect suite"):
        safe_suite_directory(output_dir, "policy")
    assert list(outside.iterdir()) == []


def test_partial_strict_log_round_trip_skips_full_aggregate_assertion(
    tmp_path: Path,
) -> None:
    bundle = build_strict_vision_archive(
        DEFAULT_STRICT_VISION_RUNS["gemma4_31b"],
        limit=2,
    )
    log_path = write_inspect_archive_log(bundle, tmp_path, embed_images=False)
    verification = verify_inspect_archive_log(bundle, log_path)

    assert verification["samples"] == 2
    assert verification["source_scores_verified"] is False
    assert verification["scores"] == {
        "strict_exact": 1.0,
        "strict_json_valid": 1.0,
        "strict_protocol_exact": 1.0,
    }


def test_inspect_viz_rows_require_four_verified_matched_runs() -> None:
    models = (
        "deepseek/deepseek-v4-flash-vision-exp",
        "google/gemma-4-31b-it",
        "zai-org/glm-4.6v",
        "qwen/qwen3.8-max",
    )
    logs = [
        {
            "archive_id": f"run-{index}",
            "input_mode": "raw_image",
            "model_id": model,
            "inspect_model_id": f"archive/{model}",
            "log_path": f"run-{index}.eval",
            "expected_metrics": {
                "exact": 20 + index,
                "json_valid": 60,
                "protocol_exact": 10 + index,
                "requests": 60,
            },
            "verification": {
                "status": "success",
                "source_scores_verified": True,
                "scores": {
                    "strict_exact": (20 + index) / 60,
                    "strict_json_valid": 1.0,
                    "strict_protocol_exact": (10 + index) / 60,
                },
            },
        }
        for index, model in enumerate(models)
    ]

    rows = strict_vision_rows({"schema": "catan-inspect-archive/v1", "logs": logs})
    assert len(rows) == 4
    assert rows[1]["model_display_name"] == "Gemma 4 31B"
    assert rows[3]["exact_count"] == "23/60"

    logs[0]["inspect_model_id"] = "archive/wrong"
    with pytest.raises(ValueError, match="unexpected archived Inspect model ID"):
        strict_vision_rows({"schema": "catan-inspect-archive/v1", "logs": logs})
    logs[0]["inspect_model_id"] = f"archive/{logs[0]['model_id']}"

    logs[0]["expected_metrics"]["requests"] = 59
    with pytest.raises(ValueError, match="not the frozen 60-sample cohort"):
        strict_vision_rows({"schema": "catan-inspect-archive/v1", "logs": logs})
    logs[0]["expected_metrics"]["requests"] = 60

    logs[0]["verification"]["source_scores_verified"] = False
    with pytest.raises(ValueError, match="unverified strict-vision log"):
        strict_vision_rows({"schema": "catan-inspect-archive/v1", "logs": logs})

    logs[0]["verification"]["source_scores_verified"] = True
    logs[0]["model_id"] = logs[1]["model_id"]
    logs[0]["inspect_model_id"] = f"archive/{logs[0]['model_id']}"
    with pytest.raises(ValueError, match="duplicate strict-vision model ID"):
        strict_vision_rows({"schema": "catan-inspect-archive/v1", "logs": logs})
